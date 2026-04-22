"""Phase C training driver — LoRA finetune of BiomedParse v2 with aux heads.

Usage:
    python scripts/train_biomedparse_phase_c.py --config configs/phase_c_stage1.yaml

Implementation notes
--------------------
BP v2's ``forward_train`` in ``biomedparse_3D.py`` is ``NotImplementedError``
upstream, so we compose our own training forward that mirrors
``forward_eval`` but:
- accepts arbitrary batches of 3-channel slices (not a single 3D case)
- exposes ``multi_scale_features`` for the aux heads
- leaves the backbone → pixel_decoder → predictor → convolution_procedure
  chain otherwise identical to BP's eval path

LoRA is injected only into `pixel_decoder` (rank 16) and
`transformer_decoder` cross-attention (rank 32). Backbone + text encoder
stay frozen. See ``src/models/bp_lora.py`` for the injector.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# Make the repo root importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.refseg_phase_c import (  # noqa: E402
    RefSegPhaseCDataset,
    RefSegPhaseCCaseDataset,
    PosNegRatioSampler,
    collate_phase_c,
)
from src.losses.phase_c_losses import PhaseCLoss, PhaseCLossWeights  # noqa: E402
from src.models.aux_heads import PhaseCAuxHeads, pool_features  # noqa: E402
from src.models.bp_lora import (  # noqa: E402
    freeze_all,
    inject_lora,
    summarize_trainable,
    unfreeze_prefixes,
)


# ---------------------------------------------------------------------------
# BP loading (mirrors scripts/biomedparse_evaluate.py:load_model)
# ---------------------------------------------------------------------------

@contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def load_bp_model(bp_root: Path, ckpt: Path, device: torch.device):
    """Instantiate BiomedParse 3D via Hydra and load the v2 checkpoint."""
    import hydra
    from hydra import compose, initialize_config_dir

    bp_src = bp_root / "src"
    sys.path.insert(0, str(bp_src))
    cfg_dir = str((bp_root / "configs").resolve())

    with _chdir(bp_root):
        initialize_config_dir(config_dir=cfg_dir, job_name="phase_c_train",
                              version_base=None)
        cfg = compose(config_name="model/biomedparse_3D")
        model_cfg = cfg.model if "model" in cfg else cfg
        model = hydra.utils.instantiate(model_cfg, _convert_="object")
        model.load_pretrained(str(ckpt))

    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Training forward (replaces BP's broken forward_train)
# ---------------------------------------------------------------------------

def _tile_prompt(num_prompts: torch.Tensor, mask_features, multi_scale_features):
    P = int(num_prompts[0])
    if num_prompts.max() > num_prompts.min():
        mask_features = mask_features.repeat_interleave(num_prompts, dim=0)
        multi_scale_features = [
            f.repeat_interleave(num_prompts, dim=0) for f in multi_scale_features
        ]
    elif P != 1:
        # B samples with uniform P>1 prompts each — replicate the 2D/3D tile_feature
        B, C = mask_features.shape[:2]
        H, W = mask_features.shape[-2:]
        mask_features = mask_features.view(B, 1, C, H, W).expand(-1, P, -1, -1, -1).reshape(B * P, C, H, W)
        new_ms = []
        for f in multi_scale_features:
            Bf, Cf = f.shape[:2]; Hf, Wf = f.shape[-2:]
            new_ms.append(f.view(Bf, 1, Cf, Hf, Wf).expand(-1, P, -1, -1, -1).reshape(Bf * P, Cf, Hf, Wf))
        multi_scale_features = new_ms
    return mask_features, multi_scale_features


def bp_forward_train(
    bp_model,
    image_3c: torch.Tensor,
    text_list: list[str],
    aux_heads: PhaseCAuxHeads,
) -> dict:
    """Mirror of BP's eval path but with grads + per-sample prompts + aux heads.

    Args:
        image_3c: (B, 3, H_in, W_in) in [0, 1].
        text_list: list[str] of length B, one prompt per sample.
        aux_heads: PhaseCAuxHeads module.
    """
    device = image_3c.device

    # BP expects 0–255 range going through the pixel_mean/std registered buffer.
    image_batch = image_3c * 255.0
    image_batch = (image_batch - bp_model.pixel_mean.mean()) / bp_model.pixel_std.mean()

    # BP v2 native resolution is 512×512.
    if image_batch.shape[-1] != 512 or image_batch.shape[-2] != 512:
        image_batch = F.interpolate(
            image_batch, size=(512, 512), mode="bilinear", align_corners=False
        )

    sem = bp_model.sem_seg_head

    image_feat = bp_model.backbone(image_batch)

    # Encode each sample's prompt independently.
    # BP's process_multi_prompts splits on [SEP], so each string becomes 1 prompt
    # as long as we avoid [SEP] in the text (we don't insert it in training).
    prompt_feat = sem.encode_prompts(text=text_list, eval=True)

    mask_features, _, multi_scale_features = sem.pixel_decoder.forward_features(image_feat)

    num_prompts = prompt_feat["num_prompts"].to(mask_features.device)
    mask_features, multi_scale_features = _tile_prompt(
        num_prompts, mask_features, multi_scale_features
    )

    predictions = sem.predictor(
        x=multi_scale_features, mask_features=mask_features, extra=prompt_feat,
    )

    pooled = pool_features(multi_scale_features, class_emb=prompt_feat["class_emb"])
    aux_out = aux_heads(pooled)

    pred_gmasks = predictions["pred_gmasks"]
    if bp_model.convolute_outputs:
        image_rep = image_batch.repeat_interleave(num_prompts, dim=0)
        pred_gmasks = bp_model.convolution_procedure(image_rep, pred_gmasks)
    else:
        pred_gmasks = pred_gmasks.mean(dim=1, keepdim=True)

    return {
        "pred_gmasks": pred_gmasks,                       # (B*P, 1, 256, 256)
        "object_existence": predictions["object_existence"],  # (B*P, 1)
        "aux_out": aux_out,                                # dict of (B*P, ...)
        "pooled_feat": pooled,                             # (B*P, feat_dim)
    }


# ---------------------------------------------------------------------------
# Metric helpers (dev eval)
# ---------------------------------------------------------------------------

@torch.no_grad()
def quick_dev_metrics(
    bp_model, aux_heads, dev_loader, loss_fn, device, max_batches: int | None = None,
) -> dict:
    bp_model.eval()
    aux_heads.eval()
    agg = {"main_dice": [], "slice_exist_f1": [], "bbox_iou_pos": [],
           "centroid_err_norm": [], "z_range_recall": [], "existence_acc": []}

    se_tp = se_fp = se_fn = 0

    for bi, batch in enumerate(dev_loader):
        if max_batches is not None and bi >= max_batches:
            break
        image_3c = batch["image_3c"].to(device, non_blocking=True)
        text = batch["text"]
        gt_mask = batch["gt_mask"].to(device, non_blocking=True)

        out = bp_forward_train(bp_model, image_3c, text, aux_heads)
        pred_logit = out["pred_gmasks"].squeeze(1)
        gt_at_pred = F.interpolate(
            gt_mask.unsqueeze(1), size=pred_logit.shape[-2:], mode="nearest"
        ).squeeze(1)
        dice = _soft_dice_np(pred_logit, gt_at_pred)
        agg["main_dice"].extend(dice)

        # slice_exist F1
        se_logit = out["aux_out"]["slice_exist_logit"]
        se_pred = (torch.sigmoid(se_logit) > 0.5).long().cpu().numpy()
        se_gt = batch["slice_exist"].long().numpy()
        se_tp += int(((se_pred == 1) & (se_gt == 1)).sum())
        se_fp += int(((se_pred == 1) & (se_gt == 0)).sum())
        se_fn += int(((se_pred == 0) & (se_gt == 1)).sum())

        # existence accuracy
        ex_logit = out["aux_out"]["existence_logit"]
        ex_pred = (torch.sigmoid(ex_logit) > 0.5).long().cpu().numpy()
        ex_gt = batch["existence"].long().numpy()
        agg["existence_acc"].extend((ex_pred == ex_gt).astype(float).tolist())

        # bbox IoU on positive samples
        bbox_pred = out["aux_out"]["bbox_3d"].cpu().numpy()
        bbox_gt = batch["bbox_3d"].numpy()
        pos_mask = batch["existence"].numpy().astype(bool)
        for p, g, keep in zip(bbox_pred, bbox_gt, pos_mask):
            if not keep:
                continue
            agg["bbox_iou_pos"].append(_iou_3d(p, g))

        # centroid err
        ct_pred = out["aux_out"]["centroid_3d"].cpu().numpy()
        ct_gt = batch["centroid_3d"].numpy()
        diag = math.sqrt(3)
        for p, g, keep in zip(ct_pred, ct_gt, pos_mask):
            if not keep:
                continue
            agg["centroid_err_norm"].append(float(np.linalg.norm(p - g) / diag))

        # z_range recall: fraction of GT z-range covered by pred z-range
        zr_pred = out["aux_out"]["z_range"].cpu().numpy()
        zr_gt = batch["z_range"].numpy()
        for p, g, keep in zip(zr_pred, zr_gt, pos_mask):
            if not keep:
                continue
            agg["z_range_recall"].append(_interval_recall(p, g))

    se_prec = se_tp / max(1, se_tp + se_fp)
    se_rec = se_tp / max(1, se_tp + se_fn)
    se_f1 = 2 * se_prec * se_rec / max(1e-9, se_prec + se_rec)

    summary = {k: (float(np.mean(v)) if v else 0.0) for k, v in agg.items()}
    summary["slice_exist_f1"] = float(se_f1)
    return summary


def _soft_dice_np(pred_logit, gt):
    prob = torch.sigmoid(pred_logit)
    num = 2 * (prob * gt).sum(dim=(-1, -2))
    den = prob.sum(dim=(-1, -2)) + gt.sum(dim=(-1, -2)) + 1e-5
    return (num / den).cpu().numpy().tolist()


def _iou_3d(p, g):
    z1, y1, x1, z2, y2, x2 = p
    gz1, gy1, gx1, gz2, gy2, gx2 = g
    iz1 = max(z1, gz1); iy1 = max(y1, gy1); ix1 = max(x1, gx1)
    iz2 = min(z2, gz2); iy2 = min(y2, gy2); ix2 = min(x2, gx2)
    inter = max(0, iz2 - iz1) * max(0, iy2 - iy1) * max(0, ix2 - ix1)
    pv = max(0, z2 - z1) * max(0, y2 - y1) * max(0, x2 - x1)
    gv = max(0, gz2 - gz1) * max(0, gy2 - gy1) * max(0, gx2 - gx1)
    union = pv + gv - inter
    return inter / union if union > 1e-9 else 0.0


def _interval_recall(pred: np.ndarray, gt: np.ndarray) -> float:
    """1D interval recall: |[p1,p2] ∩ [g1,g2]| / |[g1,g2]|."""
    p1, p2 = float(pred[0]), float(pred[1])
    g1, g2 = float(gt[0]), float(gt[1])
    inter = max(0.0, min(p2, g2) - max(p1, g1))
    gl = max(1e-9, g2 - g1)
    return inter / gl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_split(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def build_model(cfg, device):
    bp_root = Path(cfg["model"]["bp_root"]).resolve()
    ckpt = Path(cfg["model"]["ckpt"]).resolve()
    bp_model = load_bp_model(bp_root, ckpt, device)

    freeze_all(bp_model)
    lora_cfg = cfg["model"]["lora"]
    n_pd = inject_lora(
        bp_model.sem_seg_head.pixel_decoder,
        target_patterns=tuple(lora_cfg["pixel_decoder"]["target_patterns"]),
        rank=int(lora_cfg["pixel_decoder"]["rank"]),
        alpha=int(lora_cfg["pixel_decoder"]["alpha"]),
        dropout=float(lora_cfg["pixel_decoder"].get("dropout", 0.0)),
    )
    n_td = inject_lora(
        bp_model.sem_seg_head.predictor,
        target_patterns=tuple(lora_cfg["transformer_decoder"]["target_patterns"]),
        rank=int(lora_cfg["transformer_decoder"]["rank"]),
        alpha=int(lora_cfg["transformer_decoder"]["alpha"]),
        dropout=float(lora_cfg["transformer_decoder"].get("dropout", 0.0)),
    )
    print(f"[LoRA] pixel_decoder wrapped={n_pd}  predictor wrapped={n_td}")

    unfreeze_prefixes_list = cfg["model"].get("unfreeze_prefixes", [])
    if unfreeze_prefixes_list:
        n_u = unfreeze_prefixes(bp_model, tuple(unfreeze_prefixes_list))
        print(f"[unfreeze] matched {n_u} parameters by prefix")

    print(summarize_trainable(bp_model, tag="bp_after_lora"))

    # Discover feat dim from a dummy forward at low cost
    feat_dim = cfg["model"]["aux_heads"].get("feat_dim")
    if feat_dim is None:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256, device=device)
            prompt_feat = bp_model.sem_seg_head.encode_prompts(
                text=["dummy prompt"], eval=True
            )
            img_n = (dummy * 255.0 - bp_model.pixel_mean.mean()) / bp_model.pixel_std.mean()
            img_n = F.interpolate(img_n, (512, 512), mode="bilinear", align_corners=False)
            feat = bp_model.backbone(img_n)
            _, _, msf = bp_model.sem_seg_head.pixel_decoder.forward_features(feat)
            pooled = pool_features(msf, class_emb=prompt_feat["class_emb"])
            feat_dim = int(pooled.shape[-1])
        print(f"[aux_heads] feat_dim auto = {feat_dim}")

    aux_heads = PhaseCAuxHeads(
        feat_dim=feat_dim,
        hidden_dim=int(cfg["model"]["aux_heads"]["hidden_dim"]),
        dropout=float(cfg["model"]["aux_heads"].get("dropout", 0.0)),
    ).to(device)

    return bp_model, aux_heads, feat_dim


def build_data(cfg):
    data_cfg = cfg["data"]
    splits_dir = Path(data_cfg["splits_dir"])
    train_cases = read_split(splits_dir / data_cfg["train_split"])
    dev_cases = read_split(splits_dir / data_cfg["dev_split"])

    train_ds = RefSegPhaseCDataset(
        train_cases, npy_root=data_cfg["npy_root"], aux_root=data_cfg["aux_root"],
        cache_arrays=bool(data_cfg.get("cache_arrays", True)),
    )
    dev_ds = RefSegPhaseCDataset(
        dev_cases, npy_root=data_cfg["npy_root"], aux_root=data_cfg["aux_root"],
        cache_arrays=bool(data_cfg.get("cache_arrays", True)),
    )

    n_samples = cfg["train"].get("max_slices_per_epoch") or len(train_ds)
    sampler = PosNegRatioSampler(
        train_ds,
        pos_ratio=float(data_cfg.get("pos_ratio", 0.7)),
        num_samples=int(n_samples),
        seed=int(cfg["train"].get("seed", 42)),
    )
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["train"]["batch_size"]),
        sampler=sampler, collate_fn=collate_phase_c,
        num_workers=int(cfg["train"].get("num_workers", 2)),
        pin_memory=True, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False, collate_fn=collate_phase_c,
        num_workers=int(cfg["train"].get("num_workers", 2)),
        pin_memory=True,
    )
    return train_loader, dev_loader, sampler


def build_optim(cfg, bp_model, aux_heads):
    lr_lora = float(cfg["train"]["lr_lora"])
    lr_aux = float(cfg["train"]["lr_aux_heads"])
    lr_unfrozen = float(cfg["train"].get("lr_unfrozen", lr_lora))
    wd = float(cfg["train"].get("weight_decay", 0.0))

    lora_params = []
    other_bp_params = []
    for n, p in bp_model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_A" in n or "lora_B" in n:
            lora_params.append(p)
        else:
            other_bp_params.append(p)
    param_groups = [
        {"params": lora_params, "lr": lr_lora, "weight_decay": wd, "name": "lora"},
        {"params": list(aux_heads.parameters()), "lr": lr_aux, "weight_decay": wd, "name": "aux"},
    ]
    if other_bp_params:
        param_groups.append(
            {"params": other_bp_params, "lr": lr_unfrozen, "weight_decay": wd, "name": "unfrozen"}
        )
    optim = torch.optim.AdamW(param_groups)
    return optim


def cosine_lr(optim, epoch: int, total_epochs: int, warmup_epochs: int):
    for pg in optim.param_groups:
        base_lr = pg.get("_base_lr", pg["lr"])
        pg.setdefault("_base_lr", base_lr)
        if epoch < warmup_epochs:
            scale = (epoch + 1) / max(1, warmup_epochs)
        else:
            t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            scale = 0.5 * (1.0 + math.cos(math.pi * t))
        pg["lr"] = base_lr * scale


def save_ckpt(path: Path, bp_model, aux_heads, extra: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only persist trainable params (LoRA + aux) — saves space, avoids copying BP ckpt.
    state = {
        "lora": {n: p.detach().cpu() for n, p in bp_model.named_parameters() if p.requires_grad},
        "aux_heads": aux_heads.state_dict(),
        "extra": extra,
    }
    torch.save(state, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default="")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(int(cfg["train"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["logging"]["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve paths relative to the repo root so the script is cwd-agnostic.
    for k in ("bp_root", "ckpt"):
        cfg["model"][k] = str((REPO_ROOT / cfg["model"][k]).resolve())
    for k in ("npy_root", "aux_root", "splits_dir"):
        cfg["data"][k] = str((REPO_ROOT / cfg["data"][k]).resolve())

    bp_model, aux_heads, feat_dim = build_model(cfg, device)
    train_loader, dev_loader, sampler = build_data(cfg)

    # Optional resume
    resume_path = args.resume or cfg.get("resume_from", "")
    if resume_path:
        resume_path = str((REPO_ROOT / resume_path).resolve())
        print(f"[resume] loading trainable weights from {resume_path}")
        state = torch.load(resume_path, map_location=device)
        for n, p in bp_model.named_parameters():
            if n in state["lora"]:
                p.data.copy_(state["lora"][n].to(p.device))
        aux_heads.load_state_dict(state["aux_heads"])

    loss_weights = PhaseCLossWeights(**cfg["loss"])
    loss_fn = PhaseCLoss(loss_weights).to(device)

    optim = build_optim(cfg, bp_model, aux_heads)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"].get("use_amp", True)))

    epochs = int(cfg["train"]["epochs"])
    warmup = int(cfg["train"].get("warmup_epochs", 1))
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    log_every = int(cfg["logging"].get("log_every", 20))
    best_metric = cfg["train"].get("best_metric", "dev_main_dice")

    best_val = -1.0
    metrics_history = []

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        bp_model.train()
        aux_heads.train()
        cosine_lr(optim, epoch, epochs, warmup)
        t_epoch = time.time()
        optim.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            image_3c = batch["image_3c"].to(device, non_blocking=True)
            text = batch["text"]
            gt_mask = batch["gt_mask"].to(device, non_blocking=True)
            aux_gt = {
                "existence": batch["existence"].to(device),
                "slice_exist": batch["slice_exist"].to(device),
                "bbox_3d": batch["bbox_3d"].to(device),
                "centroid_3d": batch["centroid_3d"].to(device),
                "z_range": batch["z_range"].to(device),
            }
            existence_mask = aux_gt["existence"]

            with torch.amp.autocast("cuda", enabled=bool(cfg["train"].get("use_amp", True))):
                out = bp_forward_train(bp_model, image_3c, text, aux_heads)
                pred_logit = out["pred_gmasks"].squeeze(1)
                gt_resized = F.interpolate(
                    gt_mask.unsqueeze(1), size=pred_logit.shape[-2:], mode="nearest",
                ).squeeze(1)
                L, logs = loss_fn(
                    pred_seg_logit=pred_logit,
                    gt_mask=gt_resized,
                    aux_out=out["aux_out"],
                    aux_gt=aux_gt,
                    existence_mask=existence_mask,
                    epoch=epoch,
                )
                L = L / grad_accum

            scaler.scale(L).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optim.param_groups for p in g["params"]], 1.0
                )
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            if step % log_every == 0:
                lrs = " ".join(f"{pg.get('name','?')}={pg['lr']:.2e}" for pg in optim.param_groups)
                print(
                    f"ep{epoch:02d} step{step:04d}/{len(train_loader):04d}  "
                    f"loss={logs['loss/total']:.4f}  "
                    f"main={logs['loss/main']:.4f}  "
                    f"sex={logs['loss/slice_exist']:.4f}  "
                    f"bb={logs['loss/bbox']:.4f}  ramp={logs['loss/ramp']:.2f}  {lrs}",
                    flush=True,
                )

        print(f"[ep{epoch:02d}] epoch_time={time.time() - t_epoch:.1f}s — dev eval …", flush=True)
        dev = quick_dev_metrics(bp_model, aux_heads, dev_loader, loss_fn, device)
        dev["epoch"] = epoch
        metrics_history.append(dev)
        print(f"[ep{epoch:02d} dev] " + "  ".join(f"{k}={v:.4f}" for k, v in dev.items() if k != "epoch"),
              flush=True)
        with open(out_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(dev) + "\n")

        val = dev.get(best_metric.replace("dev_", ""), 0.0)
        if val > best_val:
            best_val = val
            save_ckpt(out_dir / "best.pt", bp_model, aux_heads,
                      {"epoch": epoch, "metrics": dev, "config": cfg})
            print(f"[ep{epoch:02d}] ✓ new best {best_metric}={val:.4f} saved", flush=True)
        if cfg["train"].get("save_every_epoch", True):
            save_ckpt(out_dir / f"ep{epoch:02d}.pt", bp_model, aux_heads,
                      {"epoch": epoch, "metrics": dev})

    print(f"[done] best {best_metric}={best_val:.4f}")


if __name__ == "__main__":
    main()
