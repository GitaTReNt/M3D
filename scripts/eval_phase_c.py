"""Phase C evaluation — dev / test split.

Runs the finetuned BP (base + LoRA + aux heads) over each (case, mask_id)
in the target split and produces:

    results/phase_c/<stage>/bp_direct_<split>.csv
        per-(case, mask) Dice of BP direct (pred mask vs GT)

    results/phase_c/<stage>/pseudoboxes_<split>.json
        per-slice [x1,y1,x2,y2] boxes — consumed by
        ``scripts/inference_medsam_from_pseudoboxes.py`` to rerun the
        BP→MedSAM arm.

    results/phase_c/<stage>/aux_metrics_<split>.csv
        case-level aux metrics: existence_pred, slice_exist_f1 per case,
        bbox_iou_3d, centroid_err_norm, z_range_recall.

To reconstruct the Phase B 3-way table on the finetuned model, run:

    python scripts/inference_medsam_from_pseudoboxes.py \
        --pseudobox_json results/phase_c/stage1/pseudoboxes_dev.json
    python scripts/analyze_phase_b.py \
        --bp_direct_raw   results/phase_c/stage1/bp_direct_dev.csv \
        --bp_medsam_raw   results/phase_c/stage1/12_medsam_from_bp_pseudobox.csv \
        --out_dir         results/phase_c/stage1/analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.biomedparse_evaluate import (  # noqa: E402  (reuse helpers)
    extract_per_slice_boxes,
    dice_score,
)
from scripts.train_biomedparse_phase_c import (  # noqa: E402
    bp_forward_train,
    build_model,
    read_split,
)
from src.datasets.refseg_phase_c import RefSegPhaseCCaseDataset  # noqa: E402


def case_aux_from_slices(bp_model, aux_heads, ct_volume, text, device, slice_batch: int = 4):
    """Run the per-slice forward for every slice of a case and aggregate aux heads.

    Returns a dict of numpy arrays.
    """
    D, H, W = ct_volume.shape
    # pack 3-channel per-slice: [prev, curr, next]
    prev = np.concatenate([ct_volume[:1], ct_volume[:-1]], axis=0)
    nxt = np.concatenate([ct_volume[1:], ct_volume[-1:]], axis=0)
    image_3c = np.stack([prev, ct_volume, nxt], axis=1)  # (D, 3, H, W)

    ex_logits, sex_logits, bbox, ct3, zr = [], [], [], [], []
    t = torch.from_numpy(image_3c).float().to(device)
    for i in range(0, D, slice_batch):
        b = t[i : i + slice_batch]
        text_b = [text] * b.shape[0]
        out = bp_forward_train(bp_model, b, text_b, aux_heads)
        a = out["aux_out"]
        ex_logits.append(a["existence_logit"].detach().cpu().numpy())
        sex_logits.append(a["slice_exist_logit"].detach().cpu().numpy())
        bbox.append(a["bbox_3d"].detach().cpu().numpy())
        ct3.append(a["centroid_3d"].detach().cpu().numpy())
        zr.append(a["z_range"].detach().cpu().numpy())

    return {
        "existence_logit_per_slice": np.concatenate(ex_logits, axis=0),
        "slice_exist_logit": np.concatenate(sex_logits, axis=0),
        "bbox_3d_per_slice": np.concatenate(bbox, axis=0),
        "centroid_3d_per_slice": np.concatenate(ct3, axis=0),
        "z_range_per_slice": np.concatenate(zr, axis=0),
    }


def aggregate_case(aux_per_slice: dict, slice_exist_gt: np.ndarray) -> dict:
    """Aggregate per-slice aux predictions into case-level."""
    sp = 1 / (1 + np.exp(-aux_per_slice["slice_exist_logit"]))
    pos_mask = sp > 0.5
    if pos_mask.any():
        bbox_3d = aux_per_slice["bbox_3d_per_slice"][pos_mask].mean(axis=0)
        centroid_3d = aux_per_slice["centroid_3d_per_slice"][pos_mask].mean(axis=0)
        z_range = aux_per_slice["z_range_per_slice"][pos_mask].mean(axis=0)
    else:
        bbox_3d = aux_per_slice["bbox_3d_per_slice"].mean(axis=0)
        centroid_3d = aux_per_slice["centroid_3d_per_slice"].mean(axis=0)
        z_range = aux_per_slice["z_range_per_slice"].mean(axis=0)

    existence = float(aux_per_slice["existence_logit_per_slice"].max())
    existence_prob = 1 / (1 + np.exp(-existence))

    return {
        "existence_prob": float(existence_prob),
        "slice_exist_prob": sp,
        "bbox_3d": bbox_3d,
        "centroid_3d": centroid_3d,
        "z_range": z_range,
    }


def run_bp_case(bp_model, ct_volume: np.ndarray, text: str, device,
                slice_batch_size: int, score_threshold: float, do_nms: bool) -> dict:
    """Run BP's native forward_eval on a full case; returns pred_3d_bin + logits volume."""
    # Mirror of scripts/biomedparse_evaluate.run_one, but without process_input
    # padding (our ct is already square 256).
    imgs_t = torch.from_numpy(ct_volume * 255.0).float()  # (D, H, W)
    # expand to expected (1, 1, D, H, W) — BP will pack to 3-channel internally.
    imgs_t = imgs_t.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    imgs_t = imgs_t.to(device)

    with torch.no_grad():
        out = bp_model(
            {"image": imgs_t.squeeze(0).int(), "text": [text]},
            mode="eval", slice_batch_size=slice_batch_size,
        )
    mp = out["predictions"]["pred_gmasks"]         # (N, D, h, w) logits
    oe = out["predictions"]["object_existence"]     # (N, D)

    from inference import postprocess  # BP helper
    D, H, W = ct_volume.shape
    # Upsample to native resolution for Dice/bbox on pred
    mp = F.interpolate(mp, size=(H, W), mode="bicubic", align_corners=False, antialias=True)
    pred_bin = postprocess(mp, oe, threshold=score_threshold, do_nms=do_nms)  # (N, D, H, W)
    pred_3d = (pred_bin[0] > 0.5).cpu().numpy().astype(np.uint8)
    return {"pred_3d_bin": pred_3d, "logits": mp[0].detach().cpu().numpy(), "existence": oe[0].detach().cpu().numpy()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True, help="Phase C trainable-weights ckpt (LoRA+aux)")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--max_cases", type=int, default=0)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve paths
    for k in ("bp_root", "ckpt"):
        cfg["model"][k] = str((REPO_ROOT / cfg["model"][k]).resolve())
    for k in ("npy_root", "aux_root", "splits_dir"):
        cfg["data"][k] = str((REPO_ROOT / cfg["data"][k]).resolve())
    out_dir = Path(cfg["logging"]["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_file = Path(cfg["data"]["splits_dir"]) / f"phase_c_{args.split}.txt"
    cases = read_split(split_file)
    print(f"[eval] split={args.split}  n_cases={len(cases)}")

    # Load base model + LoRA, then load finetuned trainable weights
    bp_model, aux_heads, _ = build_model(cfg, device)
    state = torch.load(args.ckpt, map_location=device)
    for n, p in bp_model.named_parameters():
        if n in state["lora"]:
            p.data.copy_(state["lora"][n].to(p.device))
    aux_heads.load_state_dict(state["aux_heads"])
    bp_model.eval(); aux_heads.eval()

    eval_cfg = cfg.get("eval", {})
    slice_batch_size = int(eval_cfg.get("slice_batch_size", 4))
    score_thr = float(eval_cfg.get("score_threshold", 0.5))
    mask_thr = float(eval_cfg.get("mask_binarize_thresh", 0.5))
    margin = int(eval_cfg.get("bbox_margin", 5))
    cc_strategy = str(eval_cfg.get("cc_strategy", "largest_cc"))

    case_ds = RefSegPhaseCCaseDataset(
        cases, npy_root=cfg["data"]["npy_root"], aux_root=cfg["data"]["aux_root"],
    )

    rows, aux_rows = [], []
    pseudoboxes: dict = {}

    n_total = len(case_ds) if args.max_cases <= 0 else min(args.max_cases, len(case_ds))
    for i in range(n_total):
        item = case_ds[i]
        cid, mid = item["case_id"], item["mask_id"]
        ct_vol = item["ct"]  # (D, H, W) in [0,1]
        gt_3d = item["gt_mask_3d"]  # (D, H, W) uint8
        text = item["text"]
        aux_gt = item["aux"]
        D, H, W = ct_vol.shape

        # BP direct
        with torch.no_grad():
            bp_out = run_bp_case(
                bp_model, ct_vol, text, device,
                slice_batch_size=slice_batch_size,
                score_threshold=score_thr, do_nms=True,
            )
        pred_3d = bp_out["pred_3d_bin"]  # (D, H, W)

        dice = dice_score(pred_3d.astype(bool), gt_3d.astype(bool))
        rows.append({
            "case_id": cid, "mask_id": mid, "text": text,
            "dice": dice,
            "pred_pos": int(pred_3d.sum()), "gt_pos": int(gt_3d.sum()),
            "D": D, "H": H, "W": W,
        })

        # Pseudo-boxes for MedSAM rerun
        boxes = extract_per_slice_boxes(
            pred_3d, margin=margin,
            largest_cc_only=(cc_strategy == "largest_cc"),
        )
        pseudoboxes.setdefault(cid, {}).setdefault(str(mid), {})["raw"] = {
            "prompt": text, "boxes": boxes,
            "gt_pos": int(gt_3d.sum()), "pred_pos": int(pred_3d.sum()),
        }

        # Aux metrics (use per-slice forward)
        with torch.no_grad():
            aux_per_slice = case_aux_from_slices(
                bp_model, aux_heads, ct_vol, text, device, slice_batch=slice_batch_size,
            )
        agg = aggregate_case(aux_per_slice, np.array(aux_gt["slice_exist"]))

        sex_pred_bin = (agg["slice_exist_prob"] > 0.5).astype(int)
        sex_gt = np.array(aux_gt["slice_exist"], dtype=int)
        tp = int(((sex_pred_bin == 1) & (sex_gt == 1)).sum())
        fp = int(((sex_pred_bin == 1) & (sex_gt == 0)).sum())
        fn = int(((sex_pred_bin == 0) & (sex_gt == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)

        # bbox IoU (3D)
        from scripts.train_biomedparse_phase_c import _iou_3d, _interval_recall
        bbox_pred = agg["bbox_3d"]
        bbox_gt = np.array(aux_gt["bbox_3d"])
        iou3d = _iou_3d(bbox_pred, bbox_gt) if aux_gt["existence"] else float("nan")

        ct3_pred = agg["centroid_3d"]
        ct3_gt = np.array(aux_gt["centroid_3d"])
        diag = float(np.sqrt(3))
        ct_err = float(np.linalg.norm(ct3_pred - ct3_gt) / diag) if aux_gt["existence"] else float("nan")

        zr_pred = agg["z_range"]
        zr_gt = np.array(aux_gt["z_range"])
        zr_rec = _interval_recall(zr_pred, zr_gt) if aux_gt["existence"] else float("nan")

        aux_rows.append({
            "case_id": cid, "mask_id": mid,
            "existence_gt": int(aux_gt["existence"]),
            "existence_prob": agg["existence_prob"],
            "slice_exist_f1": f1,
            "bbox_iou_3d": iou3d,
            "centroid_err_norm": ct_err,
            "z_range_recall": zr_rec,
        })

        if (i + 1) % 10 == 0 or i == n_total - 1:
            d_mean = float(np.mean([r["dice"] for r in rows]))
            print(f"[{i + 1}/{n_total}] {cid}/{mid}  dice={dice:.4f}  "
                  f"running_mean={d_mean:.4f}  sex_f1={f1:.3f}  iou3d={iou3d:.3f}",
                  flush=True)

    # Save outputs
    direct_csv = out_dir / f"bp_direct_{args.split}.csv"
    pd.DataFrame(rows).to_csv(direct_csv, index=False)
    aux_csv = out_dir / f"aux_metrics_{args.split}.csv"
    pd.DataFrame(aux_rows).to_csv(aux_csv, index=False)

    pb_json = out_dir / f"pseudoboxes_{args.split}.json"
    with open(pb_json, "w", encoding="utf-8") as f:
        json.dump(pseudoboxes, f)

    print(f"[done] {direct_csv}")
    print(f"[done] {aux_csv}")
    print(f"[done] {pb_json}")
    d = pd.DataFrame(rows)["dice"]
    print(f"       mean dice = {d.mean():.4f}   median = {d.median():.4f}")


if __name__ == "__main__":
    main()
