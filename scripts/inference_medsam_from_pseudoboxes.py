#!/usr/bin/env python3
"""
Phase B stage 2: BiomedParse pseudo-box -> MedSAM refinement.

Reads:
  results/12_biomedparse_pseudoboxes<suffix>.json
    Shape: {case_id: {mask_id_str: {mode: {prompt, boxes, gt_pos, pred_pos}}}}
    boxes: list of length D, entry is [x1, y1, x2, y2] in 256x256 space, or null.

  data/M3D_RefSeg_npy/<case_id>/{ct.npy, mask.npy}
    ct.npy  (1, 32, 256, 256) float32 in [0, 1]
    mask.npy (1, 32, 256, 256) label-coded

Writes:
  results/12_biomedparse_medsam_raw<suffix>.csv
  results/12_biomedparse_medsam_structured<suffix>.csv
  Each row: case, mask_id, mode, prompt, dice, iou, gt_pos, pred_pos, slices_used

Compares with (a) BiomedParse direct (from the pseudo-box CSVs already on disk)
and (b) oracle bbox -> MedSAM (results/12_medsam_oracle_bbox.csv if present).
Together they answer 新计划.md Phase B question: does BiomedParse's coarse
grounding already give MedSAM a usable box?
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage import transform as sk_transform
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "MedSAM"))
from segment_anything import sam_model_registry  # noqa: E402


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return float(2.0 * (pred & gt).sum() / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return float((pred & gt).sum() / union)


def load_volume(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def prepare_slice_for_medsam(ct_slice: np.ndarray) -> np.ndarray:
    """(H,W) float32 [0,1] -> (1024,1024,3) float32 [0,1]."""
    img_3c = np.repeat(ct_slice[:, :, None], 3, axis=-1)
    img_1024 = sk_transform.resize(
        img_3c, (1024, 1024), order=3,
        preserve_range=True, anti_aliasing=True,
    ).astype(np.float32)
    vmin, vmax = img_1024.min(), img_1024.max()
    if vmax - vmin > 1e-8:
        img_1024 = (img_1024 - vmin) / (vmax - vmin)
    return img_1024


@torch.no_grad()
def encode_slice(model, ct_slice: np.ndarray, device) -> torch.Tensor:
    img_1024 = prepare_slice_for_medsam(ct_slice)
    img_tensor = (
        torch.tensor(img_1024).half().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    with torch.amp.autocast("cuda"):
        emb = model.image_encoder(img_tensor)
    del img_tensor
    return emb


@torch.no_grad()
def medsam_decode(model, img_embed, box_1024, H, W, device) -> np.ndarray:
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=device)
    if box_torch.ndim == 2:
        box_torch = box_torch[:, None, :]
    sparse_emb, dense_emb = model.prompt_encoder(
        points=None, boxes=box_torch, masks=None,
    )
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=img_embed.float(),
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(H, W), mode="bilinear", align_corners=False,
    )
    return (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


def run_case(model, device, case_id: str, case_bboxes: dict,
             npy_root: Path, box_space: int = 256):
    """For one case, iterate (mask_id, mode) pairs and refine via MedSAM.

    Returns list of result rows.
    """
    ct_path = npy_root / case_id / "ct.npy"
    mask_path = npy_root / case_id / "mask.npy"
    if not ct_path.exists() or not mask_path.exists():
        return []
    ct_vol = load_volume(ct_path).astype(np.float32)        # (D, H, W)
    mask_vol = np.rint(load_volume(mask_path)).astype(np.int32)
    D, H, W = ct_vol.shape

    # Group work by slice so we encode each needed slice only once per case
    # Structure: slice_needs[z] = list of (mask_id, mode, [x1,y1,x2,y2])
    slice_needs = {z: [] for z in range(D)}
    work_items = []   # list of (mid, mode, gt_3d, pred_3d_placeholder)
    for mid_str, modes in case_bboxes.items():
        mid = int(mid_str)
        gt_3d = (mask_vol == mid).astype(np.uint8)
        for mode, entry in modes.items():
            boxes = entry["boxes"]
            if len(boxes) != D:
                # Defensive: if eval was run on a volume with different D, skip
                print(f"[warn] {case_id}/{mid}/{mode} boxes len={len(boxes)} "
                      f"!= D={D}, skipping", flush=True)
                continue
            pred_3d = np.zeros_like(gt_3d, dtype=np.uint8)
            slot_idx = len(work_items)
            work_items.append(dict(
                mid=mid, mode=mode, prompt=entry["prompt"],
                gt_3d=gt_3d, pred_3d=pred_3d,
                gt_pos=int(entry.get("gt_pos", gt_3d.sum())),
                bp_pred_pos=int(entry.get("pred_pos", 0)),
            ))
            for z, b in enumerate(boxes):
                if b is None:
                    continue
                slice_needs[z].append((slot_idx, b))

    # Per-slice encode -> decode loop
    scale = 1024.0 / box_space
    for z in sorted(z for z, v in slice_needs.items() if v):
        emb = encode_slice(model, ct_vol[z], device)
        for slot_idx, bbox in slice_needs[z]:
            x1, y1, x2, y2 = bbox
            # pseudo-box may have x2<=x1 or tiny size for degenerate cases;
            # skip those since MedSAM prompt encoder expects positive box area
            if x2 <= x1 or y2 <= y1:
                continue
            box_1024 = np.array(
                [x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                dtype=float,
            )[None, :]
            pred_slice = medsam_decode(model, emb, box_1024, H, W, device)
            work_items[slot_idx]["pred_3d"][z] = np.maximum(
                work_items[slot_idx]["pred_3d"][z], pred_slice
            )
        del emb
        torch.cuda.empty_cache()

    rows = []
    for w in work_items:
        gt = w["gt_3d"]
        pred = w["pred_3d"]
        d = dice_score(pred, gt)
        iou = iou_score(pred, gt)
        # slices_used = number of slices with at least one input bbox
        n_slices_boxed = sum(
            1 for z, items in slice_needs.items()
            if any(work_items[si]["mid"] == w["mid"]
                   and work_items[si]["mode"] == w["mode"]
                   for si, _ in items)
        )
        rows.append(dict(
            case=case_id, mask_id=w["mid"], mode=w["mode"],
            prompt=w["prompt"], dice=d, iou=iou,
            gt_pos=int(gt.sum()), pred_pos=int(pred.sum()),
            bp_pred_pos=w["bp_pred_pos"],
            slices_used=n_slices_boxed,
        ))
    return rows


def main():
    parser = argparse.ArgumentParser("BiomedParse pseudo-box -> MedSAM refinement")
    parser.add_argument("--pseudobox_json", required=True,
                        help="results/12_biomedparse_pseudoboxes<suffix>.json")
    parser.add_argument("--npy_root", default="data/M3D_RefSeg_npy")
    parser.add_argument("--checkpoint",
                        default="third_party/MedSAM/work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--tag", default="",
                        help="CSV filename suffix, e.g. 'bypass'")
    parser.add_argument("--box_space", type=int, default=256,
                        help="coordinate space of input bboxes (ct.npy H=W=256)")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.pseudobox_json, "r", encoding="utf-8") as f:
        all_bboxes = json.load(f)

    case_ids = sorted(all_bboxes.keys())
    if args.max_cases > 0:
        case_ids = case_ids[:args.max_cases]
    print(f"[+] {len(case_ids)} cases from {args.pseudobox_json}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[+] Loading MedSAM from {args.checkpoint}")
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    model = model.to(device)
    model.image_encoder = model.image_encoder.half()
    model.eval()
    print(f"[+] MedSAM ready. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    raw_rows, struct_rows = [], []
    t0 = time.time()
    for i, case_id in enumerate(tqdm(case_ids, desc="Cases"), 1):
        case_bboxes = all_bboxes[case_id]
        rows = run_case(model, device, case_id, case_bboxes,
                        Path(args.npy_root), box_space=args.box_space)
        for r in rows:
            (raw_rows if r["mode"] == "raw" else struct_rows).append(r)

        if i % 10 == 0 or i == len(case_ids):
            raw_d = np.array([r["dice"] for r in raw_rows]) if raw_rows else np.array([0.0])
            st_d = np.array([r["dice"] for r in struct_rows]) if struct_rows else np.array([0.0])
            raw_empty = sum(1 for r in raw_rows if r["pred_pos"] == 0)
            st_empty = sum(1 for r in struct_rows if r["pred_pos"] == 0)
            print(f"[{i}/{len(case_ids)}] raw_mean={raw_d.mean():.4f} "
                  f"(empty {raw_empty}/{len(raw_rows)})  "
                  f"struct_mean={st_d.mean():.4f} (empty {st_empty}/{len(struct_rows)})  "
                  f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    raw_csv = out_dir / f"12_biomedparse_medsam_raw{suffix}.csv"
    st_csv = out_dir / f"12_biomedparse_medsam_structured{suffix}.csv"
    pd.DataFrame(raw_rows).to_csv(raw_csv, index=False)
    pd.DataFrame(struct_rows).to_csv(st_csv, index=False)

    for name, rows, path in [
        ("raw", raw_rows, raw_csv),
        ("structured", struct_rows, st_csv),
    ]:
        if not rows:
            print(f"  [{name}] no rows")
            continue
        d = np.array([r["dice"] for r in rows])
        n_empty = sum(1 for r in rows if r["pred_pos"] == 0)
        n_hit = int((d >= 0.5).sum())
        n_any_box = sum(1 for r in rows if r["slices_used"] > 0)
        print(f"  [{name}] n={len(d)}  mean={d.mean():.4f}  "
              f"median={np.median(d):.4f}  "
              f"Dice>=0.5: {n_hit}/{len(d)} ({100*n_hit/len(d):.1f}%)  "
              f"any_box: {n_any_box}/{len(d)}  "
              f"empty_pred: {n_empty}/{len(d)}  -> {path}")


if __name__ == "__main__":
    main()
