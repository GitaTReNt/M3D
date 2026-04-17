#!/usr/bin/env python3
"""
MedSAM inference with inference-time optimizations (no retraining).

Track A: Oracle bbox improvements targeting over-segmentation.
Optimizations:
  1. Adaptive bbox margin (proportional to target size, not fixed 5px)
  2. Higher logit threshold (0.6/0.65/0.7 instead of 0.5)
  3. Otsu adaptive threshold per-slice
  4. Morphological erosion post-processing
  5. Bbox perturbation ensemble (jittered boxes → soft vote)
  6. Per-slice confidence gating (suppress low-confidence slices)

Usage:
  python inference_medsam_optimized.py \
      --npy_root D:/M3D/M3D_RefSeg_npy \
      --checkpoint D:/M3D/MedSAM/work_dir/MedSAM_finetuned/medsam_vit_b.pth \
      --max_cases 30 --device cuda:0 \
      --tricks adaptive_margin,threshold_0.65,erosion
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
from scipy import ndimage
from skimage import transform as sk_transform
from skimage.filters import threshold_otsu
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "MedSAM"))
from segment_anything import sam_model_registry


# ============================================================
# Metrics
# ============================================================

def dice_score(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return float(2.0 * inter / denom)

def iou_score(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return float(inter / union)


# ============================================================
# Data loading
# ============================================================

def load_volume(path):
    arr = np.load(str(path))
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


# ============================================================
# Bbox generation
# ============================================================

def get_bbox_2d_fixed(mask_2d, margin=5):
    """Fixed-margin bbox (baseline)."""
    ys, xs = np.where(mask_2d > 0)
    if len(ys) == 0:
        return None
    H, W = mask_2d.shape
    return [
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(W, int(xs.max()) + margin),
        min(H, int(ys.max()) + margin),
    ]

def get_bbox_2d_adaptive(mask_2d, base_ratio=0.1, min_margin=2, max_margin=15):
    """Adaptive-margin bbox: margin proportional to target size."""
    ys, xs = np.where(mask_2d > 0)
    if len(ys) == 0:
        return None
    H, W = mask_2d.shape
    bbox_w = int(xs.max()) - int(xs.min())
    bbox_h = int(ys.max()) - int(ys.min())
    margin = int(base_ratio * max(bbox_w, bbox_h))
    margin = max(min_margin, min(margin, max_margin))
    return [
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(W, int(xs.max()) + margin),
        min(H, int(ys.max()) + margin),
    ]

def jitter_bbox(bbox, H, W, jitter_px=3):
    """Generate a randomly jittered bbox."""
    offsets = np.random.randint(-jitter_px, jitter_px + 1, size=4)
    return [
        max(0, bbox[0] + offsets[0]),
        max(0, bbox[1] + offsets[1]),
        min(W, bbox[2] + offsets[2]),
        min(H, bbox[3] + offsets[3]),
    ]


# ============================================================
# MedSAM inference
# ============================================================

def prepare_slice_for_medsam(ct_slice):
    img_3c = np.repeat(ct_slice[:, :, None], 3, axis=-1)
    img_1024 = sk_transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True,
    ).astype(np.float32)
    vmin, vmax = img_1024.min(), img_1024.max()
    if vmax - vmin > 1e-8:
        img_1024 = (img_1024 - vmin) / (vmax - vmin)
    return img_1024

@torch.no_grad()
def encode_slice(model, ct_slice, device):
    img_1024 = prepare_slice_for_medsam(ct_slice)
    img_t = torch.tensor(img_1024).half().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda"):
        emb = model.image_encoder(img_t)
    del img_t
    return emb

@torch.no_grad()
def medsam_infer_soft(model, img_embed, box_1024, H, W, device):
    """Run MedSAM decoder, return SOFT sigmoid probability map (not binary)."""
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=device)
    if box_torch.ndim == 2:
        box_torch = box_torch[:, None, :]
    sparse_emb, dense_emb = model.prompt_encoder(
        points=None, boxes=box_torch, masks=None,
    )
    low_res_logits, iou_pred = model.mask_decoder(
        image_embeddings=img_embed.float(),
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )
    prob_map = torch.sigmoid(low_res_logits)
    prob_map = F.interpolate(
        prob_map, size=(H, W), mode="bilinear", align_corners=False,
    )
    return prob_map.squeeze().cpu().numpy(), iou_pred.squeeze().cpu().numpy()


# ============================================================
# Post-processing strategies
# ============================================================

def threshold_fixed(prob_map, thresh=0.5):
    return (prob_map > thresh).astype(np.uint8)

def threshold_otsu_auto(prob_map):
    """Otsu's method on the probability map. Falls back to 0.5 if uniform."""
    try:
        t = threshold_otsu(prob_map)
        # Otsu can give bad results if nearly all values are similar
        if t < 0.2 or t > 0.9:
            t = 0.5
    except ValueError:
        t = 0.5
    return (prob_map > t).astype(np.uint8)

def apply_erosion(mask_2d, iterations=1):
    """Binary erosion to combat over-segmentation."""
    if mask_2d.sum() == 0:
        return mask_2d
    struct = ndimage.generate_binary_structure(2, 1)  # cross-shaped
    eroded = ndimage.binary_erosion(mask_2d, structure=struct, iterations=iterations)
    return eroded.astype(np.uint8)


# ============================================================
# Main inference with all tricks
# ============================================================

def run_optimized_inference(
    model, device, npy_root, out_dir, max_cases=0,
    tricks=None, bbox_margin=5,
):
    if tricks is None:
        tricks = set()
    else:
        tricks = set(tricks)

    # Parse threshold from tricks
    fixed_thresh = 0.5
    for t in tricks:
        if t.startswith("threshold_"):
            try:
                fixed_thresh = float(t.split("_")[1])
            except (ValueError, IndexError):
                pass

    use_adaptive_margin = "adaptive_margin" in tricks
    use_otsu = "otsu" in tricks
    use_erosion = "erosion" in tricks
    use_ensemble = "ensemble" in tricks
    use_confidence_gate = "confidence_gate" in tricks
    erosion_iters = 1
    ensemble_n = 5
    ensemble_jitter = 3
    confidence_min = 0.55  # minimum mean prob within predicted mask

    print(f"Tricks: {tricks if tricks else 'none (baseline)'}")
    print(f"  Fixed threshold: {fixed_thresh}")
    print(f"  Adaptive margin: {use_adaptive_margin}")
    print(f"  Otsu threshold: {use_otsu}")
    print(f"  Erosion (iters={erosion_iters}): {use_erosion}")
    print(f"  Bbox ensemble (n={ensemble_n}, jitter={ensemble_jitter}px): {use_ensemble}")
    print(f"  Confidence gate (min={confidence_min}): {use_confidence_gate}")

    results = []
    case_dirs = sorted(d for d in npy_root.iterdir() if d.is_dir())
    if max_cases > 0:
        case_dirs = case_dirs[:max_cases]

    for case_dir in tqdm(case_dirs, desc="Cases"):
        case_id = case_dir.name
        ct_path = case_dir / "ct.npy"
        mask_path = case_dir / "mask.npy"
        text_path = case_dir / "text.json"
        if not ct_path.exists() or not mask_path.exists():
            continue

        ct_vol = load_volume(ct_path).astype(np.float32)
        mask_vol = np.rint(load_volume(mask_path)).astype(np.int32)
        D, H, W = ct_vol.shape

        text_map = {}
        if text_path.exists():
            with open(text_path, "r", encoding="utf-8") as f:
                text_map = json.load(f)

        label_ids = sorted(int(k) for k in text_map.keys())
        if not label_ids:
            label_ids = sorted(int(x) for x in np.unique(mask_vol) if x != 0)

        # Find needed slices
        slices_needed = set()
        for lid in label_ids:
            for z in range(D):
                if (mask_vol[z] == lid).any():
                    slices_needed.add(z)

        # Build per-slice work
        slice_to_labels = {z: [] for z in sorted(slices_needed)}
        pred_3d_all = {}
        prob_3d_all = {}  # store soft probs for confidence gating

        for lid in label_ids:
            gt_3d = (mask_vol == lid).astype(np.uint8)
            pred_3d_all[lid] = np.zeros((D, H, W), dtype=np.uint8)
            prob_3d_all[lid] = np.zeros((D, H, W), dtype=np.float32)
            for z in range(D):
                if use_adaptive_margin:
                    bbox = get_bbox_2d_adaptive(gt_3d[z])
                else:
                    bbox = get_bbox_2d_fixed(gt_3d[z], margin=bbox_margin)
                if bbox is not None:
                    slice_to_labels[z].append((lid, bbox))

        # Process slice by slice
        for z in sorted(slices_needed):
            if not slice_to_labels[z]:
                continue
            emb = encode_slice(model, ct_vol[z], device)

            for lid, bbox in slice_to_labels[z]:
                if use_ensemble:
                    # Bbox perturbation ensemble
                    prob_accum = np.zeros((H, W), dtype=np.float64)
                    for _ in range(ensemble_n):
                        jbox = jitter_bbox(bbox, H, W, jitter_px=ensemble_jitter)
                        box_1024 = np.array(jbox, dtype=float) / np.array([W, H, W, H]) * 1024
                        prob, _ = medsam_infer_soft(model, emb, box_1024[None, :], H, W, device)
                        prob_accum += prob
                    prob_map = (prob_accum / ensemble_n).astype(np.float32)
                else:
                    box_1024 = np.array(bbox, dtype=float) / np.array([W, H, W, H]) * 1024
                    prob_map, _ = medsam_infer_soft(model, emb, box_1024[None, :], H, W, device)

                # Thresholding
                if use_otsu:
                    mask_slice = threshold_otsu_auto(prob_map)
                else:
                    mask_slice = threshold_fixed(prob_map, fixed_thresh)

                # Erosion
                if use_erosion:
                    mask_slice = apply_erosion(mask_slice, iterations=erosion_iters)

                prob_3d_all[lid][z] = prob_map
                pred_3d_all[lid][z] = mask_slice

            del emb
            torch.cuda.empty_cache()

        # Confidence gating: suppress low-confidence slices
        if use_confidence_gate:
            for lid in label_ids:
                for z in range(D):
                    pred_z = pred_3d_all[lid][z]
                    if pred_z.sum() == 0:
                        continue
                    prob_z = prob_3d_all[lid][z]
                    mean_conf = prob_z[pred_z > 0].mean()
                    if mean_conf < confidence_min:
                        pred_3d_all[lid][z] = np.zeros_like(pred_z)

        # Compute metrics
        for lid in label_ids:
            gt_3d = (mask_vol == lid).astype(np.uint8)
            pred_3d = pred_3d_all[lid]
            d = dice_score(pred_3d, gt_3d)
            iou = iou_score(pred_3d, gt_3d)
            desc = text_map.get(str(lid), "")
            results.append({
                "case_id": case_id,
                "mask_id": lid,
                "label_desc": desc[:120],
                "gt_voxels": int(gt_3d.sum()),
                "pred_voxels": int(pred_3d.sum()),
                "dice": round(d, 4),
                "iou": round(iou, 4),
            })

        del pred_3d_all, prob_3d_all
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser("MedSAM optimized inference")
    parser.add_argument("--npy_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="./results_medsam_optimized")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bbox_margin", type=int, default=5)
    parser.add_argument("--tricks", type=str, default="",
                        help="Comma-separated tricks: adaptive_margin,threshold_0.65,"
                             "otsu,erosion,ensemble,confidence_gate")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading MedSAM...")
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    model = model.to(device)
    model.image_encoder = model.image_encoder.half()
    model.eval()

    tricks_list = [t.strip() for t in args.tricks.split(",") if t.strip()]
    trick_tag = "_".join(tricks_list) if tricks_list else "baseline"

    t0 = time.time()
    results = run_optimized_inference(
        model, device,
        npy_root=Path(args.npy_root),
        out_dir=out_dir,
        max_cases=args.max_cases,
        tricks=tricks_list,
        bbox_margin=args.bbox_margin,
    )
    elapsed = time.time() - t0

    df = pd.DataFrame(results)
    csv_path = out_dir / f"results_{trick_tag}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nDone in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"Config: {trick_tag}")
    print(f"Results: {len(df)} targets")
    if len(df) > 0:
        print(f"Mean Dice:   {df['dice'].mean():.4f}")
        print(f"Mean IoU:    {df['iou'].mean():.4f}")
        print(f"Median Dice: {df['dice'].median():.4f}")
        print(f"Dice >= 0.5: {(df['dice'] >= 0.5).sum()} / {len(df)}")
        print(f"Dice == 0.0: {(df['dice'] == 0.0).sum()} / {len(df)}")
        # Over-segmentation ratio
        valid = df[df['gt_voxels'] > 0]
        if len(valid) > 0:
            overseg = (valid['pred_voxels'] / valid['gt_voxels']).mean()
            print(f"Mean pred/gt ratio: {overseg:.2f}x")
    print(f"{'='*60}")
    print(f"Saved to: {csv_path}")


if __name__ == "__main__":
    main()
