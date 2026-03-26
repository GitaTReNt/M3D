#!/usr/bin/env python3
"""
Improved MedSAM inference on M3D-RefSeg dataset.

Improvements over baseline:
  1. Multi-mask output: use SAM's 3-mask mode + IoU head to pick best mask
  2. Iterative refinement: use 1st-pass mask as mask prompt for 2nd pass
  3. Post-processing: largest connected component per slice
  4. Test-time augmentation: horizontal flip + average

Usage:
  python inference_medsam_improved.py \
      --npy_root  D:/M3D/M3D_RefSeg_npy \
      --checkpoint D:/M3D/MedSAM/work_dir/MedSAM_finetuned/medsam_vit_b.pth \
      --out_dir ./results_medsam_improved \
      --max_cases 30 \
      --tricks multimask,refine,cc3d,tta
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "MedSAM"))
from segment_anything import sam_model_registry


# --------------- Metrics ---------------

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    intersection = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return float(2.0 * intersection / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return float(intersection / union)


# --------------- Data loading ---------------

def load_volume(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def get_bbox_2d(mask_2d: np.ndarray, margin: int = 5) -> list:
    ys, xs = np.where(mask_2d > 0)
    if len(ys) == 0:
        return None
    H, W = mask_2d.shape
    x_min = max(0, int(xs.min()) - margin)
    y_min = max(0, int(ys.min()) - margin)
    x_max = min(W, int(xs.max()) + margin)
    y_max = min(H, int(ys.max()) + margin)
    return [x_min, y_min, x_max, y_max]


# --------------- Post-processing ---------------

def keep_largest_cc_2d(mask_2d: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a 2D binary mask."""
    if mask_2d.sum() == 0:
        return mask_2d
    labeled, num = ndimage.label(mask_2d)
    if num <= 1:
        return mask_2d
    sizes = ndimage.sum(mask_2d, labeled, range(1, num + 1))
    largest = np.argmax(sizes) + 1
    return (labeled == largest).astype(np.uint8)


def keep_largest_cc_3d(mask_3d: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a 3D binary mask."""
    if mask_3d.sum() == 0:
        return mask_3d
    labeled, num = ndimage.label(mask_3d)
    if num <= 1:
        return mask_3d
    sizes = ndimage.sum(mask_3d, labeled, range(1, num + 1))
    largest = np.argmax(sizes) + 1
    return (labeled == largest).astype(np.uint8)


# --------------- MedSAM inference ---------------

def prepare_slice_for_medsam(ct_slice: np.ndarray) -> np.ndarray:
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
def encode_slice_flipped(model, ct_slice: np.ndarray, device) -> torch.Tensor:
    """Encode horizontally flipped slice."""
    return encode_slice(model, ct_slice[:, ::-1].copy(), device)


@torch.no_grad()
def medsam_infer_slice(model, img_embed, box_1024, H, W, device,
                       multimask=False, mask_input=None):
    """
    Run MedSAM mask decoder.
    If multimask=True, returns the mask with highest IoU prediction.
    If mask_input is provided, use it as mask prompt for refinement.
    Returns: (binary mask (H,W) uint8, logits (1,1,256,256) for refinement)
    """
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=device)
    if box_torch.ndim == 2:
        box_torch = box_torch[:, None, :]

    sparse_emb, dense_emb = model.prompt_encoder(
        points=None, boxes=box_torch, masks=mask_input,
    )
    low_res_logits, iou_pred = model.mask_decoder(
        image_embeddings=img_embed.float(),
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=multimask,
    )

    if multimask and low_res_logits.shape[1] > 1:
        # Pick mask with highest predicted IoU
        best_idx = iou_pred.argmax(dim=1)  # (B,)
        low_res_logits = low_res_logits[0, best_idx[0]].unsqueeze(0).unsqueeze(0)

    low_res_pred = torch.sigmoid(low_res_logits)
    full_res = F.interpolate(
        low_res_pred, size=(H, W), mode="bilinear", align_corners=False,
    )
    mask = (full_res.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    return mask, low_res_logits  # return logits for potential refinement


def infer_slice_with_tricks(model, emb, box_1024, H, W, device, tricks,
                            emb_flip=None, box_1024_flip=None):
    """Apply selected tricks to a single slice inference."""
    use_multimask = "multimask" in tricks
    use_refine = "refine" in tricks
    use_cc = "cc3d" in tricks
    use_tta = "tta" in tricks

    # --- Pass 1 ---
    mask, logits = medsam_infer_slice(
        model, emb, box_1024, H, W, device,
        multimask=use_multimask,
    )

    # --- Pass 2: iterative refinement using mask prompt ---
    if use_refine:
        # Resize logits to 256x256 as mask input for prompt encoder
        mask_input = F.interpolate(
            logits.float(), size=(256, 256), mode="bilinear", align_corners=False,
        )
        mask, logits = medsam_infer_slice(
            model, emb, box_1024, H, W, device,
            multimask=False,  # single mask in refinement pass
            mask_input=mask_input,
        )

    # --- TTA: horizontal flip ---
    if use_tta and emb_flip is not None:
        mask_flip, _ = medsam_infer_slice(
            model, emb_flip, box_1024_flip, H, W, device,
            multimask=use_multimask,
        )
        # Flip prediction back
        mask_flip = mask_flip[:, ::-1].copy()
        # Average: take union where both agree, or use soft voting
        # Simple: majority vote (both must agree)
        mask = ((mask.astype(float) + mask_flip.astype(float)) >= 1.0).astype(np.uint8)

    # --- Post-processing: largest connected component ---
    if use_cc:
        mask = keep_largest_cc_2d(mask)

    return mask


# --------------- Main inference loop ---------------

def run_improved_inference(
    model, device, npy_root: Path, out_dir: Path,
    bbox_margin: int = 5, max_cases: int = 0, tricks: set = None,
):
    if tricks is None:
        tricks = set()

    use_tta = "tta" in tricks
    use_cc3d = "cc3d" in tricks
    results = []
    case_dirs = sorted([d for d in npy_root.iterdir() if d.is_dir()])
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

        if text_path.exists():
            with open(text_path, "r", encoding="utf-8") as f:
                text_map = json.load(f)
        else:
            text_map = {}

        label_ids = sorted([int(k) for k in text_map.keys()])
        if not label_ids:
            label_ids = sorted([int(x) for x in np.unique(mask_vol) if x != 0])

        slices_needed = set()
        for lid in label_ids:
            for z in range(D):
                if (mask_vol[z] == lid).any():
                    slices_needed.add(z)

        slice_to_labels = {z: [] for z in sorted(slices_needed)}
        pred_3d_all = {}
        for lid in label_ids:
            gt_3d = (mask_vol == lid).astype(np.uint8)
            pred_3d_all[lid] = np.zeros_like(gt_3d)
            for z in range(D):
                bbox = get_bbox_2d(gt_3d[z], margin=bbox_margin)
                if bbox is not None:
                    slice_to_labels[z].append((lid, bbox))

        for z in sorted(slices_needed):
            if not slice_to_labels[z]:
                continue
            emb = encode_slice(model, ct_vol[z], device)
            emb_flip = None
            if use_tta:
                emb_flip = encode_slice_flipped(model, ct_vol[z], device)

            for lid, bbox in slice_to_labels[z]:
                box_1024 = np.array(bbox, dtype=float) / np.array([W, H, W, H]) * 1024
                box_1024 = box_1024[None, :]

                box_1024_flip = None
                if use_tta:
                    # Mirror bbox horizontally in 1024 space
                    box_1024_flip = box_1024.copy()
                    box_1024_flip[0, 0] = 1024 - box_1024[0, 2]  # new x_min = 1024 - old x_max
                    box_1024_flip[0, 2] = 1024 - box_1024[0, 0]  # new x_max = 1024 - old x_min

                pred_slice = infer_slice_with_tricks(
                    model, emb, box_1024, H, W, device, tricks,
                    emb_flip=emb_flip, box_1024_flip=box_1024_flip,
                )
                pred_3d_all[lid][z] = pred_slice

            del emb
            if emb_flip is not None:
                del emb_flip
            torch.cuda.empty_cache()

        # 3D post-processing: largest connected component
        for lid in label_ids:
            if use_cc3d:
                pred_3d_all[lid] = keep_largest_cc_3d(pred_3d_all[lid])

        # Compute metrics
        for lid in label_ids:
            gt_3d = (mask_vol == lid).astype(np.uint8)
            pred_3d = pred_3d_all[lid]
            gt_voxels = int(gt_3d.sum())
            slices_with_roi = sum(1 for _, bbox_list_lid in slice_to_labels.items()
                                  for l, _ in bbox_list_lid if l == lid)
            d = dice_score(pred_3d, gt_3d)
            iou = iou_score(pred_3d, gt_3d)
            desc = text_map.get(str(lid), "")

            results.append({
                "case_id": case_id,
                "mask_id": lid,
                "label_desc": desc[:120],
                "gt_voxels": gt_voxels,
                "pred_voxels": int(pred_3d.sum()),
                "slices_with_roi": slices_with_roi,
                "dice": round(d, 4),
                "iou": round(iou, 4),
            })

        del pred_3d_all
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser("Improved MedSAM inference on M3D-RefSeg")
    parser.add_argument("--npy_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="./results_medsam_improved")
    parser.add_argument("--bbox_margin", type=int, default=5)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tricks", default="multimask,refine,cc3d",
                        help="Comma-separated: multimask,refine,cc3d,tta")
    args = parser.parse_args()

    tricks = set(args.tricks.split(",")) if args.tricks else set()
    print(f"Tricks enabled: {tricks}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading MedSAM model...")
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    model = model.to(device)
    model.image_encoder = model.image_encoder.half()
    model.eval()
    print(f"Model loaded on {device} (image encoder fp16).")

    t0 = time.time()
    results = run_improved_inference(
        model, device,
        npy_root=Path(args.npy_root),
        out_dir=out_dir,
        bbox_margin=args.bbox_margin,
        max_cases=args.max_cases,
        tricks=tricks,
    )
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s")

    df = pd.DataFrame(results)
    trick_tag = "_".join(sorted(tricks)) if tricks else "baseline"
    csv_path = out_dir / f"medsam_improved_{trick_tag}_results.csv"
    df.to_csv(csv_path, index=False)

    # Summary
    print(f"\n{'='*50}")
    print(f"Tricks: {tricks}")
    print(f"Results: {len(df)} targets")
    if len(df) > 0:
        print(f"Mean Dice: {df['dice'].mean():.4f}")
        print(f"Mean IoU:  {df['iou'].mean():.4f}")
        print(f"Median Dice: {df['dice'].median():.4f}")
        print(f"Dice >= 0.5: {(df['dice'] >= 0.5).sum()} / {len(df)}")
        print(f"Dice == 0.0: {(df['dice'] == 0.0).sum()} / {len(df)}")
    print(f"{'='*50}")
    print(f"Saved to: {csv_path}")


if __name__ == "__main__":
    main()
