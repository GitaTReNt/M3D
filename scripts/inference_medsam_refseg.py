#!/usr/bin/env python3
"""
MedSAM inference on M3D-RefSeg dataset (fully on GPU).

Mode:
  oracle_bbox — use ground-truth mask to derive per-slice 2D bounding box

MedSAM is a 2D model: each z-slice is processed independently, then
predictions are stacked back into a 3D volume for evaluation.

Data format (from m3d_refseg_data_prepare.py):
  ct.npy   : (1, 32, 256, 256) float32, [0, 1]
  mask.npy : (1, 32, 256, 256) float32, label-coded
  text.json: {mask_id: description}

Usage:
  python inference_medsam_refseg.py \
      --npy_root  D:/M3D/M3D_RefSeg_npy \
      --checkpoint D:/M3D/MedSAM/work_dir/MedSAM/medsam_vit_b.pth \
      --mode oracle_bbox \
      --out_dir ./results_medsam \
      --bbox_margin 5
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

# Add MedSAM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "MedSAM"))
from segment_anything import sam_model_registry


# --------------- Metrics ---------------

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return float(2.0 * intersection / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return float(intersection / union)


# --------------- Data loading ---------------

def load_volume(path: Path) -> np.ndarray:
    """Load npy -> (D, H, W)."""
    arr = np.load(str(path))
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def get_bbox_2d(mask_2d: np.ndarray, margin: int = 5) -> list:
    """
    Get [x_min, y_min, x_max, y_max] bounding box from a 2D binary mask.
    Returns None if mask is empty.
    """
    ys, xs = np.where(mask_2d > 0)
    if len(ys) == 0:
        return None
    H, W = mask_2d.shape
    x_min = max(0, int(xs.min()) - margin)
    y_min = max(0, int(ys.min()) - margin)
    x_max = min(W, int(xs.max()) + margin)
    y_max = min(H, int(ys.max()) + margin)
    return [x_min, y_min, x_max, y_max]


# --------------- MedSAM inference ---------------

@torch.no_grad()
def medsam_infer_slice(model, img_embed, box_1024, H, W, device):
    """
    Run MedSAM mask decoder on a single slice.
    box_1024: (1, 4) numpy array in 1024x1024 coordinate space.
    Returns: binary mask (H, W) uint8.
    """
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=device)
    if box_torch.ndim == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

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


def prepare_slice_for_medsam(ct_slice: np.ndarray) -> np.ndarray:
    """
    Convert a single CT slice (H, W) float32 [0,1] to
    MedSAM input format: (1024, 1024, 3) float32 [0,1].
    """
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
    """Encode one CT slice -> image embedding on GPU. Uses fp16."""
    img_1024 = prepare_slice_for_medsam(ct_slice)
    img_tensor = (
        torch.tensor(img_1024).half().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    with torch.amp.autocast("cuda"):
        emb = model.image_encoder(img_tensor)
    del img_tensor
    return emb  # (1, 256, 64, 64) on GPU


def run_oracle_bbox_inference(
    model, device, npy_root: Path, out_dir: Path,
    bbox_margin: int = 5, max_cases: int = 0,
):
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

        ct_vol = load_volume(ct_path).astype(np.float32)   # (D, H, W)
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

        # Find which slices have any ROI
        slices_needed = set()
        for lid in label_ids:
            for z in range(D):
                if (mask_vol[z] == lid).any():
                    slices_needed.add(z)

        # Encode needed slices ONE AT A TIME on GPU (no caching of all)
        # Process all labels per slice to minimize re-encoding
        # Strategy: encode slice -> run all labels that need it -> free embedding

        # Build per-slice label needs
        slice_to_labels = {z: [] for z in sorted(slices_needed)}
        pred_3d_all = {}  # lid -> (D, H, W) array
        for lid in label_ids:
            gt_3d = (mask_vol == lid).astype(np.uint8)
            pred_3d_all[lid] = np.zeros_like(gt_3d)
            for z in range(D):
                bbox = get_bbox_2d(gt_3d[z], margin=bbox_margin)
                if bbox is not None:
                    slice_to_labels[z].append((lid, bbox))

        # Process slice by slice
        for z in sorted(slices_needed):
            if not slice_to_labels[z]:
                continue
            # Encode this slice
            emb = encode_slice(model, ct_vol[z], device)
            # Run decoder for each label on this slice
            for lid, bbox in slice_to_labels[z]:
                box_1024 = np.array(bbox, dtype=float) / np.array([W, H, W, H]) * 1024
                box_1024 = box_1024[None, :]
                pred_slice = medsam_infer_slice(model, emb, box_1024, H, W, device)
                pred_3d_all[lid][z] = pred_slice
            del emb
            torch.cuda.empty_cache()

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


# --------------- Main ---------------

def main():
    parser = argparse.ArgumentParser("MedSAM inference on M3D-RefSeg")
    parser.add_argument("--npy_root", required=True, help="Path to M3D_RefSeg_npy/")
    parser.add_argument("--checkpoint", required=True, help="Path to medsam_vit_b.pth")
    parser.add_argument("--mode", choices=["oracle_bbox"], default="oracle_bbox")
    parser.add_argument("--out_dir", default="./results_medsam")
    parser.add_argument("--bbox_margin", type=int, default=5,
                        help="Pixel margin around GT bbox (oracle mode)")
    parser.add_argument("--max_cases", type=int, default=0,
                        help="Limit number of cases (0=all, useful for debugging)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model — all on GPU, image encoder in fp16
    print("Loading MedSAM model...")
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    model = model.to(device)
    model.image_encoder = model.image_encoder.half()
    model.eval()
    print(f"Model loaded on {device} (image encoder fp16).")

    if args.mode == "oracle_bbox":
        print(f"Running oracle bounding box inference (margin={args.bbox_margin})...")
        t0 = time.time()
        results = run_oracle_bbox_inference(
            model, device,
            npy_root=Path(args.npy_root),
            out_dir=out_dir,
            bbox_margin=args.bbox_margin,
            max_cases=args.max_cases,
        )
        elapsed = time.time() - t0
        print(f"Inference done in {elapsed:.1f}s")

    # Save results
    df = pd.DataFrame(results)
    csv_path = out_dir / f"medsam_{args.mode}_results.csv"
    df.to_csv(csv_path, index=False)

    # Summary
    print(f"\n{'='*50}")
    print(f"Results: {len(df)} targets")
    print(f"Mean Dice: {df['dice'].mean():.4f}")
    print(f"Mean IoU:  {df['iou'].mean():.4f}")
    print(f"Median Dice: {df['dice'].median():.4f}")
    print(f"Dice >= 0.5: {(df['dice'] >= 0.5).sum()} / {len(df)}")
    print(f"Dice == 0.0: {(df['dice'] == 0.0).sum()} / {len(df)}")
    print(f"{'='*50}")
    print(f"Saved to: {csv_path}")


if __name__ == "__main__":
    main()
