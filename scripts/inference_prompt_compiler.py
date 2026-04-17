#!/usr/bin/env python3
"""
Text-guided MedSAM inference via Prompt Compiler.

Pipeline:
  1. Parse text → PromptPacket (structured slots)
  2. Atlas prior → coarse spatial ROI
  3. Structured retrieval → refined bbox (leave-one-case-out)
  4. Route by target_form → different prompt strategies
  5. MedSAM inference with compiled prompts

Usage:
  python inference_prompt_compiler.py \
      --npy_root D:/M3D/M3D_RefSeg_npy \
      --checkpoint D:/M3D/MedSAM/work_dir/MedSAM_finetuned/medsam_vit_b.pth \
      --out_dir ./results_prompt_compiler \
      --max_cases 30
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

# Add parent to path for prompt_compiler
sys.path.insert(0, str(Path(__file__).parent))
from prompt_compiler import (
    compile_text, build_prompt_bank, retrieve_prior,
    merge_atlas_and_retrieval,
)

# Add MedSAM
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "MedSAM"))
from segment_anything import sam_model_registry


# --------------- Metrics ---------------

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


# --------------- MedSAM ---------------

def load_volume(path):
    arr = np.load(str(path))
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr

def prepare_slice(ct_slice):
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
    img_1024 = prepare_slice(ct_slice)
    img_t = torch.tensor(img_1024).half().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda"):
        emb = model.image_encoder(img_t)
    del img_t
    return emb

@torch.no_grad()
def medsam_infer_slice(model, img_embed, box_1024, H, W, device):
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


# --------------- Post-processing by target type ---------------

def postprocess_focal(pred_3d, prior, D, H, W):
    """Keep largest component nearest to prior centroid."""
    if pred_3d.sum() == 0:
        return pred_3d
    labeled, num = ndimage.label(pred_3d)
    if num <= 1:
        return pred_3d

    # Prior centroid
    cz = (prior["z_min"] + prior["z_max"]) / 2 * D
    cy = (prior["y_min"] + prior["y_max"]) / 2 * H
    cx = (prior["x_min"] + prior["x_max"]) / 2 * W

    best_label = 1
    best_dist = float("inf")
    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)
        centroid = coords.mean(axis=0)
        dist = np.sqrt((centroid[0]-cz)**2 + (centroid[1]-cy)**2 + (centroid[2]-cx)**2)
        if dist < best_dist:
            best_dist = dist
            best_label = i

    return (labeled == best_label).astype(np.uint8)


def postprocess_multi(pred_3d, prior, D, H, W, min_size=5):
    """Keep all components, just remove tiny noise."""
    if pred_3d.sum() == 0:
        return pred_3d
    labeled, num = ndimage.label(pred_3d)
    out = np.zeros_like(pred_3d)
    for i in range(1, num + 1):
        if (labeled == i).sum() >= min_size:
            out[labeled == i] = 1
    return out


def postprocess_by_type(pred_3d, pkt, prior, D, H, W):
    """Apply type-specific post-processing."""
    if pkt.post_rule == "keep_nearest":
        return postprocess_focal(pred_3d, prior, D, H, W)
    elif pkt.post_rule in ("keep_multi", "organ_roi"):
        return postprocess_multi(pred_3d, prior, D, H, W)
    return pred_3d


# --------------- Main inference ---------------

def run_compiler_inference(
    model, device, npy_root, bank, max_cases=0,
    bbox_margin=10, top_k=5,
):
    results = []
    case_dirs = sorted(d for d in npy_root.iterdir() if d.is_dir())
    if max_cases > 0:
        case_dirs = case_dirs[:max_cases]

    for case_dir in tqdm(case_dirs, desc="Cases"):
        case_id = case_dir.name
        ct_path = case_dir / "ct.npy"
        mask_path = case_dir / "mask.npy"
        text_path = case_dir / "text.json"
        if not ct_path.exists() or not mask_path.exists() or not text_path.exists():
            continue

        ct_vol = load_volume(ct_path).astype(np.float32)
        mask_vol = np.rint(load_volume(mask_path)).astype(np.int32)
        D, H, W = ct_vol.shape

        with open(text_path, "r", encoding="utf-8") as f:
            text_map = json.load(f)

        for lid_str, desc in text_map.items():
            lid = int(lid_str)
            gt_3d = (mask_vol == lid).astype(np.uint8)

            # Step 1: Compile text
            pkt = compile_text(desc)

            # Step 2: Retrieve prior
            ret_prior, ret_score, matched = retrieve_prior(
                pkt, case_id, bank, top_k=top_k,
            )

            # Step 3: Merge atlas + retrieval
            prior = merge_atlas_and_retrieval(pkt, ret_prior, ret_score)

            # Step 4: Convert prior to pixel coords
            z_start = max(0, int(prior["z_min"] * D) - 1)
            z_end = min(D - 1, int(prior["z_max"] * D) + 1)
            y_min_px = max(0, int(prior["y_min"] * H) - bbox_margin)
            y_max_px = min(H, int(prior["y_max"] * H) + bbox_margin)
            x_min_px = max(0, int(prior["x_min"] * W) - bbox_margin)
            x_max_px = min(W, int(prior["x_max"] * W) + bbox_margin)

            # Ensure valid box
            if x_max_px <= x_min_px:
                x_min_px, x_max_px = 0, W
            if y_max_px <= y_min_px:
                y_min_px, y_max_px = 0, H

            bbox = [x_min_px, y_min_px, x_max_px, y_max_px]

            # Step 5: Run MedSAM on predicted slice range
            pred_3d = np.zeros_like(gt_3d)
            for z in range(z_start, z_end + 1):
                emb = encode_slice(model, ct_vol[z], device)
                box_1024 = np.array(bbox, dtype=float) / np.array([W, H, W, H]) * 1024
                box_1024 = box_1024[None, :]
                pred_slice = medsam_infer_slice(model, emb, box_1024, H, W, device)
                pred_3d[z] = pred_slice
                del emb

            torch.cuda.empty_cache()

            # Step 6: Post-process by target type
            pred_3d = postprocess_by_type(pred_3d, pkt, prior, D, H, W)

            d = dice_score(pred_3d, gt_3d)
            iou = iou_score(pred_3d, gt_3d)

            results.append({
                "case_id": case_id,
                "mask_id": lid,
                "label_desc": desc[:120],
                "anatomy": pkt.anatomy,
                "side": pkt.side,
                "target_form": pkt.target_form,
                "finding_type": pkt.finding_type,
                "post_rule": pkt.post_rule,
                "retrieval_score": round(ret_score, 2),
                "gt_voxels": int(gt_3d.sum()),
                "pred_voxels": int(pred_3d.sum()),
                "dice": round(d, 4),
                "iou": round(iou, 4),
            })

    return results


def main():
    parser = argparse.ArgumentParser("Prompt Compiler MedSAM inference")
    parser.add_argument("--npy_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="./results_prompt_compiler")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--bbox_margin", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_root = Path(args.npy_root)

    # Step 1: Build prompt bank
    print("Building prompt bank (parsing all texts + extracting spatial info)...")
    bank = build_prompt_bank(npy_root)
    non_empty = sum(1 for e in bank if e.gt_voxels > 0)
    print(f"  {len(bank)} entries ({non_empty} non-empty) from {len(set(e.case_id for e in bank))} cases")

    # Show parsing stats
    from collections import Counter
    anatomy_dist = Counter(e.packet.anatomy for e in bank)
    form_dist = Counter(e.packet.target_form for e in bank)
    print(f"  Anatomy: {anatomy_dist.most_common(10)}")
    print(f"  Form: {form_dist.most_common()}")

    # Step 2: Load MedSAM
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading MedSAM on {device}...")
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    model = model.to(device)
    model.image_encoder = model.image_encoder.half()
    model.eval()

    # Step 3: Run inference
    print(f"Running prompt-compiler inference (top_k={args.top_k})...")
    t0 = time.time()
    results = run_compiler_inference(
        model, device, npy_root, bank,
        max_cases=args.max_cases,
        bbox_margin=args.bbox_margin,
        top_k=args.top_k,
    )
    elapsed = time.time() - t0

    # Save
    df = pd.DataFrame(results)
    csv_path = out_dir / "prompt_compiler_results.csv"
    df.to_csv(csv_path, index=False)

    # Summary
    print(f"\nDone in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"Results: {len(df)} targets")
    if len(df) > 0:
        print(f"Mean Dice:   {df['dice'].mean():.4f}")
        print(f"Mean IoU:    {df['iou'].mean():.4f}")
        print(f"Median Dice: {df['dice'].median():.4f}")
        print(f"Dice >= 0.5: {(df['dice'] >= 0.5).sum()} / {len(df)}")
        print(f"Dice == 0.0: {(df['dice'] == 0.0).sum()} / {len(df)}")

        # Per target_form breakdown
        print(f"\n--- By target form ---")
        for form in df["target_form"].unique():
            sub = df[df["target_form"] == form]
            print(f"  {form:15s}: n={len(sub):>3d}, Dice={sub['dice'].mean():.4f}, "
                  f"Dice>=0.5: {(sub['dice']>=0.5).sum()}/{len(sub)}")

        # Per anatomy breakdown
        print(f"\n--- By anatomy (top 10) ---")
        for anat in df["anatomy"].value_counts().head(10).index:
            sub = df[df["anatomy"] == anat]
            print(f"  {anat:15s}: n={len(sub):>3d}, Dice={sub['dice'].mean():.4f}")

    print(f"{'='*60}")
    print(f"Saved to: {csv_path}")


if __name__ == "__main__":
    main()
