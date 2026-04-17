#!/usr/bin/env python3
"""Diagnose MedSAM: fp16 vs fp32, bbox vs point prompt, per-slice 2D Dice."""

import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from skimage import transform as sk_transform
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "MedSAM"))
from segment_anything import sam_model_registry

device = torch.device("cuda:0")


def dice_score(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return 2.0 * inter / denom


def prepare_slice(ct_slice):
    img_3c = np.repeat(ct_slice[:, :, None], 3, axis=-1)
    img_1024 = sk_transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.float32)
    vmin, vmax = img_1024.min(), img_1024.max()
    if vmax - vmin > 1e-8:
        img_1024 = (img_1024 - vmin) / (vmax - vmin)
    return img_1024


def get_bbox_adaptive(mask_2d):
    ys, xs = np.where(mask_2d > 0)
    if len(ys) == 0:
        return None
    H, W = mask_2d.shape
    bw, bh = int(xs.max()) - int(xs.min()), int(ys.max()) - int(ys.min())
    margin = max(2, min(15, int(0.1 * max(bw, bh))))
    return [max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(W, int(xs.max()) + margin), min(H, int(ys.max()) + margin)]


def get_centroid(mask_2d):
    ys, xs = np.where(mask_2d > 0)
    if len(ys) == 0:
        return None
    return [int(xs.mean()), int(ys.mean())]


@torch.no_grad()
def infer_bbox(model, img_embed, bbox, H, W, device):
    box_1024 = np.array(bbox, dtype=float) / np.array([W, H, W, H]) * 1024
    box_t = torch.as_tensor(box_1024[None, None, :], dtype=torch.float, device=device)
    sparse, dense = model.prompt_encoder(points=None, boxes=box_t, masks=None)
    logits, _ = model.mask_decoder(
        image_embeddings=img_embed.float(),
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    prob = torch.sigmoid(logits)
    prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
    return (prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


@torch.no_grad()
def infer_point(model, img_embed, point_xy, H, W, device):
    """Use point prompt (foreground centroid)."""
    pt_1024 = np.array(point_xy, dtype=float) / np.array([W, H]) * 1024
    coords = torch.as_tensor(pt_1024[None, None, :], dtype=torch.float, device=device)
    labels = torch.ones(1, 1, dtype=torch.int, device=device)  # foreground
    sparse, dense = model.prompt_encoder(
        points=(coords, labels), boxes=None, masks=None,
    )
    logits, _ = model.mask_decoder(
        image_embeddings=img_embed.float(),
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,  # use multi-mask for point prompts (SAM recommendation)
    )
    # Pick the mask with highest IoU prediction
    iou_preds = model.mask_decoder.iou_prediction_head(
        model.mask_decoder.output_hypernetworks_mlps[0](
            sparse  # this doesn't work directly, let's just use the logits
        )
    ) if False else None
    # Just pick mask with largest area that's reasonable
    prob = torch.sigmoid(logits)
    prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
    masks = (prob.squeeze(0) > 0.5).cpu().numpy()  # (3, H, W)
    # Pick mask with smallest nonzero area (least over-segmentation)
    areas = [m.sum() for m in masks]
    best = min(range(len(areas)), key=lambda i: areas[i] if areas[i] > 0 else float("inf"))
    return masks[best].astype(np.uint8)


@torch.no_grad()
def infer_point_v2(model, img_embed, point_xy, H, W, device):
    """Point prompt with multi-mask output, pick by IoU prediction."""
    pt_1024 = np.array(point_xy, dtype=float) / np.array([W, H]) * 1024
    coords = torch.as_tensor(pt_1024[None, None, :], dtype=torch.float, device=device)
    labels = torch.ones(1, 1, dtype=torch.int, device=device)
    sparse, dense = model.prompt_encoder(
        points=(coords, labels), boxes=None, masks=None,
    )
    logits, iou_pred = model.mask_decoder(
        image_embeddings=img_embed.float(),
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
    )
    prob = torch.sigmoid(logits)
    prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
    masks = (prob.squeeze(0) > 0.5).cpu().numpy()  # (3, H, W)
    iou_vals = iou_pred.squeeze().cpu().numpy()  # (3,)
    best = int(np.argmax(iou_vals))
    return masks[best].astype(np.uint8)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    npy_root = repo_root / "data" / "M3D_RefSeg_npy"
    ckpt_path = str(repo_root / "third_party" / "MedSAM" / "work_dir" / "MedSAM_finetuned" / "medsam_vit_b.pth")
    case_dirs = sorted(d for d in npy_root.iterdir() if d.is_dir() and (d / "ct.npy").exists())[:10]

    # Test 1: fp16 vs fp32
    print("=" * 60)
    print("TEST 1: fp16 encoder vs fp32 encoder")
    print("=" * 60)

    results_fp16 = []
    results_fp32 = []

    # fp32 model
    print("Loading MedSAM (fp32)...")
    model32 = sam_model_registry["vit_b"](
        checkpoint=ckpt_path
    ).to(device).eval()

    # fp16 model
    print("Loading MedSAM (fp16 encoder)...")
    model16 = sam_model_registry["vit_b"](
        checkpoint=ckpt_path
    ).to(device).eval()
    model16.image_encoder = model16.image_encoder.half()

    for case_dir in tqdm(case_dirs[:5], desc="fp16 vs fp32"):
        ct_vol = np.load(str(case_dir / "ct.npy")).astype(np.float32)
        if ct_vol.ndim == 4:
            ct_vol = ct_vol[0]
        mask_vol = np.rint(np.load(str(case_dir / "mask.npy"))).astype(np.int32)
        if mask_vol.ndim == 4:
            mask_vol = mask_vol[0]
        D, H, W = ct_vol.shape

        with open(str(case_dir / "text.json"), "r") as f:
            text_map = json.load(f)

        for lid_str in text_map:
            lid = int(lid_str)
            gt_3d = (mask_vol == lid).astype(np.uint8)
            if gt_3d.sum() == 0:
                continue

            pred16_3d = np.zeros_like(gt_3d)
            pred32_3d = np.zeros_like(gt_3d)

            for z in range(D):
                if not gt_3d[z].any():
                    continue
                bbox = get_bbox_adaptive(gt_3d[z])
                if bbox is None:
                    continue

                img_1024 = prepare_slice(ct_vol[z])

                # fp16
                img_t16 = torch.tensor(img_1024).half().permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda"):
                    emb16 = model16.image_encoder(img_t16)
                pred16_3d[z] = infer_bbox(model16, emb16, bbox, H, W, device)

                # fp32
                img_t32 = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
                emb32 = model32.image_encoder(img_t32)
                pred32_3d[z] = infer_bbox(model32, emb32, bbox, H, W, device)

                del emb16, emb32, img_t16, img_t32
                torch.cuda.empty_cache()

            d16 = dice_score(pred16_3d, gt_3d)
            d32 = dice_score(pred32_3d, gt_3d)
            results_fp16.append(d16)
            results_fp32.append(d32)

    print(f"\nfp16: {np.mean(results_fp16):.4f}  fp32: {np.mean(results_fp32):.4f}  delta: {np.mean(results_fp32) - np.mean(results_fp16):+.4f}")

    # Free one model
    del model32
    torch.cuda.empty_cache()

    # Test 2: bbox vs point prompt
    print("\n" + "=" * 60)
    print("TEST 2: bbox prompt vs point prompt (centroid)")
    print("=" * 60)

    results_bbox = []
    results_point = []

    for case_dir in tqdm(case_dirs[:10], desc="bbox vs point"):
        ct_vol = np.load(str(case_dir / "ct.npy")).astype(np.float32)
        if ct_vol.ndim == 4:
            ct_vol = ct_vol[0]
        mask_vol = np.rint(np.load(str(case_dir / "mask.npy"))).astype(np.int32)
        if mask_vol.ndim == 4:
            mask_vol = mask_vol[0]
        D, H, W = ct_vol.shape

        with open(str(case_dir / "text.json"), "r") as f:
            text_map = json.load(f)

        for lid_str in text_map:
            lid = int(lid_str)
            gt_3d = (mask_vol == lid).astype(np.uint8)
            if gt_3d.sum() == 0:
                continue

            pred_bbox_3d = np.zeros_like(gt_3d)
            pred_point_3d = np.zeros_like(gt_3d)

            for z in range(D):
                if not gt_3d[z].any():
                    continue

                img_1024 = prepare_slice(ct_vol[z])
                img_t = torch.tensor(img_1024).half().permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda"):
                    emb = model16.image_encoder(img_t)

                # Bbox
                bbox = get_bbox_adaptive(gt_3d[z])
                if bbox is not None:
                    pred_bbox_3d[z] = infer_bbox(model16, emb, bbox, H, W, device)

                # Point (centroid)
                pt = get_centroid(gt_3d[z])
                if pt is not None:
                    pred_point_3d[z] = infer_point_v2(model16, emb, pt, H, W, device)

                del emb, img_t
                torch.cuda.empty_cache()

            db = dice_score(pred_bbox_3d, gt_3d)
            dp = dice_score(pred_point_3d, gt_3d)
            results_bbox.append(db)
            results_point.append(dp)
            gt_sum = gt_3d.sum()
            if abs(db - dp) > 0.1:
                print(f"  {case_dir.name} L{lid}: bbox={db:.3f} point={dp:.3f} gt={gt_sum}")

    print(f"\nBbox:  {np.mean(results_bbox):.4f}")
    print(f"Point: {np.mean(results_point):.4f}")
    print(f"Delta: {np.mean(results_point) - np.mean(results_bbox):+.4f}")

    # Test 3: Per-slice 2D Dice (to distinguish "bad slice selection" from "bad segmentation")
    print("\n" + "=" * 60)
    print("TEST 3: Per-slice 2D Dice (how good is MedSAM on individual slices?)")
    print("=" * 60)

    slice_dices = []
    for case_dir in tqdm(case_dirs[:10], desc="2D Dice"):
        ct_vol = np.load(str(case_dir / "ct.npy")).astype(np.float32)
        if ct_vol.ndim == 4:
            ct_vol = ct_vol[0]
        mask_vol = np.rint(np.load(str(case_dir / "mask.npy"))).astype(np.int32)
        if mask_vol.ndim == 4:
            mask_vol = mask_vol[0]
        D, H, W = ct_vol.shape

        with open(str(case_dir / "text.json"), "r") as f:
            text_map = json.load(f)

        for lid_str in text_map:
            lid = int(lid_str)
            gt_3d = (mask_vol == lid).astype(np.uint8)
            if gt_3d.sum() == 0:
                continue

            for z in range(D):
                if not gt_3d[z].any():
                    continue
                bbox = get_bbox_adaptive(gt_3d[z])
                if bbox is None:
                    continue

                img_1024 = prepare_slice(ct_vol[z])
                img_t = torch.tensor(img_1024).half().permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda"):
                    emb = model16.image_encoder(img_t)

                pred = infer_bbox(model16, emb, bbox, H, W, device)
                d2d = dice_score(pred, gt_3d[z])
                slice_dices.append(d2d)

                del emb, img_t
                torch.cuda.empty_cache()

    print(f"Total slices: {len(slice_dices)}")
    print(f"Mean 2D Dice: {np.mean(slice_dices):.4f}")
    print(f"Median 2D Dice: {np.median(slice_dices):.4f}")
    print(f"2D Dice >= 0.5: {sum(1 for d in slice_dices if d >= 0.5)}/{len(slice_dices)}")
    print(f"2D Dice >= 0.7: {sum(1 for d in slice_dices if d >= 0.7)}/{len(slice_dices)}")
    print(f"2D Dice >= 0.9: {sum(1 for d in slice_dices if d >= 0.9)}/{len(slice_dices)}")


if __name__ == "__main__":
    main()
