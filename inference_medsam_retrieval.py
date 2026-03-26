#!/usr/bin/env python3
"""
Text-guided MedSAM via text similarity retrieval.

Approach:
  1. Build spatial database from all cases: text -> (3D bbox, centroid, slice range)
  2. For each test target, find most similar text (leave-one-out) via TF-IDF cosine
  3. Use retrieved bbox as MedSAM prompt (instead of oracle bbox)
  4. Evaluate Dice/IoU

No large models needed — just scikit-learn TF-IDF.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skimage import transform as sk_transform
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "MedSAM"))
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


# --------------- Spatial database ---------------

def build_spatial_db(npy_root: Path):
    """Build {(case_id, mask_id): {text, bbox_norm, centroid_norm, z_range, ...}}."""
    db = []
    for case_dir in sorted(d for d in npy_root.iterdir() if d.is_dir()):
        case_id = case_dir.name
        mask_path = case_dir / "mask.npy"
        text_path = case_dir / "text.json"
        if not mask_path.exists() or not text_path.exists():
            continue

        mask_vol = np.load(str(mask_path))
        if mask_vol.ndim == 4 and mask_vol.shape[0] == 1:
            mask_vol = mask_vol[0]
        mask_vol = np.rint(mask_vol).astype(np.int32)
        D, H, W = mask_vol.shape

        with open(text_path, "r", encoding="utf-8") as f:
            text_map = json.load(f)

        for lid_str, desc in text_map.items():
            lid = int(lid_str)
            gt = (mask_vol == lid).astype(np.uint8)
            if gt.sum() == 0:
                # Store empty targets too
                db.append({
                    "case_id": case_id, "mask_id": lid, "text": desc,
                    "gt_voxels": 0, "z_min": -1, "z_max": -1,
                    "y_min": -1, "y_max": -1, "x_min": -1, "x_max": -1,
                    "D": D, "H": H, "W": W,
                })
                continue

            zs, ys, xs = np.where(gt > 0)
            db.append({
                "case_id": case_id, "mask_id": lid, "text": desc,
                "gt_voxels": int(gt.sum()),
                "z_min": int(zs.min()), "z_max": int(zs.max()),
                "y_min": int(ys.min()), "y_max": int(ys.max()),
                "x_min": int(xs.min()), "x_max": int(xs.max()),
                "D": D, "H": H, "W": W,
            })

    return db


def retrieve_prior(query_text, query_case_id, db, tfidf_matrix, vectorizer, top_k=3):
    """Find top-k similar texts from OTHER cases, return aggregated bbox."""
    q_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]

    # Exclude same case and empty targets
    candidates = []
    for i, entry in enumerate(db):
        if entry["case_id"] == query_case_id:
            continue
        if entry["gt_voxels"] == 0:
            continue
        candidates.append((sims[i], i))

    candidates.sort(key=lambda x: -x[0])
    top = candidates[:top_k]

    if not top:
        return None, 0.0

    # Aggregate: use weighted average of normalized bboxes
    z_mins, z_maxs = [], []
    y_mins, y_maxs = [], []
    x_mins, x_maxs = [], []
    weights = []

    for sim, idx in top:
        e = db[idx]
        # Normalize to [0, 1] range
        z_mins.append(e["z_min"] / e["D"])
        z_maxs.append(e["z_max"] / e["D"])
        y_mins.append(e["y_min"] / e["H"])
        y_maxs.append(e["y_max"] / e["H"])
        x_mins.append(e["x_min"] / e["W"])
        x_maxs.append(e["x_max"] / e["W"])
        weights.append(sim)

    w = np.array(weights)
    w = w / w.sum()

    prior = {
        "z_min": float(np.dot(w, z_mins)),
        "z_max": float(np.dot(w, z_maxs)),
        "y_min": float(np.dot(w, y_mins)),
        "y_max": float(np.dot(w, y_maxs)),
        "x_min": float(np.dot(w, x_mins)),
        "x_max": float(np.dot(w, x_maxs)),
    }
    avg_sim = float(np.mean(weights))
    return prior, avg_sim


# --------------- MedSAM inference ---------------

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


# --------------- Main ---------------

def run_retrieval_inference(
    model, device, npy_root, db, tfidf_matrix, vectorizer,
    max_cases=0, top_k=3, bbox_margin=10,
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
        if not ct_path.exists() or not mask_path.exists():
            continue

        ct_vol = load_volume(ct_path).astype(np.float32)
        mask_vol = np.rint(load_volume(mask_path)).astype(np.int32)
        D, H, W = ct_vol.shape

        if not text_path.exists():
            continue
        with open(text_path, "r", encoding="utf-8") as f:
            text_map = json.load(f)

        for lid_str, desc in text_map.items():
            lid = int(lid_str)
            gt_3d = (mask_vol == lid).astype(np.uint8)

            # Retrieve prior from similar texts
            prior, avg_sim = retrieve_prior(desc, case_id, db, tfidf_matrix, vectorizer, top_k=top_k)

            if prior is None or prior["z_min"] < 0:
                # No valid prior found
                results.append({
                    "case_id": case_id, "mask_id": lid,
                    "label_desc": desc[:120],
                    "gt_voxels": int(gt_3d.sum()),
                    "pred_voxels": 0,
                    "avg_sim": 0.0,
                    "dice": 1.0 if gt_3d.sum() == 0 else 0.0,
                    "iou": 1.0 if gt_3d.sum() == 0 else 0.0,
                })
                continue

            # Convert normalized prior to pixel coords
            z_start = max(0, int(prior["z_min"] * D) - 1)
            z_end = min(D - 1, int(prior["z_max"] * D) + 1)
            y_min_px = max(0, int(prior["y_min"] * H) - bbox_margin)
            y_max_px = min(H, int(prior["y_max"] * H) + bbox_margin)
            x_min_px = max(0, int(prior["x_min"] * W) - bbox_margin)
            x_max_px = min(W, int(prior["x_max"] * W) + bbox_margin)

            # Run MedSAM on predicted slice range with predicted bbox
            pred_3d = np.zeros_like(gt_3d)
            bbox = [x_min_px, y_min_px, x_max_px, y_max_px]

            for z in range(z_start, z_end + 1):
                emb = encode_slice(model, ct_vol[z], device)
                box_1024 = np.array(bbox, dtype=float) / np.array([W, H, W, H]) * 1024
                box_1024 = box_1024[None, :]
                pred_slice = medsam_infer_slice(model, emb, box_1024, H, W, device)
                pred_3d[z] = pred_slice
                del emb

            torch.cuda.empty_cache()

            d = dice_score(pred_3d, gt_3d)
            iou = iou_score(pred_3d, gt_3d)
            results.append({
                "case_id": case_id, "mask_id": lid,
                "label_desc": desc[:120],
                "gt_voxels": int(gt_3d.sum()),
                "pred_voxels": int(pred_3d.sum()),
                "avg_sim": round(avg_sim, 4),
                "dice": round(d, 4),
                "iou": round(iou, 4),
            })

    return results


def main():
    parser = argparse.ArgumentParser("Text-retrieval MedSAM inference")
    parser.add_argument("--npy_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="./results_medsam_retrieval")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--bbox_margin", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build spatial database from ALL cases
    print("Building spatial database...")
    npy_root = Path(args.npy_root)
    db = build_spatial_db(npy_root)
    print(f"  {len(db)} entries from {len(set(e['case_id'] for e in db))} cases")

    # Step 2: Build TF-IDF matrix
    print("Building TF-IDF index...")
    texts = [e["text"] for e in db]
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        stop_words="english", sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

    # Step 3: Load MedSAM
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading MedSAM on {device}...")
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    model = model.to(device)
    model.image_encoder = model.image_encoder.half()
    model.eval()

    # Step 4: Run inference
    print(f"Running retrieval-guided inference (top_k={args.top_k}, margin={args.bbox_margin})...")
    t0 = time.time()
    results = run_retrieval_inference(
        model, device, npy_root, db, tfidf_matrix, vectorizer,
        max_cases=args.max_cases, top_k=args.top_k, bbox_margin=args.bbox_margin,
    )
    elapsed = time.time() - t0

    # Save
    df = pd.DataFrame(results)
    csv_path = out_dir / f"retrieval_top{args.top_k}_results.csv"
    df.to_csv(csv_path, index=False)

    # Summary
    print(f"\nDone in {elapsed:.1f}s")
    print(f"{'='*50}")
    print(f"Results: {len(df)} targets")
    if len(df) > 0:
        print(f"Mean Dice:   {df['dice'].mean():.4f}")
        print(f"Mean IoU:    {df['iou'].mean():.4f}")
        print(f"Median Dice: {df['dice'].median():.4f}")
        print(f"Dice >= 0.5: {(df['dice'] >= 0.5).sum()} / {len(df)}")
        print(f"Dice == 0.0: {(df['dice'] == 0.0).sum()} / {len(df)}")
        print(f"Mean Sim:    {df['avg_sim'].mean():.4f}")
    print(f"{'='*50}")
    print(f"Saved to: {csv_path}")


if __name__ == "__main__":
    main()
