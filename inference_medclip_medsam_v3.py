#!/usr/bin/env python3
"""
MedCLIP-SAMv2 + MedSAM text-guided inference with inference-time optimizations.

Track B optimizations (no retraining):
  1. CLS-similarity slice pre-filtering (only run IBA on top-K slices)
  2. Text keyword extraction (short prompts for BiomedCLIP)
  3. Saliency power-transform sharpening (gamma > 1)
  4. Percentile-based adaptive threshold
  5. IBA hyperparameter tuning (beta, layer)
  6. Multi-text ensemble (original + shortened + anatomy keyword)
  7. Higher logit threshold for MedSAM (0.6 instead of 0.5)

Usage:
  python inference_medclip_medsam_v3.py \
      --npy_root D:/M3D/M3D_RefSeg_npy \
      --medsam_ckpt D:/M3D/MedSAM/work_dir/MedSAM_finetuned/medsam_vit_b.pth \
      --max_cases 10 --device cuda:0 \
      --tricks cls_filter,text_shorten,gamma_sharpen,percentile_thresh
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import transform as sk_transform
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "MedSAM"))
from segment_anything import sam_model_registry

sys.path.insert(0, str(Path(__file__).parent / "MedCLIP-SAMv2" / "saliency_maps"))


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

def load_volume(path):
    arr = np.load(str(path))
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


# ============================================================
# Text processing
# ============================================================

def shorten_text(desc):
    """Extract key anatomy + finding from long radiology description."""
    # Remove parenthetical explanations
    text = re.sub(r'\([^)]*\)', '', desc)
    # Remove common filler phrases
    for phrase in [
        "is seen", "are seen", "is observed", "are observed",
        "is noted", "are noted", "is considered", "considering",
        "It is", "which is", "suggesting", "indicative of",
        "with a tendency towards", "possibly", "probably",
    ]:
        text = text.replace(phrase, "")

    # Take first sentence/clause (before first comma or period if long)
    text = text.strip()
    if len(text) > 80:
        # Split by comma and take first meaningful chunk
        parts = text.split(",")
        text = parts[0].strip()

    # Further truncate if still too long
    words = text.split()
    if len(words) > 15:
        text = " ".join(words[:15])

    return text.strip()

def extract_anatomy_keyword(desc):
    """Extract just the anatomical keyword from description."""
    # Common anatomy terms to look for
    anatomy_terms = [
        "liver", "kidney", "spleen", "lung", "heart", "aorta",
        "vertebra", "spine", "cervical", "thoracic", "lumbar",
        "lymph node", "gallbladder", "pancreas", "stomach",
        "pelvis", "inguinal", "ureter", "bladder", "thyroid",
        "esophagus", "trachea", "bronch", "rib", "sternum",
        "colon", "rectum", "adrenal", "prostate", "uterus",
        "ovary", "femur", "sacrum", "coccyx", "iliac",
        "vocal cord", "cartilage", "nodule", "mass", "lesion",
        "cyst", "tumor", "calcification", "effusion", "ascites",
    ]
    desc_lower = desc.lower()
    found = []
    for term in anatomy_terms:
        if term in desc_lower:
            found.append(term)
    if found:
        # Return the most specific (longest) match
        found.sort(key=len, reverse=True)
        return found[0]
    # Fallback: first 5 words
    return " ".join(desc.split()[:5])


# ============================================================
# BiomedCLIP saliency (IBA)
# ============================================================

def normalize(x):
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-8:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


class Estimator:
    def __init__(self, layer):
        self.layer = layer
        self.M = None
        self.S = None
        self.N = None
        self.eps = 1e-5

    def shape(self):
        return self.M.shape

    def get_layer(self):
        return self.layer

    def mean(self):
        return self.M.squeeze()

    def std(self):
        return np.sqrt(np.maximum(self.S, self.eps) / np.maximum(self.N, 1.0))


class mySequential(nn.Sequential):
    def forward(self, *input, **kwargs):
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
        return input


def replace_layer(model, target, replacement):
    for name, submodule in model.named_children():
        if submodule is target:
            if isinstance(model, nn.ModuleList):
                model[int(name)] = replacement
            elif isinstance(model, nn.Sequential):
                model[int(name)] = replacement
            else:
                model.__setattr__(name, replacement)
            return True
        elif len(list(submodule.named_children())) > 0:
            if replace_layer(submodule, target, replacement):
                return True
    return False


class InformationBottleneck(nn.Module):
    def __init__(self, mean, std, device):
        super().__init__()
        self.device = device
        self.initial_value = 5.0
        self.std = torch.tensor(std, dtype=torch.float, device=device, requires_grad=False)
        self.mean = torch.tensor(mean, dtype=torch.float, device=device, requires_grad=False)
        self.alpha = nn.Parameter(
            torch.full((1, *self.mean.shape), fill_value=self.initial_value, device=device)
        )
        self.sigmoid = nn.Sigmoid()
        self.buffer_capacity = None
        self.reset_alpha()

    def reset_alpha(self):
        with torch.no_grad():
            self.alpha.fill_(self.initial_value)

    def forward(self, x, **kwargs):
        lamb = self.sigmoid(self.alpha)
        lamb = lamb.expand(x.shape[0], x.shape[1], -1)
        masked_mu = x * lamb
        masked_var = (1 - lamb) ** 2
        self.buffer_capacity = -0.5 * (1 + torch.log(masked_var) - masked_mu**2 - masked_var)
        noise_std = masked_var.sqrt()
        eps = masked_mu.data.new(masked_mu.size()).normal_()
        t = masked_mu + noise_std * eps
        return (t,)


def extract_bert_layer(model, layer_idx):
    for _, submodule in model.named_children():
        for n, s in submodule.named_children():
            if n in ("layers", "resblocks", "blocks"):
                for n2, s2 in s.named_children():
                    if n2 == str(layer_idx):
                        return s2
    return None


def vision_heatmap_iba(text_t, image_t, model, layer_idx, beta, var,
                       lr=1, train_steps=10, batch_size=5, device="cuda"):
    """Generate vision saliency heatmap via IBA."""
    with torch.no_grad():
        states = model.vision_model(image_t, output_hidden_states=True)
        features = states["hidden_states"][layer_idx + 1]

    layer = extract_bert_layer(model.vision_model, layer_idx)
    if layer is None:
        raise RuntimeError(f"Cannot find layer {layer_idx}")

    estimator = Estimator(layer)
    estimator.M = torch.zeros_like(features).cpu().numpy()
    estimator.S = var * np.ones(features.shape)
    estimator.N = 1

    bottleneck = InformationBottleneck(estimator.mean(), estimator.std(), device=device)
    sequential = mySequential(layer, bottleneck)
    replace_layer(model.vision_model, layer, sequential)

    batch_text = text_t.expand(batch_size, -1)
    batch_image = image_t.expand(batch_size, -1, -1, -1)
    optimizer = torch.optim.Adam(lr=lr, params=bottleneck.parameters())
    bottleneck.reset_alpha()
    model.eval()
    cos_sim = torch.nn.CosineSimilarity(eps=1e-6)

    for _ in range(train_steps):
        optimizer.zero_grad()
        text_feat = model.get_text_features(batch_text)
        image_feat = model.get_image_features(batch_image)
        compression = bottleneck.buffer_capacity.mean()
        fitting = cos_sim(text_feat, image_feat).mean()
        loss = beta * compression - fitting
        loss.backward()
        optimizer.step()

    saliency = bottleneck.buffer_capacity.mean(axis=0)
    saliency = torch.nansum(saliency, -1)[1:]  # drop CLS
    dim = int(saliency.numel() ** 0.5)
    saliency = saliency.reshape(1, 1, dim, dim)
    saliency = F.interpolate(saliency, size=224, mode="bilinear")
    saliency = saliency.squeeze().cpu().detach().numpy()

    replace_layer(model.vision_model, sequential, layer)
    return normalize(saliency)


def saliency_to_bbox(saliency, threshold=0.5, min_area=50):
    binary = (saliency > threshold).astype(np.uint8)
    if binary.sum() < min_area:
        binary = (saliency > 0.3).astype(np.uint8)
    if binary.sum() == 0:
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return [x, y, x + w, y + h]


def saliency_to_bbox_percentile(saliency, percentile=90, min_area=20):
    """Percentile-based thresholding: keep only top X% of saliency."""
    thresh = np.percentile(saliency, percentile)
    binary = (saliency > thresh).astype(np.uint8)
    if binary.sum() < min_area:
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return [x, y, x + w, y + h]


# ============================================================
# MedSAM helpers
# ============================================================

def prepare_slice_medsam(ct_slice):
    img_3c = np.repeat(ct_slice[:, :, None], 3, axis=-1)
    img_1024 = sk_transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True,
    ).astype(np.float32)
    vmin, vmax = img_1024.min(), img_1024.max()
    if vmax - vmin > 1e-8:
        img_1024 = (img_1024 - vmin) / (vmax - vmin)
    return img_1024

@torch.no_grad()
def encode_slice_medsam(model, ct_slice, device):
    img_1024 = prepare_slice_medsam(ct_slice)
    img_t = torch.tensor(img_1024).half().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda"):
        emb = model.image_encoder(img_t)
    del img_t
    return emb

@torch.no_grad()
def medsam_infer_slice(model, img_embed, box_1024, H, W, device, threshold=0.5):
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
    return (low_res_pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)


def prepare_slice_biomedclip(ct_slice):
    img_3c = np.repeat(ct_slice[:, :, None], 3, axis=-1)
    img_224 = sk_transform.resize(
        img_3c, (224, 224), order=3, preserve_range=True, anti_aliasing=True,
    ).astype(np.float32)
    vmin, vmax = img_224.min(), img_224.max()
    if vmax - vmin > 1e-8:
        img_224 = (img_224 - vmin) / (vmax - vmin)
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img_224 = (img_224 - mean) / std
    return torch.tensor(img_224, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


# ============================================================
# CLS-level slice ranking (fast, no IBA)
# ============================================================

@torch.no_grad()
def rank_slices_by_cls_similarity(clip_model, tokenizer, ct_vol, desc, device, top_k=8):
    """Rank slices by global CLS cosine similarity with text. Return sorted (z, sim) list."""
    text_ids = tokenizer(desc, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(device)
    text_feat = clip_model.get_text_features(text_ids)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    slice_scores = []
    D = ct_vol.shape[0]
    for z in range(D):
        s = ct_vol[z].astype(np.float32)
        if s.max() - s.min() < 1e-8:
            slice_scores.append((z, -1.0))
            continue
        img = prepare_slice_biomedclip(s).to(device)
        img_feat = clip_model.get_image_features(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ text_feat.T).item()
        slice_scores.append((z, sim))
        del img

    # Sort by similarity descending
    slice_scores.sort(key=lambda x: x[1], reverse=True)
    return slice_scores[:top_k]


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser("MedCLIP-SAMv2 v3 optimized text-guided inference")
    parser.add_argument("--npy_root", required=True)
    parser.add_argument("--medsam_ckpt", required=True)
    parser.add_argument("--out_dir", default="./results_medclip_medsam_v3")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vlayer", type=int, default=7)
    parser.add_argument("--vbeta", type=float, default=0.1)
    parser.add_argument("--vvar", type=float, default=1.0)
    parser.add_argument("--iba_steps", type=int, default=10)
    parser.add_argument("--tricks", type=str, default="",
                        help="Comma-separated: cls_filter,text_shorten,text_ensemble,"
                             "gamma_sharpen,percentile_thresh,high_medsam_thresh")
    args = parser.parse_args()

    tricks = set(t.strip() for t in args.tricks.split(",") if t.strip())
    print(f"Tricks: {tricks if tricks else 'none (baseline v2)'}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_root = Path(args.npy_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Load BiomedCLIP ---
    print("Loading BiomedCLIP (fine-tuned)...")
    from transformers import AutoModel, AutoTokenizer
    clip_model_dir = str(Path(__file__).parent / "MedCLIP-SAMv2" / "saliency_maps" / "model")
    clip_model = AutoModel.from_pretrained(clip_model_dir, trust_remote_code=True).to(device)
    clip_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )
    print("  BiomedCLIP loaded.")

    # --- Load MedSAM ---
    print("Loading MedSAM...")
    medsam_model = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt)
    medsam_model = medsam_model.to(device)
    medsam_model.image_encoder = medsam_model.image_encoder.half()
    medsam_model.eval()
    print("  MedSAM loaded.")

    # --- Config ---
    use_cls_filter = "cls_filter" in tricks
    use_text_shorten = "text_shorten" in tricks
    use_text_ensemble = "text_ensemble" in tricks
    use_gamma = "gamma_sharpen" in tricks
    use_percentile = "percentile_thresh" in tricks
    use_high_thresh = "high_medsam_thresh" in tricks

    cls_top_k = 10
    gamma = 2.0
    percentile = 90
    medsam_thresh = 0.6 if use_high_thresh else 0.5

    print(f"  CLS filter (top-{cls_top_k}): {use_cls_filter}")
    print(f"  Text shorten: {use_text_shorten}")
    print(f"  Text ensemble: {use_text_ensemble}")
    print(f"  Gamma sharpen ({gamma}): {use_gamma}")
    print(f"  Percentile thresh ({percentile}%): {use_percentile}")
    print(f"  MedSAM thresh: {medsam_thresh}")

    # --- Run ---
    results = []
    case_dirs = sorted(d for d in npy_root.iterdir() if d.is_dir())
    if args.max_cases > 0:
        case_dirs = case_dirs[:args.max_cases]

    t0 = time.time()
    for case_dir in tqdm(case_dirs, desc="Cases"):
        case_id = case_dir.name
        ct_path = case_dir / "ct.npy"
        mask_path = case_dir / "mask.npy"
        text_path = case_dir / "text.json"
        if not all(p.exists() for p in [ct_path, mask_path, text_path]):
            continue

        ct_vol = load_volume(ct_path).astype(np.float32)
        mask_vol = np.rint(load_volume(mask_path)).astype(np.int32)
        D, H, W = ct_vol.shape

        with open(text_path, "r", encoding="utf-8") as f:
            text_map = json.load(f)

        for lid_str, desc in text_map.items():
            lid = int(lid_str)
            gt_3d = (mask_vol == lid).astype(np.uint8)

            # --- Prepare text variants ---
            text_variants = [desc]  # always include original

            if use_text_shorten or use_text_ensemble:
                short = shorten_text(desc)
                if use_text_shorten and not use_text_ensemble:
                    text_variants = [short]
                elif use_text_ensemble:
                    keyword = extract_anatomy_keyword(desc)
                    text_variants = [desc, short, keyword]

            # Tokenize all variants
            all_text_ids = []
            for tv in text_variants:
                tids = tokenizer(
                    tv, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"].to(device)
                all_text_ids.append(tids)

            # --- Slice selection ---
            if use_cls_filter:
                # Use primary text for CLS ranking
                ranked = rank_slices_by_cls_similarity(
                    clip_model, tokenizer, ct_vol,
                    text_variants[0], device, top_k=cls_top_k,
                )
                candidate_slices = [z for z, sim in ranked if sim > -0.5]
            else:
                candidate_slices = list(range(D))

            pred_3d = np.zeros_like(gt_3d)
            slices_with_bbox = 0

            for z in candidate_slices:
                ct_slice = ct_vol[z]
                if ct_slice.max() - ct_slice.min() < 1e-8:
                    continue

                img_clip = prepare_slice_biomedclip(ct_slice).to(device)

                # --- Generate saliency (possibly multi-text ensemble) ---
                saliency_accum = np.zeros((224, 224), dtype=np.float64)
                n_variants = 0

                for tids in all_text_ids:
                    try:
                        sal = vision_heatmap_iba(
                            tids, img_clip, clip_model,
                            layer_idx=args.vlayer, beta=args.vbeta, var=args.vvar,
                            train_steps=args.iba_steps, batch_size=5, device=device,
                        )
                        saliency_accum += sal
                        n_variants += 1
                    except Exception:
                        continue

                del img_clip
                if n_variants == 0:
                    continue

                saliency = (saliency_accum / n_variants).astype(np.float32)
                saliency = normalize(saliency)

                # --- Gamma sharpening ---
                if use_gamma:
                    saliency = saliency ** gamma
                    saliency = normalize(saliency)

                # --- Bbox extraction ---
                if use_percentile:
                    bbox_224 = saliency_to_bbox_percentile(saliency, percentile=percentile)
                else:
                    bbox_224 = saliency_to_bbox(saliency, threshold=0.5)

                if bbox_224 is None:
                    continue

                slices_with_bbox += 1

                # Scale bbox
                scale_x = W / 224.0
                scale_y = H / 224.0
                bbox_orig = [
                    max(0, int(bbox_224[0] * scale_x) - 3),
                    max(0, int(bbox_224[1] * scale_y) - 3),
                    min(W, int(bbox_224[2] * scale_x) + 3),
                    min(H, int(bbox_224[3] * scale_y) + 3),
                ]

                # MedSAM
                emb = encode_slice_medsam(medsam_model, ct_slice, device)
                box_1024 = np.array(bbox_orig, dtype=float) / np.array([W, H, W, H]) * 1024
                pred_slice = medsam_infer_slice(
                    medsam_model, emb, box_1024[None, :], H, W, device,
                    threshold=medsam_thresh,
                )
                pred_3d[z] = pred_slice
                del emb

            torch.cuda.empty_cache()

            d = dice_score(pred_3d, gt_3d)
            iou = iou_score(pred_3d, gt_3d)
            results.append({
                "case_id": case_id,
                "mask_id": lid,
                "label_desc": desc[:120],
                "gt_voxels": int(gt_3d.sum()),
                "pred_voxels": int(pred_3d.sum()),
                "slices_with_bbox": slices_with_bbox,
                "dice": round(d, 4),
                "iou": round(iou, 4),
            })

    elapsed = time.time() - t0
    df = pd.DataFrame(results)
    trick_tag = "_".join(sorted(tricks)) if tricks else "baseline_v2"
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
    print(f"{'='*60}")
    print(f"Saved to: {csv_path}")


if __name__ == "__main__":
    main()
