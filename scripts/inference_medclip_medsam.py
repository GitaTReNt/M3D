#!/usr/bin/env python3
"""
MedCLIP-SAMv2 + MedSAM pipeline for text-guided segmentation on M3D-RefSeg.

Pipeline per slice:
  1. BiomedCLIP encodes text + image jointly
  2. M2IB (Information Bottleneck) generates saliency map: which pixels match the text
  3. Saliency map → bounding box
  4. BBox → MedSAM → segmentation mask

Usage:
  python inference_medclip_medsam.py \
      --npy_root D:/M3D/M3D_RefSeg_npy \
      --medsam_ckpt D:/M3D/MedSAM/work_dir/MedSAM_finetuned/medsam_vit_b.pth \
      --max_cases 10 --device cuda:0
"""

import argparse
import json
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

# MedSAM
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "MedSAM"))
from segment_anything import sam_model_registry

# HuggingFace (for BiomedCLIP)
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "MedCLIP-SAMv2" / "saliency_maps"))


# ============================================================
# BiomedCLIP + IBA saliency (adapted from MedCLIP-SAMv2)
# ============================================================

def normalize(x):
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-8:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


# --- IBA (Information Bottleneck Attribution) ---

class Estimator:
    def __init__(self, layer):
        self.layer = layer
        self.M = None
        self.S = None
        self.N = None
        self.num_seen = 0
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
            if n == "layers" or n == "resblocks" or n == "blocks":
                for n2, s2 in s.named_children():
                    if n2 == str(layer_idx):
                        return s2
    # For BiomedCLIP text encoder: transformer.layer.{idx}
    for _, submodule in model.named_children():
        for n, s in submodule.named_children():
            if n == "layer":
                for n2, s2 in s.named_children():
                    if n2 == str(layer_idx):
                        return s2
    return None


def vision_heatmap_iba(text_t, image_t, model, layer_idx, beta, var,
                       lr=1, train_steps=10, batch_size=10, device="cuda"):
    """Generate vision saliency heatmap via IBA."""
    # Extract feature map at target layer
    with torch.no_grad():
        states = model.vision_model(image_t, output_hidden_states=True)
        features = states["hidden_states"][layer_idx + 1]

    layer = extract_bert_layer(model.vision_model, layer_idx)
    if layer is None:
        raise RuntimeError(f"Cannot find layer {layer_idx} in vision model")

    # Create estimator
    estimator = Estimator(layer)
    estimator.M = torch.zeros_like(features).cpu().numpy()
    estimator.S = var * np.ones(features.shape)
    estimator.N = 1

    # Create bottleneck
    bottleneck = InformationBottleneck(estimator.mean(), estimator.std(), device=device)
    sequential = mySequential(layer, bottleneck)

    # Replace layer with bottleneck
    replace_layer(model.vision_model, layer, sequential)

    # Optimize
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

    # Extract saliency
    saliency = bottleneck.buffer_capacity.mean(axis=0)
    saliency = torch.nansum(saliency, -1)[1:]  # drop CLS token
    dim = int(saliency.numel() ** 0.5)
    saliency = saliency.reshape(1, 1, dim, dim)
    saliency = F.interpolate(saliency, size=224, mode="bilinear")
    saliency = saliency.squeeze().cpu().detach().numpy()

    # Restore original layer
    replace_layer(model.vision_model, sequential, layer)

    return normalize(saliency)


def saliency_to_bbox(saliency, threshold=0.5, min_area=50):
    """Convert saliency map to bounding box. Returns [x_min, y_min, x_max, y_max] or None."""
    binary = (saliency > threshold).astype(np.uint8)
    if binary.sum() < min_area:
        # Try lower threshold
        binary = (saliency > 0.3).astype(np.uint8)
    if binary.sum() == 0:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Take largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return [x, y, x + w, y + h]


# ============================================================
# MedSAM inference helpers
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


# ============================================================
# Main pipeline
# ============================================================

def prepare_slice_biomedclip(ct_slice):
    """Convert CT slice [0,1] float32 (H,W) to BiomedCLIP input (1,3,224,224)."""
    img_3c = np.repeat(ct_slice[:, :, None], 3, axis=-1)
    img_224 = sk_transform.resize(
        img_3c, (224, 224), order=3, preserve_range=True, anti_aliasing=True,
    ).astype(np.float32)
    vmin, vmax = img_224.min(), img_224.max()
    if vmax - vmin > 1e-8:
        img_224 = (img_224 - vmin) / (vmax - vmin)
    # Normalize with ImageNet stats (BiomedCLIP uses standard normalization)
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img_224 = (img_224 - mean) / std
    return torch.tensor(img_224, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser("MedCLIP-SAMv2 + MedSAM text-guided inference")
    parser.add_argument("--npy_root", required=True)
    parser.add_argument("--medsam_ckpt", required=True)
    parser.add_argument("--out_dir", default="./results_medclip_medsam")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    # IBA hyperparams
    parser.add_argument("--vlayer", type=int, default=7)
    parser.add_argument("--vbeta", type=float, default=0.1)
    parser.add_argument("--vvar", type=float, default=1.0)
    parser.add_argument("--iba_steps", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_root = Path(args.npy_root)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Load BiomedCLIP (HuggingFace format with DHN-NCE fine-tuned weights) ---
    print("Loading BiomedCLIP (fine-tuned)...")
    from transformers import AutoModel, AutoTokenizer
    clip_model_dir = str(Path(__file__).parent.parent / "third_party" / "MedCLIP-SAMv2" / "saliency_maps" / "model")
    clip_wrapper = AutoModel.from_pretrained(clip_model_dir, trust_remote_code=True).to(device)
    clip_wrapper.eval()
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

    # --- Run inference ---
    results = []
    case_dirs = sorted(d for d in npy_root.iterdir() if d.is_dir())
    if args.max_cases > 0:
        case_dirs = case_dirs[:args.max_cases]

    print(f"Running on {len(case_dirs)} cases...")
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

            # Tokenize text once
            text_tokens = tokenizer(
                desc, return_tensors="pt", truncation=True, max_length=256
            )["input_ids"].to(device)

            pred_3d = np.zeros_like(gt_3d)
            slices_with_bbox = 0

            # Pass 1: compute saliency for all slices and rank by focus
            slice_saliencies = []
            for z in range(D):
                ct_slice = ct_vol[z]
                if ct_slice.max() - ct_slice.min() < 1e-8:
                    continue

                img_clip = prepare_slice_biomedclip(ct_slice).to(device)
                try:
                    saliency = vision_heatmap_iba(
                        text_tokens, img_clip, clip_wrapper,
                        layer_idx=args.vlayer, beta=args.vbeta, var=args.vvar,
                        train_steps=args.iba_steps, batch_size=5, device=device,
                    )
                except Exception as e:
                    del img_clip
                    continue
                del img_clip

                bbox_224 = saliency_to_bbox(saliency, threshold=0.5)
                if bbox_224 is None:
                    continue

                # Focus score: how concentrated is the saliency?
                # High peak + low coverage = focused = good
                coverage = (saliency > 0.3).sum() / saliency.size
                peak = saliency.max()
                focus_score = peak * (1.0 - coverage)

                slice_saliencies.append((z, saliency, bbox_224, focus_score))

            # Pass 2: select top slices and segment
            # Sort by focus score, keep top fraction
            if slice_saliencies:
                slice_saliencies.sort(key=lambda x: x[3], reverse=True)
                # Keep at most half the slices (best focused ones)
                max_slices = max(1, len(slice_saliencies) // 2)
                selected = slice_saliencies[:max_slices]

                scale_x = W / 224.0
                scale_y = H / 224.0

                for z, saliency, bbox_224, focus in selected:
                    slices_with_bbox += 1
                    bbox_orig = [
                        max(0, int(bbox_224[0] * scale_x) - 3),
                        max(0, int(bbox_224[1] * scale_y) - 3),
                        min(W, int(bbox_224[2] * scale_x) + 3),
                        min(H, int(bbox_224[3] * scale_y) + 3),
                    ]

                    emb = encode_slice_medsam(medsam_model, ct_vol[z], device)
                    box_1024 = np.array(bbox_orig, dtype=float) / np.array([W, H, W, H]) * 1024
                    box_1024 = box_1024[None, :]
                    pred_slice = medsam_infer_slice(medsam_model, emb, box_1024, H, W, device)
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
    csv_path = out_dir / "medclip_medsam_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nDone in {elapsed:.1f}s")
    print(f"{'='*60}")
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
