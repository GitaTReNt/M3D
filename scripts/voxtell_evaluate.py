#!/usr/bin/env python3
"""
VoxTell zero-shot evaluation on M3D-RefSeg (raw + structured prompts).

Refactor from the naive version: VoxTell's ``embed_text_prompts`` moves
Qwen3-Embedding-4B (≈8 GB in fp16) between GPU and CPU on every call.
Doing that once per case fragments CPU RAM on Windows and OOMs after ~1
case on a 32 GB box. This script instead:

  1. Scans all cases to build the complete prompt inventory (dedup'd).
  2. Loads Qwen once, encodes every prompt in small chunks, then *deletes*
     the text backbone entirely — no repeated .to('cpu') calls.
  3. Runs per-case sliding-window inference with cached embeddings via
     the lower-level ``predict_sliding_window_return_logits``.

Usage:
  python scripts/voxtell_evaluate.py \
      --data_root data/M3D_RefSeg \
      --model_dir third_party/VoxTell/weights/voxtell_v1.1 \
      --out_dir results \
      --max_cases 0
"""

import argparse
import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from voxtell.inference.predictor import VoxTellPredictor
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image


FINDING_TYPES = [
    "lymph nodes", "lymph node",
    "cysts", "cyst",
    "masses", "mass",
    "tumors", "tumours", "tumor", "tumour",
    "nodules", "nodule",
    "calcifications", "calcification",
    "stones", "calculi", "calculus",
    "fluid collection", "effusion",
    "hematoma", "hemorrhage", "bleeding",
    "abscess",
    "opacities", "opacity", "infiltrate",
    "emphysema", "atelectasis",
    "fracture",
    "aneurysm",
    "thickening",
    "lesions", "lesion",
]

_NORMALIZE = {
    "lymph nodes": "lymph node",
    "cysts": "cyst",
    "masses": "mass",
    "tumors": "tumor", "tumours": "tumor", "tumour": "tumor",
    "nodules": "nodule",
    "calcifications": "calcification",
    "stones": "stone", "calculi": "calculus",
    "opacities": "opacity",
    "lesions": "lesion",
}


def extract_finding_type(text: str) -> str:
    tl = text.lower()
    for ft in FINDING_TYPES:
        if re.search(r"\b" + re.escape(ft) + r"\b", tl):
            return _NORMALIZE.get(ft, ft)
    return "lesion"


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return float(2.0 * (pred & gt).sum() / denom)


def encode_all_prompts_once(predictor: VoxTellPredictor,
                            unique_prompts: list,
                            chunk_size: int = 8) -> torch.Tensor:
    """Encode every prompt in one GPU residence. Frees Qwen afterwards.

    Returns a CPU tensor of shape (N_prompts, embed_dim) in fp16.
    """
    device = predictor.device
    predictor.text_backbone = predictor.text_backbone.to(device)
    all_embeds = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(unique_prompts), chunk_size),
                      desc="encoding prompts"):
            chunk = unique_prompts[i:i + chunk_size]
            wrapped = wrap_with_instruction(chunk)
            tokens = predictor.tokenizer(
                wrapped, padding=True, truncation=True,
                max_length=predictor.max_text_length, return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out = predictor.text_backbone(**tokens)
            embeds = last_token_pool(out.last_hidden_state,
                                     tokens['attention_mask'])
            all_embeds.append(embeds.detach().to('cpu', dtype=torch.float16))
            del out, embeds, tokens
    # Free the text backbone entirely — no `.to('cpu')` round-trip, which
    # is what was fragmenting CPU RAM on Windows.
    predictor.text_backbone = None
    predictor.tokenizer = None
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return torch.cat(all_embeds, dim=0)


def predict_with_cached_embeds(predictor: VoxTellPredictor,
                               img: np.ndarray,
                               case_embeds: torch.Tensor) -> np.ndarray:
    """Sliding-window inference using precomputed text embeddings."""
    data, bbox, orig_shape = predictor.preprocess(img)
    emb = case_embeds.unsqueeze(0).to(predictor.device)
    logits = predictor.predict_sliding_window_return_logits(data, emb).to('cpu')
    with torch.no_grad():
        pred = torch.sigmoid(logits.float()) > 0.5
    seg = np.zeros([pred.shape[0], *orig_shape], dtype=np.uint8)
    seg = insert_crop_into_image(seg, pred, bbox)
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/M3D_RefSeg")
    ap.add_argument("--model_dir",
                    default="third_party/VoxTell/weights/voxtell_v1.1")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--max_cases", type=int, default=0,
                    help="Limit number of cases (0 = all)")
    ap.add_argument("--chunk_size", type=int, default=8,
                    help="Prompt batch size for Qwen encoding")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    case_dirs = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and (d / "ct.nii.gz").exists() and (d / "text.json").exists()
    )
    if args.max_cases > 0:
        case_dirs = case_dirs[:args.max_cases]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading VoxTell predictor...")
    t0 = time.time()
    predictor = VoxTellPredictor(model_dir=args.model_dir, device=device)
    print(f"Predictor ready in {time.time()-t0:.1f}s")

    reader = NibabelIOWithReorient()

    # ---- Pass 1: scan dataset, build prompt inventory ----
    print(f"\nPass 1/2: scanning {len(case_dirs)} cases for prompts...")
    plan = []          # list of (case_dir, case_id, [(mode, lid, prompt_text)])
    unique_prompts = set()

    for case_dir in tqdm(case_dirs, desc="scan"):
        case_id = case_dir.name
        try:
            with open(case_dir / "text.json", encoding="utf-8") as f:
                text_map = json.load(f)
            seg_gt, _ = reader.read_seg(str(case_dir / "mask.nii.gz"))
            present = set(np.unique(np.rint(seg_gt[0]).astype(np.int32)).tolist())
        except Exception as e:
            print(f"[skip {case_id}] scan error: {e}")
            continue

        case_prompts = []
        for lid_str, text in text_map.items():
            try:
                lid = int(lid_str)
            except ValueError:
                continue
            if lid not in present:
                continue
            raw_p = text.strip()
            struct_p = extract_finding_type(raw_p)
            case_prompts.append(("raw", lid, raw_p))
            case_prompts.append(("structured", lid, struct_p))
            unique_prompts.add(raw_p)
            unique_prompts.add(struct_p)

        if case_prompts:
            plan.append((case_dir, case_id, case_prompts))
        del seg_gt

    unique_prompts = sorted(unique_prompts)
    prompt_to_idx = {p: i for i, p in enumerate(unique_prompts)}
    print(f"Plan: {len(plan)} cases, {len(unique_prompts)} unique prompts")

    # ---- Encode all prompts in a single Qwen residence ----
    print("\nEncoding all prompts (single GPU residence)...")
    t0 = time.time()
    all_embeds = encode_all_prompts_once(predictor, unique_prompts,
                                         chunk_size=args.chunk_size)
    print(f"Encoded {len(unique_prompts)} prompts in {time.time()-t0:.1f}s "
          f"(shape={tuple(all_embeds.shape)}, dtype={all_embeds.dtype})")
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info()
        print(f"GPU free after backbone release: {free/1e9:.1f} / {total/1e9:.1f} GB")

    # ---- Pass 2: per-case sliding-window inference ----
    print(f"\nPass 2/2: running inference on {len(plan)} cases...")
    raw_rows, struct_rows = [], []

    for case_dir, case_id, case_prompts in tqdm(plan, desc="inference"):
        try:
            img, _ = reader.read_images([str(case_dir / "ct.nii.gz")])
            seg_gt, _ = reader.read_seg(str(case_dir / "mask.nii.gz"))
            gt_full = np.rint(seg_gt[0]).astype(np.int32)
        except Exception as e:
            print(f"[skip {case_id}] read error: {e}")
            continue

        idxs = torch.tensor([prompt_to_idx[p[2]] for p in case_prompts])
        case_embeds = all_embeds[idxs]  # (n_case, embed_dim)

        t0 = time.time()
        try:
            seg_pred = predict_with_cached_embeds(predictor, img, case_embeds)
        except torch.cuda.OutOfMemoryError:
            print(f"[skip {case_id}] OOM with {len(case_prompts)} prompts")
            torch.cuda.empty_cache()
            continue
        dt = time.time() - t0

        for i, (mode, lid, prompt) in enumerate(case_prompts):
            pred = seg_pred[i].astype(bool)
            gt = (gt_full == lid)
            d = dice_score(pred, gt)
            row = {
                "case_id": case_id,
                "mask_id": lid,
                "prompt": prompt[:120],
                "pred_voxels": int(pred.sum()),
                "gt_voxels": int(gt.sum()),
                "dice": round(d, 4),
                "case_time_s": round(dt, 2),
            }
            (raw_rows if mode == "raw" else struct_rows).append(row)

        del seg_pred, img, seg_gt, gt_full
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---- Save CSVs + summary ----
    out_root = Path(args.out_dir)
    for tag, rows, subdir in [
        ("raw prompt", raw_rows, "10_voxtell_zeroshot_raw"),
        ("structured prompt", struct_rows, "11_voxtell_zeroshot_structured"),
    ]:
        outd = out_root / subdir
        outd.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        csv = outd / "voxtell_results.csv"
        df.to_csv(csv, index=False)
        print(f"\n=== VoxTell zero-shot — {tag} ===")
        if len(df) == 0:
            print("  (no results)")
            continue
        print(f"  N         = {len(df)}")
        print(f"  Mean Dice = {df['dice'].mean():.4f}")
        print(f"  Median    = {df['dice'].median():.4f}")
        print(f"  Dice>=0.5 = {(df['dice']>=0.5).sum()} "
              f"({(df['dice']>=0.5).mean()*100:.1f}%)")
        print(f"  Dice==0.0 = {(df['dice']==0.0).sum()} "
              f"({(df['dice']==0.0).mean()*100:.1f}%)")
        print(f"  Saved     -> {csv}")


if __name__ == "__main__":
    main()
