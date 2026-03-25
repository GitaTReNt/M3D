#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EDA: semantic vs morphology association for RefSeg

Inputs:
- --csv: M3D_RefSeg.csv (Image, Mask_ID, Question, Answer)
- --npy_root: M3D_RefSeg_npy (each case has mask.npy + text.json)

This script:
1) Build region-level table: region_id = case_id__Mask_ID
2) Extract text features from:
   - label_desc (from per-case text.json by Mask_ID)
   - questions (5 paraphrases per region)
3) Extract morphology features from mask.npy:
   - voxels, bbox dims, slices_covered
   - components (optional if scipy exists)
   - surface_to_volume (optional if scipy exists)
4) Bucket by size and complexity; summarize text feature distributions per bucket
Outputs:
- tables/region_sem_morph_features.csv
- tables/bucket_summary_size.csv
- tables/bucket_summary_complexity.csv
- figs/ laterality_rate_by_size_bucket.png, desc_words_box_by_size_bucket.png, etc.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------
# helpers
# -----------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")

def infer_case_id(image_path: str) -> str:
    s = str(image_path).replace("\\", "/")
    m = re.search(r"(s\d+)", s)
    if m:
        return m.group(1)
    parts = [p for p in s.split("/") if p]
    return parts[0] if parts else s

def word_count(s: str) -> int:
    s = str(s)
    return len([w for w in re.split(r"\s+", s.strip()) if w])

UNCERTAINTY = re.compile(r"\b(possible|possibly|probable|probably|likely|consider|suggest|suspect|maybe)\b", re.I)
NEGATION = re.compile(r"\b(no|not|without|absent|negative)\b", re.I)
LATERALITY = re.compile(r"\b(left|right|bilateral)\b", re.I)

def has_pat(pat: re.Pattern, s: str) -> int:
    return int(bool(pat.search(str(s))))

def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    # Spearman via rank + Pearson, no scipy needed
    x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def load_text_json(case_dir: Path) -> Dict[str, str]:
    p = case_dir / "text.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # sometimes BOM or encoding issues
        return json.loads(p.read_text(encoding="utf-8-sig"))

def load_mask(case_dir: Path) -> np.ndarray:
    p = case_dir / "mask.npy"
    if not p.exists():
        raise FileNotFoundError(str(p))
    m = np.load(p)
    if m.ndim == 4 and m.shape[0] == 1:
        m = m[0]
    # round to int labels (prepare saved float)
    m = np.rint(m).astype(np.int32, copy=False)
    return m  # (Z,H,W)


def compute_bbox_dims(binary: np.ndarray) -> Tuple[int, int, int]:
    """Return bbox dims (d,h,w). If empty -> (0,0,0)."""
    if binary.sum() == 0:
        return (0, 0, 0)
    z_any = np.where(binary.any(axis=(1, 2)))[0]
    y_any = np.where(binary.any(axis=(0, 2)))[0]
    x_any = np.where(binary.any(axis=(0, 1)))[0]
    d = int(z_any[-1] - z_any[0] + 1)
    h = int(y_any[-1] - y_any[0] + 1)
    w = int(x_any[-1] - x_any[0] + 1)
    return (d, h, w)

def compute_centroid(binary: np.ndarray) -> Tuple[float, float, float]:
    """Centroid in (z,y,x) normalized to [0,1]. If empty -> (nan,nan,nan)."""
    if binary.sum() == 0:
        return (float("nan"), float("nan"), float("nan"))
    coords = np.argwhere(binary)
    z, y, x = coords.mean(axis=0)
    Z, H, W = binary.shape
    return (float(z / max(Z - 1, 1)), float(y / max(H - 1, 1)), float(x / max(W - 1, 1)))

def compute_components(binary: np.ndarray) -> float:
    """3D connected components count (optional)."""
    try:
        from scipy.ndimage import label
        _, n = label(binary.astype(np.uint8))
        return float(n)
    except Exception:
        return float("nan")

def compute_surface_to_volume(binary: np.ndarray) -> float:
    """
    Approximate surface voxels using binary erosion (optional).
    surface = vol - eroded(vol)
    """
    v = int(binary.sum())
    if v == 0:
        return float("nan")
    try:
        from scipy.ndimage import binary_erosion
        interior = binary_erosion(binary, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
        surface = binary & (~interior)
        s = int(surface.sum())
        return float(s / v)
    except Exception:
        return float("nan")


def bucket_by_quantiles(values: pd.Series, labels: List[str]) -> pd.Series:
    """
    Bucket numeric series by quantiles into len(labels) bins.
    Values must be non-negative; NaN allowed.
    """
    v = values.copy()
    out = pd.Series(["unknown"] * len(v), index=v.index, dtype="object")

    mask = np.isfinite(v.to_numpy()) & (v.to_numpy() >= 0)
    vv = v[mask]
    if vv.nunique() <= 1:
        out[mask] = labels[0]
        return out

    qs = np.linspace(0, 1, num=len(labels) + 1)
    edges = vv.quantile(qs).to_numpy()
    # make edges strictly increasing
    edges = np.maximum.accumulate(edges)
    # if duplicates, jitter tiny
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    # assign
    bins = pd.cut(vv, bins=edges, labels=labels, include_lowest=True, duplicates="drop")
    out.loc[bins.index] = bins.astype(str)
    return out


# -----------------------
# main
# -----------------------

def main():
    ap = argparse.ArgumentParser("EDA semantic vs morphology for RefSeg")
    ap.add_argument("--csv", required=True, help="Path to M3D_RefSeg.csv")
    ap.add_argument("--npy_root", required=True, help="Root dir of M3D_RefSeg_npy (sXXXX folders)")
    ap.add_argument("--out_dir", default="./sem_morph_out")
    ap.add_argument("--max_cases", type=int, default=-1, help="Process at most N cases (debug). -1 = all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    tables = out_dir / "tables"
    figs = out_dir / "figs"
    ensure_dir(tables)
    ensure_dir(figs)

    df = safe_read_csv(Path(args.csv))
    if not {"Image", "Mask_ID", "Question", "Answer"}.issubset(df.columns):
        raise ValueError("CSV must contain Image, Mask_ID, Question, Answer")

    df["case_id"] = df["Image"].map(infer_case_id)
    df["Mask_ID"] = df["Mask_ID"].astype(int)
    df["region_id"] = df["case_id"].astype(str) + "__" + df["Mask_ID"].astype(str)

    # region-level text aggregation
    g = df.groupby("region_id")
    region_text = g.agg(
        case_id=("case_id", "first"),
        Mask_ID=("Mask_ID", "first"),
        n_rows=("region_id", "size"),
        question_words_mean=("Question", lambda s: float(np.mean([word_count(x) for x in s]))),
        answer_words_mean=("Answer", lambda s: float(np.mean([word_count(x) for x in s]))),
        q_unc_rate=("Question", lambda s: float(np.mean([has_pat(UNCERTAINTY, x) for x in s]))),
        q_neg_rate=("Question", lambda s: float(np.mean([has_pat(NEGATION, x) for x in s]))),
        q_lat_rate=("Question", lambda s: float(np.mean([has_pat(LATERALITY, x) for x in s]))),
        q_unc_any=("Question", lambda s: int(any(has_pat(UNCERTAINTY, x) for x in s))),
        q_neg_any=("Question", lambda s: int(any(has_pat(NEGATION, x) for x in s))),
        q_lat_any=("Question", lambda s: int(any(has_pat(LATERALITY, x) for x in s))),
    ).reset_index()

    # load label_desc from per-case text.json
    npy_root = Path(args.npy_root)
    needed = region_text[["case_id", "Mask_ID"]].drop_duplicates()
    case_to_ids = needed.groupby("case_id")["Mask_ID"].apply(list).to_dict()

    rng = np.random.default_rng(args.seed)
    case_ids = sorted(case_to_ids.keys())
    if args.max_cases and args.max_cases > 0 and len(case_ids) > args.max_cases:
        case_ids = sorted(rng.choice(case_ids, size=args.max_cases, replace=False).tolist())

    morph_rows = []
    label_desc_rows = []

    for case_id in tqdm(case_ids, desc="Computing morphology + label_desc"):
        case_dir = npy_root / case_id
        if not case_dir.exists():
            continue

        # label desc map
        desc_map = load_text_json(case_dir)

        # mask
        try:
            mask = load_mask(case_dir)  # (Z,H,W)
        except Exception:
            continue

        for mid in case_to_ids.get(case_id, []):
            desc = desc_map.get(str(mid), "")
            label_desc_rows.append({
                "case_id": case_id,
                "Mask_ID": int(mid),
                "label_desc": desc,
                "label_desc_words": int(word_count(desc)),
                "label_desc_has_lat": int(has_pat(LATERALITY, desc)),
                "label_desc_has_unc": int(has_pat(UNCERTAINTY, desc)),
                "label_desc_has_neg": int(has_pat(NEGATION, desc)),
            })

            binary = (mask == int(mid))
            vox = int(binary.sum())
            d, h, w = compute_bbox_dims(binary)
            slices_cov = int(binary.any(axis=(1, 2)).sum()) if vox > 0 else 0
            cz, cy, cx = compute_centroid(binary)
            comp = compute_components(binary) if vox > 0 else float("nan")
            stv = compute_surface_to_volume(binary) if vox > 0 else float("nan")

            morph_rows.append({
                "case_id": case_id,
                "Mask_ID": int(mid),
                "voxels": vox,
                "bbox_d": d,
                "bbox_h": h,
                "bbox_w": w,
                "slices_covered": slices_cov,
                "centroid_z_norm": cz,
                "centroid_y_norm": cy,
                "centroid_x_norm": cx,
                "components": comp,
                "surface_to_volume": stv,
                "empty": int(vox == 0),
            })

    morph = pd.DataFrame(morph_rows)
    desc_df = pd.DataFrame(label_desc_rows)

    # merge to region-level
    region = region_text.merge(morph, on=["case_id", "Mask_ID"], how="left").merge(
        desc_df, on=["case_id", "Mask_ID"], how="left"
    )

    # buckets
    # size bucket (exclude empty for quantiles)
    size_labels = ["Q1_small", "Q2_medium", "Q3_large", "Q4_xlarge"]
    size_bucket = bucket_by_quantiles(region.loc[region["empty"] == 0, "voxels"], size_labels)
    region["size_bucket"] = "empty"
    region.loc[size_bucket.index, "size_bucket"] = size_bucket.values
    region.loc[region["empty"] == 1, "size_bucket"] = "empty"

    # complexity bucket: by surface_to_volume quantiles (if available) else by components
    if region["surface_to_volume"].notna().sum() >= 10:
        comp_labels = ["low_complex", "mid_complex", "high_complex"]
        cb = bucket_by_quantiles(region["surface_to_volume"], comp_labels)
        region["complexity_bucket"] = cb
    else:
        # fallback
        region["complexity_bucket"] = np.where(region["components"] > 1, "multi_component", "single_component")

    # summaries
    def summarize(group_col: str) -> pd.DataFrame:
        out = region.groupby(group_col).agg(
            n_regions=("region_id", "nunique"),
            n_cases=("case_id", "nunique"),
            empty_rate=("empty", "mean"),
            voxels_mean=("voxels", "mean"),
            slices_mean=("slices_covered", "mean"),
            components_mean=("components", "mean"),
            stv_mean=("surface_to_volume", "mean"),
            label_desc_words_mean=("label_desc_words", "mean"),
            q_words_mean=("question_words_mean", "mean"),
            a_words_mean=("answer_words_mean", "mean"),
            q_lat_any_rate=("q_lat_any", "mean"),
            q_unc_any_rate=("q_unc_any", "mean"),
            q_neg_any_rate=("q_neg_any", "mean"),
            desc_lat_rate=("label_desc_has_lat", "mean"),
        ).reset_index()
        return out.sort_values(group_col)

    sum_size = summarize("size_bucket")
    sum_cplx = summarize("complexity_bucket")

    region.to_csv(tables / "region_sem_morph_features.csv", index=False)
    sum_size.to_csv(tables / "bucket_summary_size.csv", index=False)
    sum_cplx.to_csv(tables / "bucket_summary_complexity.csv", index=False)

    # correlations (proxy: does language become "more explicit" when morphology changes?)
    corr_report = {
        "spearman_logvox_vs_label_desc_words": spearman_corr(np.log1p(region["voxels"].fillna(0).to_numpy()),
                                                            region["label_desc_words"].fillna(0).to_numpy()),
        "spearman_logvox_vs_question_words_mean": spearman_corr(np.log1p(region["voxels"].fillna(0).to_numpy()),
                                                               region["question_words_mean"].fillna(0).to_numpy()),
        "spearman_stv_vs_question_words_mean": spearman_corr(region["surface_to_volume"].fillna(0).to_numpy(),
                                                            region["question_words_mean"].fillna(0).to_numpy()),
    }
    (tables / "semantic_morphology_correlations.json").write_text(
        json.dumps(corr_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # plots
    import matplotlib.pyplot as plt

    # laterality rate by size bucket
    plt.figure(figsize=(8, 4.8))
    x = sum_size["size_bucket"].astype(str).tolist()
    y = sum_size["q_lat_any_rate"].to_numpy()
    plt.bar(x, y)
    plt.title("Laterality presence (any question) by ROI size bucket")
    plt.ylabel("rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figs / "laterality_rate_by_size_bucket.png", dpi=220)
    plt.close()

    # label_desc_words boxplot by size bucket
    plt.figure(figsize=(8, 4.8))
    order = [b for b in size_labels if b in region["size_bucket"].unique()] + (["empty"] if "empty" in region["size_bucket"].unique() else [])
    data = [region.loc[region["size_bucket"] == b, "label_desc_words"].fillna(0).to_numpy() for b in order]
    plt.boxplot(data, labels=order, showfliers=False)
    plt.title("Label description length by ROI size bucket")
    plt.ylabel("label_desc_words")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figs / "desc_words_box_by_size_bucket.png", dpi=220)
    plt.close()

    # scatter: log1p voxels vs label_desc_words
    plt.figure(figsize=(6.6, 5.2))
    xv = np.log1p(region["voxels"].fillna(0).to_numpy())
    yv = region["label_desc_words"].fillna(0).to_numpy()
    plt.scatter(xv, yv, s=10)
    plt.title("log1p(voxels) vs label_desc_words")
    plt.xlabel("log1p(voxels)")
    plt.ylabel("label_desc_words")
    plt.tight_layout()
    plt.savefig(figs / "scatter_logvox_vs_desc_words.png", dpi=220)
    plt.close()

    print("Done. Outputs:", out_dir.resolve())


if __name__ == "__main__":
    main()
