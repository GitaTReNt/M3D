#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deeper EDA for M3D-RefSeg (built on top of eda_m3d_refseg_npy.py outputs)

What it does (mirrors "what others did" idea from meeting, but for RefSeg):
1) Quantify empty masks and long-tail ROI sizes; bucket statistics and plots
2) Diagnose empty root cause: erased by prepare vs missing in original (id mismatch)
3) Laterality vs centroid consistency (AUC test + plots)
4) Text template/duplication analysis (question normalization + template coverage)
5) Quantify "text separability" via simple classifiers (TF-IDF -> anatomy/laterality)
6) Intensity & ROI slice coverage distribution (sampling from ct.npy/mask.npy)

Inputs:
- Base EDA outputs:
  eda_out/tables/regions_text_summary.csv
  eda_out/tables/regions_image_features.csv
  (optional) eda_out/tables/empty_regions_after_prepare.csv

- Original CSV (for template analysis, and mapping to original mask path):
  M3D_RefSeg.csv

- Optional: orig_root (to read original mask.nii.gz for empty root cause)

Run example:
  python eda_refseg_deeper.py ^
    --csv "D:\M3D\M3D_RefSeg\M3D_RefSeg.csv" ^
    --eda_out ".\eda_out" ^
    --npy_root "D:\M3D\M3D_RefSeg_npy" ^
    --orig_root "D:\M3D\M3D_RefSeg" ^
    --out_dir ".\eda_deeper_out" ^
    --do_all
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# -------------------------
# Helpers
# -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_case_id(path_str: str) -> str:
    parts = str(path_str).replace("\\", "/").split("/")
    return parts[0] if parts else str(path_str)

def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def plot_bar_from_df(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str, rotate: int = 45):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4.8))
    plt.bar(df[x_col].astype(str).tolist(), df[y_col].tolist())
    plt.xticks(rotation=rotate, ha="right")
    plt.title(title)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_hist(values: np.ndarray, out_path: Path, title: str, xlabel: str, bins: int = 60, log1p: bool = False):
    import matplotlib.pyplot as plt
    v = values[np.isfinite(values)]
    if log1p:
        v = np.log1p(v)
    plt.figure(figsize=(7, 4.6))
    plt.hist(v, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel + (" (log1p)" if log1p else ""))
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_box_by_group(df: pd.DataFrame, value_col: str, group_col: str, out_path: Path, title: str):
    import matplotlib.pyplot as plt
    groups = []
    labels = []
    for g, sub in df.groupby(group_col):
        vals = sub[value_col].dropna().values
        if len(vals) == 0:
            continue
        groups.append(vals)
        labels.append(str(g))
    if not groups:
        return
    plt.figure(figsize=(9, 4.8))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def normalize_question(q: str) -> str:
    q = str(q).lower().strip()
    q = re.sub(r"\d+", "<num>", q)
    q = re.sub(r"[^a-z0-9<>\s]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def remove_laterality_tokens(q: str) -> str:
    q = re.sub(r"\b(left|right|bilateral)\b", "<lat>", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def voxel_bucket(v: float) -> str:
    # bins tuned for resized grid; includes empty=0
    if not np.isfinite(v):
        return "NA"
    v = float(v)
    edges = [0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    labels = ["0", "1-9", "10-99", "100-999", "1k-9k", "10k-99k", "100k-999k", ">=1M"]
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return labels[i]
    return labels[-1]


# -------------------------
# Core analyses
# -------------------------

def load_base_tables(eda_out: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    tdir = eda_out / "tables"
    text_path = tdir / "regions_text_summary.csv"
    img_path = tdir / "regions_image_features.csv"
    empty_path = tdir / "empty_regions_after_prepare.csv"

    if not text_path.exists() or not img_path.exists():
        raise FileNotFoundError(
            "Missing base tables. Please run eda_m3d_refseg_npy.py first with "
            "--text_eda --compute_image_features (and preferably --compute_intensity). "
            f"Expected:\n  {text_path}\n  {img_path}"
        )

    region_text = pd.read_csv(text_path)
    region_img = pd.read_csv(img_path)
    empty_df = pd.read_csv(empty_path) if empty_path.exists() else None
    return region_text, region_img, empty_df

def merge_region_tables(region_text: pd.DataFrame, region_img: pd.DataFrame) -> pd.DataFrame:
    # normalize keys
    for df in (region_text, region_img):
        if "Mask_ID" in df.columns:
            df["Mask_ID"] = df["Mask_ID"].astype(int)
        if "case_id" in df.columns:
            df["case_id"] = df["case_id"].astype(str)
        if "region_id" in df.columns:
            df["region_id"] = df["region_id"].astype(str)

    merged = region_text.merge(region_img, on=["region_id", "case_id", "Mask_ID"], how="left", suffixes=("", "_img"))
    return merged

def analyze_empty_and_buckets(merged: pd.DataFrame, out_tables: Path, out_figs: Path) -> None:
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    if "empty" not in merged.columns:
        print("[WARN] 'empty' column missing in merged table. Skipping empty analysis.")
        return

    merged["voxels_bucket"] = merged["voxels"].map(voxel_bucket)

    overall = {
        "n_regions": int(len(merged)),
        "n_empty": int((merged["empty"] == 1).sum()),
        "empty_rate": float((merged["empty"] == 1).mean()),
        "voxels_p50": float(np.nanpercentile(merged["voxels"], 50)),
        "voxels_p90": float(np.nanpercentile(merged["voxels"], 90)),
        "voxels_p95": float(np.nanpercentile(merged["voxels"], 95)),
    }
    save_json(overall, out_tables / "empty_overall_summary.json")

    by_anat = (
        merged.groupby("anatomy_group", dropna=False)
        .agg(n=("region_id", "size"), n_empty=("empty", lambda x: int((x == 1).sum())))
        .reset_index()
    )
    by_anat["empty_rate"] = by_anat["n_empty"] / by_anat["n"].replace(0, np.nan)
    by_anat = by_anat.sort_values("empty_rate", ascending=False)
    by_anat.to_csv(out_tables / "empty_rate_by_anatomy.csv", index=False)
    plot_bar_from_df(by_anat, "anatomy_group", "empty_rate", out_figs / "empty_rate_by_anatomy.png",
                     "Empty rate by anatomy_group")

    by_bucket = (
        merged.groupby("voxels_bucket", dropna=False)
        .agg(n=("region_id", "size"), n_empty=("empty", lambda x: int((x == 1).sum())))
        .reset_index()
    )
    by_bucket["empty_rate"] = by_bucket["n_empty"] / by_bucket["n"].replace(0, np.nan)
    # order buckets nicely
    bucket_order = ["0", "1-9", "10-99", "100-999", "1k-9k", "10k-99k", "100k-999k", ">=1M", "NA"]
    by_bucket["voxels_bucket"] = pd.Categorical(by_bucket["voxels_bucket"], categories=bucket_order, ordered=True)
    by_bucket = by_bucket.sort_values("voxels_bucket")
    by_bucket.to_csv(out_tables / "empty_rate_by_voxel_bucket.csv", index=False)
    plot_bar_from_df(by_bucket, "voxels_bucket", "empty_rate", out_figs / "empty_rate_by_voxel_bucket.png",
                     "Empty rate by ROI voxel bucket", rotate=0)

    # ROI size distribution
    plot_hist(merged["voxels"].to_numpy(dtype=float), out_figs / "roi_voxels_hist.png",
              "ROI voxel count distribution", "voxels", bins=60, log1p=False)
    plot_hist(merged["voxels"].to_numpy(dtype=float), out_figs / "roi_voxels_hist_log1p.png",
              "ROI voxel count distribution", "voxels", bins=60, log1p=True)

def diagnose_empty_root_cause(
    csv_path: Path,
    empty_regions: pd.DataFrame,
    orig_root: Path,
    npy_root: Path,
    out_tables: Path,
    out_figs: Path,
) -> None:
    """
    For each empty region (npy voxels == 0), check whether original mask.nii.gz contains Mask_ID voxels.
    -> erased_by_prepare vs missing_in_original_or_id_mismatch
    """
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    try:
        import nibabel as nib
    except Exception as e:
        print("[WARN] nibabel not available, cannot run root cause analysis:", e)
        return

    df = pd.read_csv(csv_path)
    df["case_id"] = df["Image"].map(parse_case_id)
    df["Mask_ID"] = df["Mask_ID"].astype(int)
    df["region_id"] = df["case_id"].astype(str) + "__" + df["Mask_ID"].astype(str)

    # map region -> original mask relative path
    map_df = df[["case_id", "Mask_ID", "region_id", "Mask"]].drop_duplicates()

    targets = empty_regions[["case_id", "Mask_ID", "region_id"]].drop_duplicates()
    targets["Mask_ID"] = targets["Mask_ID"].astype(int)
    targets = targets.merge(map_df, on=["case_id", "Mask_ID", "region_id"], how="left")

    cache_orig_mask: Dict[str, np.ndarray] = {}

    rows = []
    for case_id, mask_id, region_id, mask_rel in tqdm(
        targets[["case_id", "Mask_ID", "region_id", "Mask"]].itertuples(index=False),
        total=len(targets),
        desc="Empty root-cause check",
    ):
        case_id = str(case_id)
        mask_id = int(mask_id)

        orig_mask_path = orig_root / str(mask_rel) if isinstance(mask_rel, str) else None
        npy_mask_path = npy_root / case_id / "mask.npy"

        if orig_mask_path is None or not orig_mask_path.exists():
            rows.append({
                "region_id": region_id, "case_id": case_id, "Mask_ID": mask_id,
                "status": "missing_original_mask_file",
                "orig_mask_path": str(orig_mask_path) if orig_mask_path else "",
                "npy_mask_path": str(npy_mask_path),
                "orig_voxels": np.nan, "npy_voxels": 0,
                "cause": "unknown"
            })
            continue

        # load original mask once per case
        if case_id not in cache_orig_mask:
            try:
                arr = nib.load(str(orig_mask_path)).get_fdata()
                arr = np.rint(arr).astype(np.int32)
                cache_orig_mask[case_id] = arr
            except Exception as e:
                rows.append({
                    "region_id": region_id, "case_id": case_id, "Mask_ID": mask_id,
                    "status": f"orig_load_error:{e}",
                    "orig_mask_path": str(orig_mask_path),
                    "npy_mask_path": str(npy_mask_path),
                    "orig_voxels": np.nan, "npy_voxels": 0,
                    "cause": "unknown"
                })
                continue

        orig_mask = cache_orig_mask[case_id]
        orig_vox = int((orig_mask == mask_id).sum())

        if orig_vox > 0:
            cause = "erased_by_prepare"
        else:
            cause = "missing_in_original_or_id_mismatch"

        rows.append({
            "region_id": region_id, "case_id": case_id, "Mask_ID": mask_id,
            "status": "ok",
            "orig_mask_path": str(orig_mask_path),
            "npy_mask_path": str(npy_mask_path),
            "orig_voxels": orig_vox,
            "npy_voxels": 0,
            "cause": cause
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_tables / "empty_root_cause_report.csv", index=False)

    summary = out["cause"].value_counts(dropna=False).to_dict()
    save_json(summary, out_tables / "empty_root_cause_summary.json")

    # bar plot
    import matplotlib.pyplot as plt
    items = list(summary.items())
    labels, counts = zip(*items) if items else ([], [])
    plt.figure(figsize=(7, 4.6))
    plt.bar([str(x) for x in labels], list(counts))
    plt.xticks(rotation=30, ha="right")
    plt.title("Empty root cause counts")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_figs / "empty_root_cause_bar.png", dpi=220)
    plt.close()

def laterality_centroid_consistency(merged: pd.DataFrame, out_tables: Path, out_figs: Path) -> None:
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    need_cols = {"laterality", "centroid_x_norm", "empty"}
    if not need_cols.issubset(set(merged.columns)):
        print("[WARN] Missing columns for laterality-centroid analysis:", need_cols - set(merged.columns))
        return

    df = merged[(merged["empty"] == 0) & (merged["centroid_x_norm"].notna())].copy()
    if len(df) == 0:
        print("[WARN] No non-empty regions with centroid_x_norm available.")
        return

    stats = (
        df.groupby("laterality", dropna=False)["centroid_x_norm"]
          .agg(["count", "mean", "median", "std", "min", "max"])
          .reset_index()
    )
    stats.to_csv(out_tables / "laterality_centroid_stats.csv", index=False)
    plot_box_by_group(df, "centroid_x_norm", "laterality",
                      out_figs / "laterality_centroid_x_box.png",
                      "centroid_x_norm by laterality (non-empty)")

    # AUC test for left vs right (orientation might be flipped)
    from sklearn.metrics import roc_auc_score

    lr = df[df["laterality"].isin(["left", "right"])].copy()
    auc_info = {"n_left": int((lr["laterality"] == "left").sum()),
                "n_right": int((lr["laterality"] == "right").sum())}

    if auc_info["n_left"] > 0 and auc_info["n_right"] > 0:
        y = (lr["laterality"] == "right").astype(int).values
        x = lr["centroid_x_norm"].astype(float).values
        auc_x = float(roc_auc_score(y, x))
        auc_flip = float(roc_auc_score(y, 1.0 - x))
        # pick best orientation
        if auc_x >= auc_flip:
            best_auc = auc_x
            orientation = "x_small=left (right tends to larger x)"
        else:
            best_auc = auc_flip
            orientation = "x_small=right (right tends to smaller x)"
        auc_info.update({"auc_x": auc_x, "auc_flip": auc_flip,
                         "best_auc": best_auc,
                         "inferred_orientation": orientation})
    else:
        auc_info.update({"auc_x": None, "auc_flip": None, "best_auc": None,
                         "inferred_orientation": "insufficient_left_right_samples"})

    save_json(auc_info, out_tables / "laterality_centroid_auc.json")

def question_template_analysis(csv_path: Path, out_tables: Path, out_figs: Path) -> None:
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    df = pd.read_csv(csv_path)
    df["case_id"] = df["Image"].map(parse_case_id)
    df["Mask_ID"] = df["Mask_ID"].astype(int)
    df["region_id"] = df["case_id"].astype(str) + "__" + df["Mask_ID"].astype(str)

    df["q_norm"] = df["Question"].map(normalize_question)
    df["q_template"] = df["q_norm"].map(remove_laterality_tokens)

    # Template counts
    cnt = df["q_template"].value_counts().reset_index()
    cnt.columns = ["question_template", "count"]
    cnt.to_csv(out_tables / "question_template_counts.csv", index=False)

    # Coverage by top-K
    total = len(df)
    cover = {}
    for k in [5, 10, 20, 50, 100]:
        cover[f"top{k}_coverage"] = float(cnt["count"].head(k).sum() / total) if total else 0.0

    summary = {
        "n_rows": int(len(df)),
        "n_cases": int(df["case_id"].nunique()),
        "n_regions": int(df["region_id"].nunique()),
        "n_unique_questions": int(df["Question"].astype(str).nunique()),
        "n_unique_templates": int(df["q_template"].astype(str).nunique()),
        **cover,
    }
    save_json(summary, out_tables / "question_template_summary.json")

    # Plot top templates
    top = cnt.head(20).copy()
    # shorten labels for readability
    top["short"] = top["question_template"].str.slice(0, 55) + top["question_template"].apply(lambda s: "..." if len(s) > 55 else "")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.barh(top["short"][::-1], top["count"][::-1])
    plt.title("Top-20 question templates (laterality normalized)")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(out_figs / "top20_question_templates.png", dpi=220)
    plt.close()

def text_predictability(region_text: pd.DataFrame, out_tables: Path, out_figs: Path) -> None:
    """
    Quantify how well text alone predicts anatomy_group / laterality.
    This turns t-SNE intuition into numbers.
    """
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    # Use label_desc primarily; fallback to questions_joined if empty
    df = region_text.copy()
    df["text"] = df["label_desc"].fillna("").astype(str)
    fallback = df["text"].str.strip() == ""
    if "questions_joined" in df.columns:
        df.loc[fallback, "text"] = df.loc[fallback, "questions_joined"].fillna("").astype(str)

    results = {}

    def run_task(y_col: str, name: str, drop_label: Optional[str] = None):
        sub = df[[y_col, "text"]].dropna().copy()
        if drop_label is not None:
            sub = sub[sub[y_col].astype(str) != drop_label].copy()
        sub[y_col] = sub[y_col].astype(str)
        if sub[y_col].nunique() < 2 or len(sub) < 20:
            results[name] = {"status": "skip", "reason": "too_few_classes_or_samples",
                             "n": int(len(sub)), "n_classes": int(sub[y_col].nunique())}
            return

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_true = sub[y_col].values
        y_pred = cross_val_predict(pipe, sub["text"].values, y_true, cv=skf)

        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average="macro"))
        results[name] = {"status": "ok", "n": int(len(sub)), "n_classes": int(sub[y_col].nunique()),
                         "accuracy": acc, "macro_f1": f1m}

        # confusion matrix plot
        labels = sorted(pd.unique(y_true).tolist())
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 6))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix: {name}")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(out_figs / f"confmat_{name}.png", dpi=220)
        plt.close()

    if "anatomy_group" in df.columns:
        run_task("anatomy_group", "predict_anatomy_group", drop_label=None)

    if "laterality" in df.columns:
        # often "none" dominates; keep it (realistic) and also run a version excluding "none"
        run_task("laterality", "predict_laterality_all", drop_label=None)
        run_task("laterality", "predict_laterality_no_none", drop_label="none")

    save_json(results, out_tables / "text_predictability_metrics.json")

def intensity_and_slice_stats(
    npy_root: Path,
    merged: pd.DataFrame,
    out_tables: Path,
    out_figs: Path,
    max_cases: int = 0,
    sample_per_case: int = 100_000,
    seed: int = 42,
) -> None:
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    if not npy_root.exists():
        print("[WARN] npy_root not found, skip intensity stats:", npy_root)
        return

    rng = np.random.default_rng(seed)
    case_ids = sorted(merged["case_id"].astype(str).unique().tolist())
    if max_cases and max_cases > 0:
        case_ids = case_ids[:max_cases]

    case_rows = []
    all_samples = []

    for cid in tqdm(case_ids, desc="Intensity stats (ct.npy sampling)"):
        ct_path = npy_root / cid / "ct.npy"
        mask_path = npy_root / cid / "mask.npy"
        if not ct_path.exists() or not mask_path.exists():
            continue

        ct = np.load(str(ct_path), mmap_mode="r")
        if ct.ndim == 4 and ct.shape[0] == 1:
            ct = ct[0]
        ct = np.asarray(ct, dtype=np.float32)

        mask = np.load(str(mask_path), mmap_mode="r")
        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = mask[0]
        mask = np.asarray(mask)

        flat = ct.reshape(-1)
        n = flat.shape[0]
        m = min(sample_per_case, n)
        idx = rng.choice(n, size=m, replace=False)
        samp = flat[idx]
        all_samples.append(samp)

        # ROI vs background sampling (union mask > 0)
        mflat = mask.reshape(-1)
        roi_idx = np.where(mflat > 0)[0]
        bg_idx = np.where(mflat == 0)[0]
        roi_mean = np.nan
        bg_mean = np.nan
        if roi_idx.size > 0:
            ridx = rng.choice(roi_idx, size=min(20_000, roi_idx.size), replace=False)
            roi_mean = float(np.mean(flat[ridx]))
        if bg_idx.size > 0:
            bidx = rng.choice(bg_idx, size=min(20_000, bg_idx.size), replace=False)
            bg_mean = float(np.mean(flat[bidx]))

        case_rows.append({
            "case_id": cid,
            "ct_min": float(np.min(ct)),
            "ct_max": float(np.max(ct)),
            "ct_mean": float(np.mean(ct)),
            "ct_std": float(np.std(ct)),
            "ct_p01": float(np.percentile(ct, 1)),
            "ct_p99": float(np.percentile(ct, 99)),
            "roi_union_mean": roi_mean,
            "bg_mean": bg_mean,
            "roi_minus_bg": float(roi_mean - bg_mean) if np.isfinite(roi_mean) and np.isfinite(bg_mean) else np.nan,
            "roi_union_voxels": int((mask > 0).sum()),
        })

    if case_rows:
        case_df = pd.DataFrame(case_rows)
        case_df.to_csv(out_tables / "intensity_case_stats.csv", index=False)

    if all_samples:
        samples = np.concatenate(all_samples, axis=0)
        plot_hist(samples, out_figs / "ct_intensity_samples_hist.png",
                  "CT intensity distribution (sampled from ct.npy)", "intensity", bins=80, log1p=False)

    # slices_covered distribution (from merged table if exists)
    if "slices_covered" in merged.columns:
        sc = merged.loc[merged["empty"] == 0, "slices_covered"].astype(float).to_numpy()
        plot_hist(sc, out_figs / "roi_slices_covered_hist.png",
                  "ROI slices_covered distribution (non-empty)", "slices_covered", bins=30, log1p=False)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser("Deeper EDA for M3D-RefSeg (built on base EDA outputs)")
    ap.add_argument("--csv", required=True, help="Path to M3D_RefSeg.csv")
    ap.add_argument("--eda_out", required=True, help="Base EDA output dir from eda_m3d_refseg_npy.py (contains tables/)")
    ap.add_argument("--npy_root", required=True, help="Path to M3D_RefSeg_npy/")
    ap.add_argument("--orig_root", default="", help="Optional: original root folder containing sXXXX/mask.nii.gz")
    ap.add_argument("--out_dir", default="./eda_deeper_out", help="Output directory for deeper EDA")

    ap.add_argument("--do_all", action="store_true", help="Run all analyses")
    ap.add_argument("--do_empty", action="store_true")
    ap.add_argument("--do_empty_root_cause", action="store_true")
    ap.add_argument("--do_laterality", action="store_true")
    ap.add_argument("--do_templates", action="store_true")
    ap.add_argument("--do_text_predictability", action="store_true")
    ap.add_argument("--do_intensity", action="store_true")

    ap.add_argument("--max_cases", type=int, default=0, help="Limit number of cases for intensity sampling (0=all)")
    ap.add_argument("--sample_per_case", type=int, default=100_000, help="Intensity samples per case")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.do_all:
        args.do_empty = True
        args.do_laterality = True
        args.do_templates = True
        args.do_text_predictability = True
        args.do_intensity = True
        # root cause only if orig_root provided
        args.do_empty_root_cause = bool(args.orig_root)

    eda_out = Path(args.eda_out)
    out_dir = Path(args.out_dir)
    out_tables = out_dir / "tables"
    out_figs = out_dir / "figs"
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    # load base tables
    region_text, region_img, empty_df = load_base_tables(eda_out)
    merged = merge_region_tables(region_text, region_img)
    merged.to_csv(out_tables / "merged_regions.csv", index=False)

    # empty list
    if empty_df is None:
        empty_df = merged[merged.get("empty", 0) == 1][["case_id", "Mask_ID", "region_id"]].drop_duplicates()

    # run analyses
    if args.do_empty:
        analyze_empty_and_buckets(merged, out_tables, out_figs)

    if args.do_empty_root_cause:
        if not args.orig_root:
            print("[WARN] --do_empty_root_cause requires --orig_root. Skipping.")
        else:
            orig_root = Path(args.orig_root)
            npy_root = Path(args.npy_root)
            diagnose_empty_root_cause(Path(args.csv), empty_df, orig_root, npy_root, out_tables, out_figs)

    if args.do_laterality:
        laterality_centroid_consistency(merged, out_tables, out_figs)

    if args.do_templates:
        question_template_analysis(Path(args.csv), out_tables, out_figs)

    if args.do_text_predictability:
        text_predictability(region_text, out_tables, out_figs)

    if args.do_intensity:
        intensity_and_slice_stats(Path(args.npy_root), merged, out_tables, out_figs,
                                  max_cases=args.max_cases, sample_per_case=args.sample_per_case, seed=args.seed)

    print("\nDone. Deeper EDA outputs saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
