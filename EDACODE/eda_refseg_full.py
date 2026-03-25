
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-contained EDA for M3D-RefSeg (CSV + prepared NPY folder).

What you get (outputs saved to out_dir/):
- tables/regions.csv: region-level merged table (text + ROI geometry + intensity + QA stats)
- tables/question_template_counts.csv, question_template_summary.json
- tables/empty_* summaries + plots
- tables/laterality_centroid_* summaries + plots
- tables/text_predictability_metrics.json + confusion matrices
- figs/tsne_* plots for text and ROI features (optional)

Designed to mirror common dataset "bias/leakage/quality" checks discussed in meetings,
adapted to RefSeg: template duplication, region-level leakage risk, empty masks, long-tail ROI sizes,
laterality consistency, etc.

Assumptions about prepared folder (from your m3d_refseg_data_prepare.py):
  <npy_root>/<case_id>/ct.npy
  <npy_root>/<case_id>/mask.npy
  <npy_root>/<case_id>/text.json

Optional original root (for empty root-cause):
  <orig_root>/<case_id>/mask.nii.gz
  <orig_root>/<case_id>/text.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------
# Basic utils
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def parse_case_id(image_path: str) -> str:
    parts = str(image_path).replace("\\", "/").split("/")
    return parts[0] if parts else str(image_path)

def squeeze_to_3d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    return arr

def safe_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))

def plot_hist(values: np.ndarray, out_path: Path, title: str, xlabel: str, bins: int = 60, log1p: bool = False):
    import matplotlib.pyplot as plt
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    if log1p:
        v = np.log1p(np.clip(v, a_min=0, a_max=None))
    plt.figure(figsize=(7, 4.6))
    plt.hist(v, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel + (" (log1p)" if log1p else ""))
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_bar(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str, rotate: int = 45):
    import matplotlib.pyplot as plt
    if df.empty:
        return
    plt.figure(figsize=(10, 4.8))
    plt.bar(df[x].astype(str).tolist(), df[y].astype(float).tolist())
    plt.xticks(rotation=rotate, ha="right")
    plt.title(title)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_box_by_group(df: pd.DataFrame, value_col: str, group_col: str, out_path: Path, title: str):
    import matplotlib.pyplot as plt
    groups = []
    labels = []
    for g, sub in df.groupby(group_col, dropna=False):
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


# ----------------------------
# Text heuristics
# ----------------------------

_LAT_PATTERNS = [
    (re.compile(r"\bbilateral\b", re.I), "bilateral"),
    (re.compile(r"\bleft\b", re.I), "left"),
    (re.compile(r"\bright\b", re.I), "right"),
]

def infer_laterality(text: str) -> str:
    t = str(text)
    for pat, lab in _LAT_PATTERNS:
        if pat.search(t):
            return lab
    return "none"


# anatomy_group keyword mapping (heuristic)
_ANATOMY_KEYWORDS: Dict[str, List[str]] = {
    "cardio_vascular": [
        "heart", "cardiac", "coronary", "artery", "aorta", "vascular", "vein", "atrium", "ventricle", "myocard",
    ],
    "pulmonary": [
        "lung", "pulmonary", "pleura", "bronch", "airway", "alveol", "pneumo",
    ],
    "renal_urinary": [
        "kidney", "renal", "ureter", "urethra", "bladder", "pelvis", "calyx", "calyceal", "nephro",
    ],
    "hepato_biliary": [
        "liver", "hepatic", "gallbladder", "biliary", "bile", "pancreas", "pancreatic", "cholecyst",
    ],
    "gastro_intestinal": [
        "stomach", "gastric", "intestin", "bowel", "colon", "rectum", "duoden", "ileum", "jejun", "esophag", "appendix",
    ],
    "musculoskeletal": [
        "bone", "fracture", "vertebra", "spine", "spinal", "femur", "tibia", "fibula", "humerus", "radius", "ulna",
        "joint", "muscle", "ligament", "cartilage", "rib", "sternum", "clavicle", "scapula", "pelvic bone",
    ],
    "neuro_headneck": [
        "brain", "cerebral", "cerebell", "head", "neck", "skull", "sinus", "orbit", "pituitary", "thyroid",
    ],
    "reproductive": [
        "uterus", "ovary", "ovarian", "cervix", "testis", "testicle", "prostate", "seminal", "adnexa", "fallopian",
    ],
}

def infer_anatomy_group(text: str) -> str:
    t = str(text).lower()
    for group, kws in _ANATOMY_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return group
    return "other"


# ----------------------------
# QA template / duplication
# ----------------------------

def normalize_question(q: str) -> str:
    q = str(q).lower().strip()
    q = re.sub(r"\d+", "<num>", q)
    q = re.sub(r"[^a-z0-9<>\s]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def remove_laterality_tokens(q_norm: str) -> str:
    q = re.sub(r"\b(left|right|bilateral)\b", "<lat>", q_norm)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def token_set(s: str) -> set:
    return set(str(s).split())

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ----------------------------
# ROI features
# ----------------------------

@dataclass
class RoiGeom:
    empty: int
    voxels: int
    zmin: Optional[int] = None
    zmax: Optional[int] = None
    ymin: Optional[int] = None
    ymax: Optional[int] = None
    xmin: Optional[int] = None
    xmax: Optional[int] = None
    dz: Optional[int] = None
    dy: Optional[int] = None
    dx: Optional[int] = None
    centroid_z: Optional[float] = None
    centroid_y: Optional[float] = None
    centroid_x: Optional[float] = None
    centroid_z_norm: Optional[float] = None
    centroid_y_norm: Optional[float] = None
    centroid_x_norm: Optional[float] = None
    slices_covered: Optional[int] = None


def compute_roi_geom(mask3d: np.ndarray, mask_id: int) -> RoiGeom:
    m = mask3d
    coords = np.argwhere(m == mask_id)
    vox = int(coords.shape[0])
    if vox == 0:
        return RoiGeom(empty=1, voxels=0, slices_covered=0)

    z = coords[:, 0]
    y = coords[:, 1]
    x = coords[:, 2]
    zmin, zmax = int(z.min()), int(z.max())
    ymin, ymax = int(y.min()), int(y.max())
    xmin, xmax = int(x.min()), int(x.max())
    dz = zmax - zmin + 1
    dy = ymax - ymin + 1
    dx = xmax - xmin + 1
    cz = float(z.mean())
    cy = float(y.mean())
    cx = float(x.mean())

    D, H, W = m.shape
    # normalize to [0,1]
    z_norm = cz / (D - 1) if D > 1 else 0.0
    y_norm = cy / (H - 1) if H > 1 else 0.0
    x_norm = cx / (W - 1) if W > 1 else 0.0

    slices_covered = int(np.unique(z).size)

    return RoiGeom(
        empty=0, voxels=vox,
        zmin=zmin, zmax=zmax, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax,
        dz=dz, dy=dy, dx=dx,
        centroid_z=cz, centroid_y=cy, centroid_x=cx,
        centroid_z_norm=z_norm, centroid_y_norm=y_norm, centroid_x_norm=x_norm,
        slices_covered=slices_covered,
    )


def voxel_bucket(v: float) -> str:
    if not np.isfinite(v):
        return "NA"
    v = float(v)
    edges = [0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    labels = ["0", "1-9", "10-99", "100-999", "1k-9k", "10k-99k", "100k-999k", ">=1M"]
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return labels[i]
    return labels[-1]


# ----------------------------
# TSNE plotting (fixed legend colors)
# ----------------------------

def run_tsne_and_plot(
    X: np.ndarray,
    meta: pd.DataFrame,
    out_prefix: str,
    out_figs: Path,
    color_cols: List[str],
    perplexity: int = 30,
    seed: int = 42,
) -> None:
    """
    t-SNE + plots. For categorical columns, plot per category so matplotlib uses default colors
    and legend matches the colors.
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    ensure_dir(out_figs)

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n_samples, n_features = X.shape
    if n_samples < 3:
        print(f"[WARN] Too few samples for t-SNE: {n_samples}. Skip {out_prefix}.")
        return

    # safe perplexity
    max_perp = max(2, min(perplexity, (n_samples - 1) // 3))
    if max_perp != perplexity:
        print(f"[INFO] Adjust t-SNE perplexity {perplexity} -> {max_perp} due to n_samples={n_samples}")

    # reduce dims if huge
    if n_features > 100:
        # PCA to 50 dims for speed
        pca_dim = min(50, n_samples - 1, n_features)
        X_red = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    else:
        X_red = X

    # init: use 'pca' only if features >= 2
    init = "pca" if X_red.shape[1] >= 2 else "random"

    tsne = TSNE(
        n_components=2,
        perplexity=max_perp,
        random_state=seed,
        init=init,
        learning_rate="auto",
        n_iter=1000,
    )
    Y = tsne.fit_transform(X_red)

    coords = meta.copy().reset_index(drop=True)
    coords["tsne_x"] = Y[:, 0]
    coords["tsne_y"] = Y[:, 1]
    coords.to_csv(out_figs / f"{out_prefix}_tsne_coords.csv", index=False)

    import matplotlib.pyplot as plt
    for col in color_cols:
        if col not in coords.columns:
            continue
        c = coords[col]

        plt.figure(figsize=(7, 6))
        if pd.api.types.is_numeric_dtype(c):
            uniq_vals = sorted(pd.Series(c).dropna().unique().tolist())
            if len(uniq_vals) <= 12:
                # treat as categorical
                cats = c.astype(str).fillna("NA")
                uniq = sorted(cats.unique())
                for u in uniq:
                    idx = (cats == u).values
                    plt.scatter(coords.loc[idx, "tsne_x"], coords.loc[idx, "tsne_y"], s=10, label=u)
                plt.legend(title=col, bbox_to_anchor=(1.04, 1), loc="upper left")
            else:
                sc = plt.scatter(coords["tsne_x"], coords["tsne_y"], c=c, s=10)
                plt.colorbar(sc, label=col)
        else:
            cats = c.astype(str).fillna("NA")
            uniq = sorted(cats.unique())
            for u in uniq:
                idx = (cats == u).values
                plt.scatter(coords.loc[idx, "tsne_x"], coords.loc[idx, "tsne_y"], s=10, label=u)
            plt.legend(title=col, bbox_to_anchor=(1.04, 1), loc="upper left")

        plt.title(f"t-SNE colored by {col}")
        plt.tight_layout()
        plt.savefig(out_figs / f"{out_prefix}_tsne_{col}.png", dpi=220)
        plt.close()


# ----------------------------
# Predictability (text -> labels)
# ----------------------------

def text_predictability(df_regions: pd.DataFrame, out_tables: Path, out_figs: Path, seed: int = 42) -> None:
    """
    Quantify how much information text contains for anatomy_group / laterality.
    """
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    df = df_regions.copy()
    # use label_desc; fallback to questions_joined
    df["text_for_clf"] = df["label_desc"].fillna("").astype(str)
    fallback = df["text_for_clf"].str.strip() == ""
    df.loc[fallback, "text_for_clf"] = df.loc[fallback, "questions_joined"].fillna("").astype(str)

    results = {}

    def run_task(y_col: str, name: str, drop_label: Optional[str] = None):
        sub = df[[y_col, "text_for_clf"]].dropna().copy()
        if drop_label is not None:
            sub = sub[sub[y_col].astype(str) != drop_label].copy()

        sub[y_col] = sub[y_col].astype(str)
        if sub[y_col].nunique() < 2 or len(sub) < 20:
            results[name] = {
                "status": "skip",
                "reason": "too_few_classes_or_samples",
                "n": int(len(sub)),
                "n_classes": int(sub[y_col].nunique()),
            }
            return

        # pick n_splits <= min class count
        min_count = int(sub[y_col].value_counts().min())
        n_splits = min(5, max(2, min_count))
        if n_splits < 2:
            results[name] = {"status": "skip", "reason": "min_class_count<2"}
            return

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ])

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y_true = sub[y_col].values
        y_pred = cross_val_predict(pipe, sub["text_for_clf"].values, y_true, cv=skf)

        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average="macro"))
        results[name] = {
            "status": "ok",
            "n": int(len(sub)),
            "n_classes": int(sub[y_col].nunique()),
            "n_splits": int(n_splits),
            "accuracy": acc,
            "macro_f1": f1m,
        }

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
        run_task("anatomy_group", "predict_anatomy_group")

    if "laterality" in df.columns:
        run_task("laterality", "predict_laterality_all")
        run_task("laterality", "predict_laterality_no_none", drop_label="none")

    save_json(results, out_tables / "text_predictability_metrics.json")


# ----------------------------
# Main pipeline
# ----------------------------

def load_case_text_json(npy_root: Path, orig_root: Optional[Path], case_id: str) -> Dict[str, str]:
    # Try prepared folder first
    p1 = npy_root / case_id / "text.json"
    if p1.exists():
        try:
            return json.loads(p1.read_text(encoding="utf-8"))
        except Exception:
            pass
    if orig_root is not None:
        p2 = orig_root / case_id / "text.json"
        if p2.exists():
            try:
                return json.loads(p2.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}

def build_region_table(
    csv_path: Path,
    npy_root: Path,
    orig_root: Optional[Path],
    out_tables: Path,
    out_figs: Path,
    compute_intensity: bool = True,
    intensity_max_samples: int = 20000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build region-level table:
    - qa_rows, unique_questions, within-region question similarity
    - label_desc from text.json
    - inferred laterality / anatomy_group from label_desc
    - ROI geometry from mask.npy
    - ROI intensity stats from ct.npy (optional)
    """
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    rng = np.random.default_rng(seed)

    df = pd.read_csv(csv_path)
    # columns: Image, Mask, Mask_ID, Question_Type, Question, Answer
    df["case_id"] = df["Image"].map(parse_case_id)
    df["Mask_ID"] = df["Mask_ID"].astype(int)
    df["region_id"] = df["case_id"].astype(str) + "__" + df["Mask_ID"].astype(str)

    # region-level QA aggregates
    agg_rows = []
    for rid, sub in df.groupby("region_id"):
        q_norms = sub["Question"].map(normalize_question).tolist()
        q_sets = [token_set(q) for q in q_norms]
        # pairwise jaccard (small K ~ 4-5)
        sims = []
        for i in range(len(q_sets)):
            for j in range(i + 1, len(q_sets)):
                sims.append(jaccard(q_sets[i], q_sets[j]))
        mean_sim = float(np.mean(sims)) if sims else float("nan")

        agg_rows.append({
            "region_id": rid,
            "case_id": str(sub["case_id"].iloc[0]),
            "Mask_ID": int(sub["Mask_ID"].iloc[0]),
            "qa_rows": int(len(sub)),
            "unique_questions": int(len(set(q_norms))),
            "question_mean_jaccard": mean_sim,
            "questions_joined": " || ".join(sub["Question"].astype(str).tolist()),
            "answers_joined": " || ".join(sub["Answer"].astype(str).tolist()),
        })
    region_df = pd.DataFrame(agg_rows)

    # load per-case caches
    mask_cache: Dict[str, np.ndarray] = {}
    ct_cache: Dict[str, np.ndarray] = {}
    text_cache: Dict[str, Dict[str, str]] = {}

    # compute label_desc, laterality, anatomy_group
    label_descs = []
    lats = []
    groups = []
    for case_id, mask_id in tqdm(region_df[["case_id", "Mask_ID"]].itertuples(index=False),
                                 total=len(region_df),
                                 desc="Load label_desc / infer labels"):
        case_id = str(case_id)
        mask_id = int(mask_id)
        if case_id not in text_cache:
            text_cache[case_id] = load_case_text_json(npy_root, orig_root, case_id)
        tmap = text_cache[case_id]
        desc = tmap.get(str(mask_id), "")
        label_descs.append(desc)
        lats.append(infer_laterality(desc))
        groups.append(infer_anatomy_group(desc))
    region_df["label_desc"] = label_descs
    region_df["laterality"] = lats
    region_df["anatomy_group"] = groups

    # ROI geometry + intensity
    geom_rows = []
    intensity_rows = []

    unique_cases = sorted(region_df["case_id"].unique().tolist())

    for case_id in tqdm(unique_cases, desc="Compute ROI geometry/intensity per case"):
        case_id = str(case_id)
        mask_path = npy_root / case_id / "mask.npy"
        ct_path = npy_root / case_id / "ct.npy"

        if not mask_path.exists():
            print(f"[WARN] Missing mask.npy for case {case_id}: {mask_path}")
            continue

        # load mask
        if case_id not in mask_cache:
            mask = np.load(str(mask_path), mmap_mode="r")
            mask = squeeze_to_3d(mask)
            mask = np.rint(mask).astype(np.int32, copy=False)
            mask_cache[case_id] = mask
        else:
            mask = mask_cache[case_id]

        # load ct only if needed
        ct = None
        if compute_intensity and ct_path.exists():
            if case_id not in ct_cache:
                ct_arr = np.load(str(ct_path), mmap_mode="r")
                ct_arr = squeeze_to_3d(ct_arr)
                ct_cache[case_id] = np.asarray(ct_arr, dtype=np.float32)
            ct = ct_cache[case_id]

        sub_regions = region_df[region_df["case_id"] == case_id][["region_id", "Mask_ID"]]
        for rid, mid in sub_regions.itertuples(index=False):
            geom = compute_roi_geom(mask, int(mid))
            geom_rows.append({
                "region_id": rid,
                "empty": geom.empty,
                "voxels": geom.voxels,
                "zmin": geom.zmin, "zmax": geom.zmax,
                "ymin": geom.ymin, "ymax": geom.ymax,
                "xmin": geom.xmin, "xmax": geom.xmax,
                "dz": geom.dz, "dy": geom.dy, "dx": geom.dx,
                "centroid_z": geom.centroid_z,
                "centroid_y": geom.centroid_y,
                "centroid_x": geom.centroid_x,
                "centroid_z_norm": geom.centroid_z_norm,
                "centroid_y_norm": geom.centroid_y_norm,
                "centroid_x_norm": geom.centroid_x_norm,
                "slices_covered": geom.slices_covered,
            })

            if compute_intensity and ct is not None and geom.empty == 0:
                # sample ROI voxels if huge
                coords = np.argwhere(mask == int(mid))
                if coords.shape[0] > intensity_max_samples:
                    idx = rng.choice(coords.shape[0], size=intensity_max_samples, replace=False)
                    coords = coords[idx]
                vals = ct[coords[:, 0], coords[:, 1], coords[:, 2]]
                vals = np.asarray(vals, dtype=float)
                intensity_rows.append({
                    "region_id": rid,
                    "roi_mean": float(np.mean(vals)) if vals.size else float("nan"),
                    "roi_std": float(np.std(vals)) if vals.size else float("nan"),
                    "roi_p05": safe_percentile(vals, 5),
                    "roi_p50": safe_percentile(vals, 50),
                    "roi_p95": safe_percentile(vals, 95),
                })
            else:
                intensity_rows.append({
                    "region_id": rid,
                    "roi_mean": float("nan"),
                    "roi_std": float("nan"),
                    "roi_p05": float("nan"),
                    "roi_p50": float("nan"),
                    "roi_p95": float("nan"),
                })

    geom_df = pd.DataFrame(geom_rows)
    inten_df = pd.DataFrame(intensity_rows)

    region_df = region_df.merge(geom_df, on="region_id", how="left")
    region_df = region_df.merge(inten_df, on="region_id", how="left")

    # buckets
    region_df["voxels_bucket"] = region_df["voxels"].astype(float).map(voxel_bucket)

    region_df.to_csv(out_tables / "regions.csv", index=False)
    return region_df


def analyze_empty(region_df: pd.DataFrame, out_tables: Path, out_figs: Path) -> None:
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    if "empty" not in region_df.columns:
        return

    overall = {
        "n_regions": int(len(region_df)),
        "n_empty": int((region_df["empty"] == 1).sum()),
        "empty_rate": float((region_df["empty"] == 1).mean()),
        "voxels_p50": safe_percentile(region_df["voxels"].values, 50),
        "voxels_p90": safe_percentile(region_df["voxels"].values, 90),
        "voxels_p95": safe_percentile(region_df["voxels"].values, 95),
    }
    save_json(overall, out_tables / "empty_overall_summary.json")

    # empty rate by anatomy_group
    by_anat = (
        region_df.groupby("anatomy_group", dropna=False)
        .agg(n=("region_id", "size"), n_empty=("empty", lambda x: int((x == 1).sum())))
        .reset_index()
    )
    by_anat["empty_rate"] = by_anat["n_empty"] / by_anat["n"].replace(0, np.nan)
    by_anat = by_anat.sort_values("empty_rate", ascending=False)
    by_anat.to_csv(out_tables / "empty_rate_by_anatomy.csv", index=False)
    plot_bar(by_anat, "anatomy_group", "empty_rate", out_figs / "empty_rate_by_anatomy.png",
             "Empty rate by anatomy_group")

    # empty rate by voxel bucket
    by_bucket = (
        region_df.groupby("voxels_bucket", dropna=False)
        .agg(n=("region_id", "size"), n_empty=("empty", lambda x: int((x == 1).sum())))
        .reset_index()
    )
    by_bucket["empty_rate"] = by_bucket["n_empty"] / by_bucket["n"].replace(0, np.nan)
    bucket_order = ["0", "1-9", "10-99", "100-999", "1k-9k", "10k-99k", "100k-999k", ">=1M", "NA"]
    by_bucket["voxels_bucket"] = pd.Categorical(by_bucket["voxels_bucket"], categories=bucket_order, ordered=True)
    by_bucket = by_bucket.sort_values("voxels_bucket")
    by_bucket.to_csv(out_tables / "empty_rate_by_voxel_bucket.csv", index=False)
    plot_bar(by_bucket, "voxels_bucket", "empty_rate", out_figs / "empty_rate_by_voxel_bucket.png",
             "Empty rate by ROI voxel bucket", rotate=0)

    # voxel distribution
    plot_hist(region_df["voxels"].values, out_figs / "roi_voxels_hist.png",
              "ROI voxel count distribution", "voxels", bins=60, log1p=False)
    plot_hist(region_df["voxels"].values, out_figs / "roi_voxels_hist_log1p.png",
              "ROI voxel count distribution", "voxels", bins=60, log1p=True)


def empty_root_cause(
    csv_path: Path,
    region_df: pd.DataFrame,
    orig_root: Path,
    npy_root: Path,
    out_tables: Path,
    out_figs: Path,
) -> None:
    """
    Check whether each empty region exists in the original mask.nii.gz
    -> erased_by_prepare vs missing_in_original_or_id_mismatch
    """
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    try:
        import nibabel as nib
    except Exception as e:
        print("[WARN] nibabel not available; cannot run empty_root_cause:", e)
        return

    df = pd.read_csv(csv_path)
    df["case_id"] = df["Image"].map(parse_case_id)
    df["Mask_ID"] = df["Mask_ID"].astype(int)
    df["region_id"] = df["case_id"].astype(str) + "__" + df["Mask_ID"].astype(str)
    map_df = df[["case_id", "Mask_ID", "region_id", "Mask"]].drop_duplicates()

    empties = region_df[region_df["empty"] == 1][["case_id", "Mask_ID", "region_id"]].drop_duplicates()
    if empties.empty:
        print("[INFO] No empty regions found. Skip root-cause.")
        return

    targets = empties.merge(map_df, on=["case_id", "Mask_ID", "region_id"], how="left")

    cache_orig: Dict[str, np.ndarray] = {}

    rows = []
    for case_id, mask_id, region_id, mask_rel in tqdm(
        targets[["case_id", "Mask_ID", "region_id", "Mask"]].itertuples(index=False),
        total=len(targets),
        desc="Empty root-cause",
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
                "cause": "unknown",
            })
            continue

        if case_id not in cache_orig:
            try:
                arr = nib.load(str(orig_mask_path)).get_fdata()
                arr = np.rint(arr).astype(np.int32)
                cache_orig[case_id] = arr
            except Exception as e:
                rows.append({
                    "region_id": region_id, "case_id": case_id, "Mask_ID": mask_id,
                    "status": f"orig_load_error:{e}",
                    "orig_mask_path": str(orig_mask_path),
                    "npy_mask_path": str(npy_mask_path),
                    "orig_voxels": np.nan, "npy_voxels": 0,
                    "cause": "unknown",
                })
                continue

        orig_mask = cache_orig[case_id]
        orig_vox = int((orig_mask == mask_id).sum())

        cause = "erased_by_prepare" if orig_vox > 0 else "missing_in_original_or_id_mismatch"
        rows.append({
            "region_id": region_id, "case_id": case_id, "Mask_ID": mask_id,
            "status": "ok",
            "orig_mask_path": str(orig_mask_path),
            "npy_mask_path": str(npy_mask_path),
            "orig_voxels": orig_vox,
            "npy_voxels": 0,
            "cause": cause,
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_tables / "empty_root_cause_report.csv", index=False)
    summary = out["cause"].value_counts(dropna=False).to_dict()
    save_json(summary, out_tables / "empty_root_cause_summary.json")

    # plot
    import matplotlib.pyplot as plt
    labels = list(summary.keys())
    counts = [summary[k] for k in labels]
    plt.figure(figsize=(7, 4.6))
    plt.bar([str(x) for x in labels], counts)
    plt.xticks(rotation=30, ha="right")
    plt.title("Empty root cause counts")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_figs / "empty_root_cause_bar.png", dpi=220)
    plt.close()


def laterality_centroid(region_df: pd.DataFrame, out_tables: Path, out_figs: Path) -> None:
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    need = {"laterality", "centroid_x_norm", "empty"}
    if not need.issubset(set(region_df.columns)):
        return

    df = region_df[(region_df["empty"] == 0) & (region_df["centroid_x_norm"].notna())].copy()
    if df.empty:
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

    # AUC for left vs right (orientation could be flipped)
    from sklearn.metrics import roc_auc_score
    lr = df[df["laterality"].isin(["left", "right"])].copy()
    auc_info = {
        "n_left": int((lr["laterality"] == "left").sum()),
        "n_right": int((lr["laterality"] == "right").sum()),
    }
    if auc_info["n_left"] > 0 and auc_info["n_right"] > 0:
        y = (lr["laterality"] == "right").astype(int).values
        x = lr["centroid_x_norm"].astype(float).values
        auc_x = float(roc_auc_score(y, x))
        auc_flip = float(roc_auc_score(y, 1.0 - x))
        if auc_x >= auc_flip:
            best_auc = auc_x
            orient = "x_small=left (right tends to larger x)"
        else:
            best_auc = auc_flip
            orient = "x_small=right (right tends to smaller x)"
        auc_info.update({
            "auc_x": auc_x,
            "auc_flip": auc_flip,
            "best_auc": best_auc,
            "inferred_orientation": orient,
        })
    else:
        auc_info.update({
            "auc_x": None, "auc_flip": None, "best_auc": None,
            "inferred_orientation": "insufficient_left_right_samples",
        })
    save_json(auc_info, out_tables / "laterality_centroid_auc.json")


def question_templates(csv_path: Path, out_tables: Path, out_figs: Path) -> None:
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    df = pd.read_csv(csv_path)
    df["q_norm"] = df["Question"].map(normalize_question)
    df["q_template"] = df["q_norm"].map(remove_laterality_tokens)

    cnt = df["q_template"].value_counts().reset_index()
    cnt.columns = ["question_template", "count"]
    cnt.to_csv(out_tables / "question_template_counts.csv", index=False)

    total = len(df)
    cover = {}
    for k in [5, 10, 20, 50, 100]:
        cover[f"top{k}_coverage"] = float(cnt["count"].head(k).sum() / total) if total else 0.0

    summary = {
        "n_rows": int(len(df)),
        "n_unique_questions": int(df["Question"].astype(str).nunique()),
        "n_unique_templates": int(df["q_template"].astype(str).nunique()),
        **cover,
    }
    save_json(summary, out_tables / "question_template_summary.json")

    top = cnt.head(20).copy()
    top["short"] = top["question_template"].str.slice(0, 60) + top["question_template"].apply(lambda s: "..." if len(s) > 60 else "")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.barh(top["short"][::-1], top["count"][::-1])
    plt.title("Top-20 question templates (laterality normalized)")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(out_figs / "top20_question_templates.png", dpi=220)
    plt.close()


def intensity_case_stats(npy_root: Path, out_tables: Path, out_figs: Path,
                         max_cases: int = 0, sample_per_case: int = 100000, seed: int = 42) -> None:
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    rng = np.random.default_rng(seed)
    case_dirs = [p for p in npy_root.iterdir() if p.is_dir()]
    case_ids = sorted([p.name for p in case_dirs])
    if max_cases and max_cases > 0:
        case_ids = case_ids[:max_cases]

    rows = []
    all_samples = []

    for cid in tqdm(case_ids, desc="Intensity stats (case-level sampling)"):
        ct_path = npy_root / cid / "ct.npy"
        mask_path = npy_root / cid / "mask.npy"
        if not ct_path.exists() or not mask_path.exists():
            continue

        ct = squeeze_to_3d(np.load(str(ct_path), mmap_mode="r"))
        ct = np.asarray(ct, dtype=np.float32)
        mask = squeeze_to_3d(np.load(str(mask_path), mmap_mode="r"))
        mask = np.rint(mask).astype(np.int32, copy=False)

        flat = ct.reshape(-1)
        n = flat.shape[0]
        m = min(sample_per_case, n)
        idx = rng.choice(n, size=m, replace=False)
        samp = flat[idx]
        all_samples.append(samp)

        # union ROI vs background
        mflat = mask.reshape(-1)
        roi_idx = np.where(mflat > 0)[0]
        bg_idx = np.where(mflat == 0)[0]
        roi_mean = float("nan")
        bg_mean = float("nan")
        if roi_idx.size > 0:
            ridx = rng.choice(roi_idx, size=min(20000, roi_idx.size), replace=False)
            roi_mean = float(np.mean(flat[ridx]))
        if bg_idx.size > 0:
            bidx = rng.choice(bg_idx, size=min(20000, bg_idx.size), replace=False)
            bg_mean = float(np.mean(flat[bidx]))

        rows.append({
            "case_id": cid,
            "ct_min": float(np.min(ct)),
            "ct_max": float(np.max(ct)),
            "ct_mean": float(np.mean(ct)),
            "ct_std": float(np.std(ct)),
            "ct_p01": float(np.percentile(ct, 1)),
            "ct_p99": float(np.percentile(ct, 99)),
            "roi_union_mean": roi_mean,
            "bg_mean": bg_mean,
            "roi_minus_bg": float(roi_mean - bg_mean) if np.isfinite(roi_mean) and np.isfinite(bg_mean) else float("nan"),
            "roi_union_voxels": int((mask > 0).sum()),
        })

    if rows:
        case_df = pd.DataFrame(rows)
        case_df.to_csv(out_tables / "intensity_case_stats.csv", index=False)

    if all_samples:
        samples = np.concatenate(all_samples, axis=0)
        plot_hist(samples, out_figs / "ct_intensity_samples_hist.png",
                  "CT intensity distribution (sampled from ct.npy)", "intensity", bins=80, log1p=False)


def main():
    ap = argparse.ArgumentParser("Self-contained EDA for M3D-RefSeg (CSV + NPY)")
    ap.add_argument("--csv", required=True, help="Path to M3D_RefSeg.csv")
    ap.add_argument("--npy_root", required=True, help="Path to M3D_RefSeg_npy/")
    ap.add_argument("--out_dir", default="./eda_full_out", help="Output directory")
    ap.add_argument("--orig_root", default="", help="Optional: original root with sXXXX/mask.nii.gz for root-cause")

    ap.add_argument("--no_intensity", action="store_true", help="Skip ROI intensity stats from ct.npy")
    ap.add_argument("--do_tsne_text", action="store_true")
    ap.add_argument("--do_tsne_roi", action="store_true")
    ap.add_argument("--tsne_perplexity", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--do_empty_root_cause", action="store_true", help="Requires --orig_root")
    ap.add_argument("--do_intensity_case_stats", action="store_true", help="Case-level intensity hist")
    ap.add_argument("--max_cases", type=int, default=0, help="Limit cases for case-level intensity stats (0=all)")
    ap.add_argument("--sample_per_case", type=int, default=100000)

    args = ap.parse_args()

    csv_path = Path(args.csv)
    npy_root = Path(args.npy_root)
    out_dir = Path(args.out_dir)
    out_tables = out_dir / "tables"
    out_figs = out_dir / "figs"
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    orig_root = Path(args.orig_root) if args.orig_root else None

    # 1) build region table
    region_df = build_region_table(
        csv_path=csv_path,
        npy_root=npy_root,
        orig_root=orig_root,
        out_tables=out_tables,
        out_figs=out_figs,
        compute_intensity=(not args.no_intensity),
        seed=args.seed,
    )

    # 2) analyses
    analyze_empty(region_df, out_tables, out_figs)
    laterality_centroid(region_df, out_tables, out_figs)
    question_templates(csv_path, out_tables, out_figs)
    text_predictability(region_df, out_tables, out_figs, seed=args.seed)

    if args.do_empty_root_cause:
        if orig_root is None:
            raise ValueError("--do_empty_root_cause requires --orig_root")
        empty_root_cause(csv_path, region_df, orig_root, npy_root, out_tables, out_figs)

    if args.do_intensity_case_stats:
        intensity_case_stats(npy_root, out_tables, out_figs, max_cases=args.max_cases,
                             sample_per_case=args.sample_per_case, seed=args.seed)

    # 3) optional t-SNE
    if args.do_tsne_text:
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = region_df["label_desc"].fillna("").astype(str).tolist()
        # if many empty, fallback to questions
        for i, t in enumerate(texts):
            if not t.strip():
                texts[i] = str(region_df["questions_joined"].iloc[i])
        vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(texts).toarray()
        meta = region_df[["region_id", "case_id", "Mask_ID", "qa_rows", "anatomy_group", "laterality"]].copy()
        run_tsne_and_plot(
            X, meta,
            out_prefix="text_label_desc_tfidf",
            out_figs=out_figs,
            color_cols=["anatomy_group", "laterality", "qa_rows"],
            perplexity=args.tsne_perplexity,
            seed=args.seed,
        )

    if args.do_tsne_roi:
        # numeric features
        feats = region_df[[
            "empty", "voxels", "dz", "dy", "dx",
            "centroid_x_norm", "centroid_y_norm", "centroid_z_norm",
            "slices_covered",
        ]].copy()
        feats = feats.fillna(0.0)
        feats["voxels_log1p"] = np.log1p(np.clip(feats["voxels"].astype(float).values, a_min=0, a_max=None))
        use_cols = ["empty", "voxels_log1p", "dz", "dy", "dx", "centroid_x_norm", "centroid_y_norm", "centroid_z_norm", "slices_covered"]
        X = feats[use_cols].to_numpy(dtype=float)

        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

        meta = region_df[["region_id", "case_id", "Mask_ID", "empty", "voxels", "anatomy_group", "laterality"]].copy()
        run_tsne_and_plot(
            X, meta,
            out_prefix="roi_numeric_features",
            out_figs=out_figs,
            color_cols=["empty", "voxels", "anatomy_group", "laterality"],
            perplexity=args.tsne_perplexity,
            seed=args.seed,
        )

    print("\nDone. Outputs saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
