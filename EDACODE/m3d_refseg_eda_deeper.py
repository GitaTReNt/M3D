
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deeper EDA for M3D-RefSeg (works with either NIfTI or prepared npy folder, but optimized for npy).

What this script adds beyond basic text EDA:
1) Template mining (top question prefixes) + visualization
2) Prompt diversity per region (pairwise cosine similarity of paraphrase questions)
3) Build synthetic multi-finding "report" dataset from label descriptions
4) Optional image/ROI feature extraction from ct.npy/mask.npy (voxels, bbox, centroid, components, intensity)
5) Optional t-SNE on text embeddings (question or label_desc), plus basic coloring

Inputs you need:
- --csv : M3D_RefSeg.csv (the QA table)
- --npy_root : folder like M3D_RefSeg_npy/ containing sXXXX/ct.npy, mask.npy, text.json
              (if you don't provide npy_root, the script will still run some text-only analyses)

Outputs:
- <out_dir>/tables/*.csv
- <out_dir>/figs/*.png
- <out_dir>/synthetic/*.csv  (synthetic report datasets)

Install deps:
  pip install pandas numpy scikit-learn matplotlib tqdm
Optional (for image features / connected components):
  pip install scipy
Optional (for SBERT embedding, needs network or local model cache):
  pip install sentence-transformers

Typical usage:
  python m3d_refseg_eda_deeper.py --csv "D:\...\M3D_RefSeg.csv" --npy_root "D:\...\M3D_RefSeg_npy" --out_dir ./eda_deeper --all

Or choose steps:
  python m3d_refseg_eda_deeper.py --csv ... --out_dir ... --template_mining --prompt_diversity --tsne_text
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------
# Text heuristics
# ---------------------------

UNCERTAINTY_TERMS = [
    "possible", "possibly", "probable", "probably", "likely", "unlikely",
    "may", "might", "could", "consider", "considering", "suggest", "suggests",
    "suspicious", "suspected", "cannot exclude", "can't exclude", "rule out",
    "appears", "appearance of", "compatible with", "consistent with"
]
NEGATION_TERMS = [
    "no ", "not ", "without", "absence of", "negative for", "free of", "denies"
]
LATERALITY_TERMS = {
    "left": [" left ", "left-sided", "left side", " left.", "left "],
    "right": [" right ", "right-sided", "right side", " right.", "right "],
    "bilateral": ["bilateral", "both sides", "both lungs", "both kidneys"],
}
ANATOMY_GROUPS = {
    "cardio_vascular": ["heart", "cardiac", "coronary", "aorta", "artery", "vein"],
    "pulmonary": ["lung", "pulmonary", "pleura", "bronch", "airway"],
    "hepato_biliary": ["liver", "hepatic", "gallbladder", "biliary"],
    "renal_urinary": ["kidney", "renal", "ureter", "bladder", "pelvis"],
    "gastro_intestinal": ["stomach", "bowel", "colon", "rectum", "pancreas", "spleen"],
    "neuro_headneck": ["brain", "cerebral", "intracranial", "skull", "sinus"],
    "musculoskeletal": ["bone", "vertebra", "spine", "femur", "rib", "fracture"],
    "reproductive": ["prostate", "uterus", "ovary", "testis"],
}

YESNO_PREFIXES = [
    "is there any",
    "are there any",
    "is there a",
    "are there",
    "does the",
    "do we",
    "can we",
]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_case_id(path_str: str) -> str:
    parts = re.split(r"[\\/]+", str(path_str).strip())
    return parts[0] if parts else str(path_str)

def word_count(text: str) -> int:
    toks = re.findall(r"[A-Za-z0-9']+", str(text))
    return len(toks)

def prefix(text: str, n: int = 3) -> str:
    toks = re.findall(r"[A-Za-z']+", str(text).lower())
    return " ".join(toks[:n]) if toks else ""

def text_flags(text: str) -> Tuple[int, int, str, str]:
    t = f" {str(text).lower()} "
    has_unc = int(any(term in t for term in UNCERTAINTY_TERMS))
    has_neg = int(any(term in t for term in NEGATION_TERMS))

    lat = "none"
    for k, terms in LATERALITY_TERMS.items():
        if any(term in t for term in terms):
            lat = k
            break

    anatomy = "other"
    for group, kws in ANATOMY_GROUPS.items():
        if any(kw in t for kw in kws):
            anatomy = group
            break

    return has_unc, has_neg, lat, anatomy

def safe_read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_npy_3d(path: Path) -> np.ndarray:
    arr = np.load(str(path), mmap_mode="r")
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape} from {path}")
    return arr

# ---------------------------
# Embedding helpers
# ---------------------------

def embed_tfidf(texts: List[str], max_features: int = 8000) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )
    X = vec.fit_transform(texts)
    n_components = min(50, X.shape[1] - 1) if X.shape[1] > 1 else 1
    if n_components > 1:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        Xr = svd.fit_transform(X)
    else:
        Xr = X.toarray()
    return Xr.astype(np.float32)

def embed_sbert(texts: List[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)

def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    # cosine similarity, assumes X is dense float
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return (Xn @ Xn.T).astype(np.float32)

# ---------------------------
# Image/ROI features
# ---------------------------

def compute_region_features(
    ct_vol: Optional[np.ndarray],
    mask_vol_int: np.ndarray,
    label_id: int,
    compute_intensity: bool = True,
    compute_components: bool = True,
) -> Dict[str, float]:
    bin_mask = (mask_vol_int == label_id)
    vol = int(bin_mask.sum())
    D, H, W = mask_vol_int.shape

    feat: Dict[str, float] = {
        "label_id": int(label_id),
        "voxels": vol,
        "empty": 1 if vol == 0 else 0,
        "D": int(D), "H": int(H), "W": int(W),
    }

    # Defaults
    def _na(keys):
        for k in keys:
            feat[k] = float("nan")

    bbox_keys = [
        "zmin", "zmax", "ymin", "ymax", "xmin", "xmax",
        "bbox_d", "bbox_h", "bbox_w",
        "centroid_z", "centroid_y", "centroid_x",
        "centroid_x_norm",
        "slices_covered",
        "surface_voxels", "surface_to_volume",
    ]
    _na(bbox_keys)
    if compute_intensity:
        _na(["roi_mean", "roi_std", "roi_min", "roi_max", "roi_p05", "roi_p50", "roi_p95"])
    if compute_components:
        _na(["components"])

    if vol == 0:
        return feat

    coords = np.argwhere(bin_mask)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    feat.update({
        "zmin": int(zmin), "zmax": int(zmax),
        "ymin": int(ymin), "ymax": int(ymax),
        "xmin": int(xmin), "xmax": int(xmax),
        "bbox_d": int(zmax - zmin + 1),
        "bbox_h": int(ymax - ymin + 1),
        "bbox_w": int(xmax - xmin + 1),
        "slices_covered": int(np.unique(coords[:, 0]).size),
    })
    centroid = coords.mean(axis=0)
    feat.update({
        "centroid_z": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_x": float(centroid[2]),
        "centroid_x_norm": float(centroid[2] / (W - 1) if W > 1 else 0.0),
    })

    # boundary approx
    try:
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(bin_mask, iterations=1)
        boundary = bin_mask & (~eroded)
        surface_vox = int(boundary.sum())
        feat["surface_voxels"] = surface_vox
        feat["surface_to_volume"] = float(surface_vox / vol) if vol > 0 else float("nan")
    except Exception:
        pass

    if compute_intensity and ct_vol is not None:
        roi_vals = ct_vol[bin_mask]
        feat.update({
            "roi_mean": float(np.mean(roi_vals)),
            "roi_std": float(np.std(roi_vals)),
            "roi_min": float(np.min(roi_vals)),
            "roi_max": float(np.max(roi_vals)),
            "roi_p05": float(np.percentile(roi_vals, 5)),
            "roi_p50": float(np.percentile(roi_vals, 50)),
            "roi_p95": float(np.percentile(roi_vals, 95)),
        })

    if compute_components:
        try:
            from scipy.ndimage import label
            structure = np.zeros((3, 3, 3), dtype=np.uint8)
            structure[1, 1, :] = 1
            structure[1, :, 1] = 1
            structure[:, 1, 1] = 1
            _, ncomp = label(bin_mask.astype(np.uint8), structure=structure)
            feat["components"] = int(ncomp)
        except Exception:
            pass

    return feat


def make_overlay(ct: np.ndarray, bin_mask: np.ndarray, out_path: Path, title: str = "") -> None:
    import matplotlib.pyplot as plt
    if bin_mask.sum() == 0:
        return
    areas = bin_mask.sum(axis=(1, 2))
    z = int(np.argmax(areas))
    plt.figure(figsize=(6, 6))
    plt.imshow(ct[z], cmap="gray")
    try:
        plt.contour(bin_mask[z].astype(np.uint8), levels=[0.5], linewidths=1)
    except Exception:
        pass
    plt.title(f"{title} (z={z}, area={int(areas[z])})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# ---------------------------
# Plot helpers
# ---------------------------

def plot_bar(counter: Counter, title: str, out_path: Path, top_k: int = 20) -> None:
    import matplotlib.pyplot as plt
    items = counter.most_common(top_k)
    if not items:
        return
    labels, counts = zip(*items)
    plt.figure(figsize=(10, 4.5))
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_hist(values: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 50, log1p: bool = False) -> None:
    import matplotlib.pyplot as plt
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float).values
    if log1p:
        v = np.log1p(v)
    plt.figure(figsize=(7, 4.5))
    plt.hist(v, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel + (" (log1p)" if log1p else ""))
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def run_tsne(X: np.ndarray, meta: pd.DataFrame, out_dir: Path, prefix: str, color_cols: List[str], perplexity: float = 30.0, seed: int = 42) -> None:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    n = X.shape[0]
    if n < 5:
        print(f"[t-SNE] Too few samples ({n}), skip.")
        return

    perplexity = min(perplexity, max(2.0, (n - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed)
    Y = tsne.fit_transform(X)

    coords = meta.copy()
    coords["tsne_x"] = Y[:, 0]
    coords["tsne_y"] = Y[:, 1]
    coords.to_csv(out_dir / f"{prefix}_tsne_coords.csv", index=False)

    for col in color_cols:
        if col not in coords.columns:
            continue
        plt.figure(figsize=(7, 6))
        c = coords[col]
        if pd.api.types.is_numeric_dtype(c):
            sc = plt.scatter(coords["tsne_x"], coords["tsne_y"], c=c, s=10)
            plt.colorbar(sc, label=col)
        else:
            cats = c.astype(str).fillna("NA")
            uniq = sorted(cats.unique())
            mapping = {u: i for i, u in enumerate(uniq)}
            ci = cats.map(mapping).values
            plt.scatter(coords["tsne_x"], coords["tsne_y"], c=ci, s=10)
            if len(uniq) <= 12:
                handles = [plt.Line2D([0], [0], marker='o', linestyle='', label=u, markersize=6) for u in uniq]
                plt.legend(handles=handles, title=col, bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.title(f"t-SNE colored by {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_tsne_{col}.png", dpi=220)
        plt.close()


# ---------------------------
# Main workflow
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Deeper EDA for M3D-RefSeg (npy)")
    ap.add_argument("--csv", required=True, help="Path to M3D_RefSeg.csv")
    ap.add_argument("--npy_root", default="", help="Path to M3D_RefSeg_npy/ (optional but recommended)")
    ap.add_argument("--out_dir", default="./eda_deeper", help="Output directory")
    ap.add_argument("--max_rows", type=int, default=0, help="Limit rows for debugging (0=all)")
    ap.add_argument("--max_cases", type=int, default=0, help="Limit cases for image features (0=all)")

    # steps
    ap.add_argument("--template_mining", action="store_true")
    ap.add_argument("--prompt_diversity", action="store_true")
    ap.add_argument("--tsne_text", action="store_true")
    ap.add_argument("--build_synthetic_reports", action="store_true")
    ap.add_argument("--compute_image_features", action="store_true")
    ap.add_argument("--n_overlay", type=int, default=0)

    # options
    ap.add_argument("--embed_method", choices=["tfidf", "sbert"], default="tfidf")
    ap.add_argument("--sbert_model", default="all-MiniLM-L6-v2")
    ap.add_argument("--tsne_mode", choices=["question_row", "label_desc_region"], default="label_desc_region")
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--clean_csv_out", default="", help="If set, write a cleaned CSV (filters rows without [SEG])")
    ap.add_argument("--all", action="store_true", help="Run template_mining + prompt_diversity + tsne_text + build_synthetic_reports (image features excluded)")

    args = ap.parse_args()
    np.random.seed(args.seed)

    if args.all:
        args.template_mining = True
        args.prompt_diversity = True
        args.tsne_text = True
        args.build_synthetic_reports = True

    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figs"
    synth_dir = out_dir / "synthetic"
    overlay_dir = out_dir / "overlays"
    ensure_dir(tables_dir)
    ensure_dir(figs_dir)
    ensure_dir(synth_dir)
    ensure_dir(overlay_dir)

    df = pd.read_csv(args.csv)
    if args.max_rows and args.max_rows > 0:
        df = df.iloc[: args.max_rows].copy()

    # Basic columns
    df["case_id"] = df["Image"].map(parse_case_id) if "Image" in df.columns else df.get("case_id", "")
    df["region_id"] = df["case_id"].astype(str) + "__" + df["Mask_ID"].astype(str)

    # Row-level text features
    df["q_words"] = df["Question"].map(word_count)
    df["a_words"] = df["Answer"].map(word_count)
    df["has_seg_token"] = df["Answer"].astype(str).str.contains(r"\[SEG\]", regex=True).astype(int)
    df["q_prefix3"] = df["Question"].map(lambda t: prefix(t, 3))

    flags = df["Question"].map(text_flags)
    df["q_has_unc"] = flags.map(lambda x: x[0])
    df["q_has_neg"] = flags.map(lambda x: x[1])
    df["q_laterality"] = flags.map(lambda x: x[2])
    df["q_anatomy_group"] = flags.map(lambda x: x[3])

    df.to_csv(tables_dir / "rows_text_features_v2.csv", index=False)

    # Clean CSV (for segmentation training)
    if args.clean_csv_out:
        clean = df[df["has_seg_token"] == 1].copy()
        Path(args.clean_csv_out).parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(args.clean_csv_out, index=False)
        print(f"[CLEAN] Wrote cleaned CSV (keep [SEG] only) to: {args.clean_csv_out}")

    # Region-level aggregation (from QA table)
    region_df = (
        df.groupby(["region_id", "case_id", "Mask_ID"], as_index=False)
          .agg(
              qa_rows=("Question", "size"),
              type0_rows=("Question_Type", lambda x: int((pd.Series(x) == 0).sum()) if len(x) else 0),
              type1_rows=("Question_Type", lambda x: int((pd.Series(x) == 1).sum()) if len(x) else 0),
              q_words_mean=("q_words", "mean"),
              a_words_mean=("a_words", "mean"),
              unc_rate=("q_has_unc", "mean"),
              neg_rate=("q_has_neg", "mean"),
              any_missing_seg=("has_seg_token", lambda x: int((pd.Series(x) == 0).any())),
              questions_joined=("Question", lambda x: " || ".join(x.astype(str).tolist())),
          )
    )

    # Load label_desc from text.json if npy_root exists
    npy_root = Path(args.npy_root) if args.npy_root else None
    case_to_text: Dict[str, Dict[str, str]] = {}
    if npy_root and npy_root.exists():
        cases = sorted(region_df["case_id"].unique().tolist())
        if args.max_cases and args.max_cases > 0:
            cases = cases[: args.max_cases]
        for cid in cases:
            tj = npy_root / cid / "text.json"
            m = safe_read_json(tj)
            if isinstance(m, dict):
                case_to_text[cid] = {str(k): str(v) for k, v in m.items()}

    def lookup_label_desc(row) -> str:
        cid = row["case_id"]
        mid = str(int(row["Mask_ID"]))
        return case_to_text.get(cid, {}).get(mid, "")

    region_df["label_desc"] = region_df.apply(lookup_label_desc, axis=1) if case_to_text else ""
    # use label_desc for anatomy/laterality if available, else fallback to questions
    def region_anatomy(row) -> str:
        base = row["label_desc"] if isinstance(row["label_desc"], str) and row["label_desc"].strip() else row["questions_joined"]
        return text_flags(base)[3]
    def region_laterality(row) -> str:
        base = row["label_desc"] if isinstance(row["label_desc"], str) and row["label_desc"].strip() else row["questions_joined"]
        return text_flags(base)[2]

    region_df["anatomy_group"] = region_df.apply(region_anatomy, axis=1)
    region_df["laterality"] = region_df.apply(region_laterality, axis=1)

    region_df.to_csv(tables_dir / "regions_summary_v2.csv", index=False)

    # Template mining
    if args.template_mining:
        print("[STEP] Template mining...")
        prefix_counts = Counter(df["q_prefix3"].tolist())
        prefix_df = pd.DataFrame(prefix_counts.most_common(), columns=["prefix3", "count"])
        prefix_df.to_csv(tables_dir / "question_prefix3_counts.csv", index=False)
        plot_bar(prefix_counts, "Top Question Prefixes (first 3 words)", figs_dir / "top_question_prefix3.png", top_k=25)

        # Also flag yes/no style questions
        def is_yesno_pref(p: str) -> int:
            p = p.strip().lower()
            return int(any(p.startswith(x) for x in YESNO_PREFIXES))
        df["is_yesno_style"] = df["q_prefix3"].map(is_yesno_pref)
        df["is_yesno_style"].value_counts().to_csv(tables_dir / "yesno_style_counts.csv")

    # Prompt diversity (pairwise cosine similarity among paraphrases)
    if args.prompt_diversity:
        print("[STEP] Prompt diversity per region...")
        # Embed ALL questions once (row-level), then compute per-region similarity
        texts = df["Question"].astype(str).tolist()
        texts = [t if t.strip() else "EMPTY_TEXT" for t in texts]

        if args.embed_method == "tfidf":
            X = embed_tfidf(texts)
        else:
            try:
                X = embed_sbert(texts, args.sbert_model)
            except Exception as e:
                print(f"[WARN] SBERT failed ({e}), fallback to TF-IDF.")
                X = embed_tfidf(texts)

        # Normalize for cosine
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # Map row index by region
        idx_by_region: Dict[str, List[int]] = defaultdict(list)
        for i, rid in enumerate(df["region_id"].tolist()):
            idx_by_region[rid].append(i)

        records = []
        for rid, idxs in tqdm(idx_by_region.items(), desc="Regions"):
            if len(idxs) < 2:
                continue
            v = Xn[idxs]
            sim = v @ v.T
            # take upper triangle without diagonal
            iu = np.triu_indices(sim.shape[0], k=1)
            sims = sim[iu]
            records.append({
                "region_id": rid,
                "n_prompts": len(idxs),
                "mean_cos_sim": float(np.mean(sims)) if sims.size else float("nan"),
                "min_cos_sim": float(np.min(sims)) if sims.size else float("nan"),
                "max_cos_sim": float(np.max(sims)) if sims.size else float("nan"),
                "std_cos_sim": float(np.std(sims)) if sims.size else float("nan"),
                "diversity_1_minus_mean": float(1.0 - np.mean(sims)) if sims.size else float("nan"),
            })

        div_df = pd.DataFrame(records)
        div_df.to_csv(tables_dir / "prompt_diversity_per_region.csv", index=False)
        plot_hist(div_df["mean_cos_sim"].to_numpy(), "Prompt similarity within each region", "Mean cosine similarity", figs_dir / "prompt_similarity_mean_hist.png", bins=50)
        plot_hist(div_df["diversity_1_minus_mean"].to_numpy(), "Prompt diversity within each region", "1 - mean cosine similarity", figs_dir / "prompt_diversity_hist.png", bins=50)

    # Build synthetic multi-finding reports
    if args.build_synthetic_reports:
        print("[STEP] Building synthetic multi-finding reports (case-level)...")

        # Prefer label_desc (needs npy_root). If missing, fallback to questions_joined.
        if "label_desc" in region_df.columns and region_df["label_desc"].astype(str).str.strip().ne("").any():
            findings = region_df[["case_id", "Mask_ID", "label_desc", "anatomy_group", "laterality"]].copy()
            findings = findings.rename(columns={"label_desc": "finding_text"})
        else:
            findings = region_df[["case_id", "Mask_ID", "questions_joined", "anatomy_group", "laterality"]].copy()
            findings = findings.rename(columns={"questions_joined": "finding_text"})

        # Build report per case by concatenating all findings
        case_reports = []
        for cid, g in findings.groupby("case_id"):
            finding_list = g["finding_text"].astype(str).tolist()
            # simple enumerated report
            report = "\n".join([f"{i+1}. {t}" for i, t in enumerate(finding_list)])
            case_reports.append({
                "case_id": cid,
                "n_findings": int(len(finding_list)),
                "report_text": report,
                "report_words": int(word_count(report)),
            })
        case_reports_df = pd.DataFrame(case_reports)
        case_reports_df.to_csv(synth_dir / "synthetic_case_reports.csv", index=False)
        plot_hist(case_reports_df["n_findings"].to_numpy(), "Findings per case (synthetic reports)", "n_findings", figs_dir / "findings_per_case_hist.png", bins=20)
        plot_hist(case_reports_df["report_words"].to_numpy(), "Synthetic report length (words)", "words", figs_dir / "synthetic_report_words_hist.png", bins=40)

        # Build a region-level dataset where input is long report, target is one finding (Mask_ID)
        merged = findings.merge(case_reports_df[["case_id", "n_findings", "report_text", "report_words"]], on="case_id", how="left")
        # A simple “query”: use the target finding sentence itself (more realistic disambiguation can be added later)
        merged["target_finding_text"] = merged["finding_text"]
        merged["input_text"] = merged["report_text"]
        merged["task_instruction"] = merged["target_finding_text"].map(lambda t: f"Please segment the finding described as: {t}")
        merged.to_csv(synth_dir / "synthetic_report_to_roi_dataset.csv", index=False)

    # Text t-SNE
    if args.tsne_text:
        print("[STEP] t-SNE on text embeddings...")
        if args.tsne_mode == "question_row":
            meta = df[["case_id", "Mask_ID", "region_id", "Question_Type", "q_has_unc", "q_has_neg", "q_anatomy_group", "q_laterality"]].copy()
            texts = df["Question"].astype(str).tolist()
            color_cols = ["Question_Type", "q_anatomy_group", "q_laterality", "q_has_unc"]
            tsne_prefix = f"text_question_row_{args.embed_method}"
        else:
            meta = region_df[["case_id", "Mask_ID", "region_id", "qa_rows", "type1_rows", "anatomy_group", "laterality", "unc_rate"]].copy()
            texts = region_df["label_desc"].astype(str).tolist() if "label_desc" in region_df.columns else region_df["questions_joined"].astype(str).tolist()
            color_cols = ["anatomy_group", "laterality", "type1_rows", "unc_rate"]
            tsne_prefix = f"text_label_desc_region_{args.embed_method}"

        texts = [t if t.strip() else "EMPTY_TEXT" for t in texts]
        if args.embed_method == "tfidf":
            X = embed_tfidf(texts)
        else:
            try:
                X = embed_sbert(texts, args.sbert_model)
            except Exception as e:
                print(f"[WARN] SBERT failed ({e}), fallback to TF-IDF.")
                X = embed_tfidf(texts)

        run_tsne(X, meta, figs_dir, prefix=tsne_prefix, color_cols=color_cols, perplexity=args.tsne_perplexity, seed=args.seed)

    # Image features (optional)
    if args.compute_image_features:
        if not (npy_root and npy_root.exists()):
            print("[WARN] --compute_image_features requires --npy_root. Skipping.")
        else:
            print("[STEP] Computing image/ROI features from ct.npy/mask.npy ...")
            cases = sorted(region_df["case_id"].unique().tolist())
            if args.max_cases and args.max_cases > 0:
                cases = cases[: args.max_cases]

            # region ids per case
            mids_by_case = defaultdict(list)
            for cid, mid in region_df[["case_id", "Mask_ID"]].itertuples(index=False):
                mids_by_case[str(cid)].append(int(mid))

            feat_rows = []
            overlay_candidates = []

            for cid in tqdm(cases, desc="Cases"):
                case_dir = npy_root / cid
                ct_path = case_dir / "ct.npy"
                mask_path = case_dir / "mask.npy"
                if not mask_path.exists():
                    continue
                try:
                    mask_vol = load_npy_3d(mask_path)
                    mask_int = np.rint(mask_vol).astype(np.int32)
                except Exception:
                    continue

                ct_vol = None
                if ct_path.exists():
                    try:
                        ct_vol = load_npy_3d(ct_path).astype(np.float32)
                    except Exception:
                        ct_vol = None

                for mid in sorted(set(mids_by_case[cid])):
                    feat = {"case_id": cid, "Mask_ID": int(mid), "region_id": f"{cid}__{mid}"}
                    feat.update(compute_region_features(
                        ct_vol=ct_vol,
                        mask_vol_int=mask_int,
                        label_id=int(mid),
                        compute_intensity=True,  # uses ct if available
                        compute_components=True,
                    ))
                    feat_rows.append(feat)

                    if args.n_overlay > 0 and ct_vol is not None and feat["voxels"] > 0:
                        overlay_candidates.append((feat["voxels"], cid, mid))

            feat_df = pd.DataFrame(feat_rows)
            feat_df.to_csv(tables_dir / "regions_image_features_v2.csv", index=False)

            plot_hist(feat_df["voxels"].to_numpy(), "ROI voxels per region", "voxels", figs_dir / "roi_voxels_hist.png", bins=60)
            plot_hist(feat_df["voxels"].to_numpy(), "ROI voxels per region", "voxels", figs_dir / "roi_voxels_hist_log1p.png", bins=60, log1p=True)
            plot_hist(feat_df["bbox_d"].to_numpy(), "ROI bbox depth (D)", "bbox_d", figs_dir / "roi_bbox_d_hist.png", bins=40)
            plot_hist(feat_df["bbox_h"].to_numpy(), "ROI bbox height (H)", "bbox_h", figs_dir / "roi_bbox_h_hist.png", bins=40)
            plot_hist(feat_df["bbox_w"].to_numpy(), "ROI bbox width (W)", "bbox_w", figs_dir / "roi_bbox_w_hist.png", bins=40)

            empty_df = feat_df[feat_df["empty"] == 1].copy()
            empty_df.to_csv(tables_dir / "empty_regions_after_prepare_v2.csv", index=False)

            # overlays: pick top-N by voxels
            if args.n_overlay > 0 and overlay_candidates:
                overlay_candidates.sort(key=lambda x: x[0], reverse=True)
                chosen = overlay_candidates[: args.n_overlay]
                print(f"[QC] Saving {len(chosen)} overlays...")
                for vox, cid, mid in tqdm(chosen, desc="Overlays"):
                    case_dir = npy_root / cid
                    ct_path = case_dir / "ct.npy"
                    mask_path = case_dir / "mask.npy"
                    if not ct_path.exists() or not mask_path.exists():
                        continue
                    try:
                        ct_vol = load_npy_3d(ct_path).astype(np.float32)
                        mask_int = np.rint(load_npy_3d(mask_path)).astype(np.int32)
                    except Exception:
                        continue
                    bin_mask = (mask_int == int(mid))
                    out_path = overlay_dir / f"{cid}__{mid}_vox{int(vox)}.png"
                    make_overlay(ct_vol, bin_mask, out_path, title=f"{cid} Mask_ID={mid}")

    print("\nDone. Outputs in:", out_dir.resolve())


if __name__ == "__main__":
    main()
