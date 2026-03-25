#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
High-level template bias analysis for M3D-RefSeg questions.

Outputs:
- tables/top_prefix_templates_n{N}.csv
- tables/top_frame_templates.csv
- tables/template_bias_summary.json
- figs/top_prefix_templates_n{N}.png
- figs/top_frame_templates.png
- figs/template_coverage_curves.png
- figs/template_concentration_by_type.png (if Question_Type exists)

Run:
  python eda_highlevel_templates.py --csv "D:\\M3D\\M3D_RefSeg\\M3D_RefSeg.csv" --out_dir ".\\tmpl_out"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# ---------- Basic text normalization ----------
LATERALITY_RE = re.compile(r"\b(left|right|bilateral)\b", re.I)
NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")
PUNCT_RE = re.compile(r"[^a-z0-9<>\s]+")

def normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = NUM_RE.sub("<num>", s)
    s = LATERALITY_RE.sub("<lat>", s)
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    if not s:
        return []
    return s.split()


# ---------- High-level "frame" template ----------
# A small stopword/function-word list (no external deps).
STOPWORDS = {
    "a","an","the","this","that","these","those",
    "is","are","was","were","be","been","being",
    "do","does","did","can","could","would","should","may","might","will","please",
    "you","we","i","me","us","our","your",
    "where","what","which","why","how",
    "in","on","at","to","from","of","for","with","without","within","into","over","under","between",
    "and","or","but","as","by","than","then",
    "there","any","some","such","it","its","they","them","their",
    "<num>","<lat>"
}

# Keep a whitelist of key action/intent words that define the question frame.
FRAME_KEYWORDS = {
    "segment","segmentation","locate","localize","identify","find","show","mark",
    "appear","appears","located","location","region","area","lesion","abnormal","abnormality",
    "ct","image","scan","volume"
}

def to_prefix_template(tokens: List[str], n: int) -> str:
    if not tokens:
        return ""
    return " ".join(tokens[:n])

def compress_placeholders(tokens: List[str], placeholder: str = "<ent>") -> List[str]:
    out = []
    for t in tokens:
        if out and out[-1] == placeholder and t == placeholder:
            continue
        out.append(t)
    return out

def to_frame_template(tokens: List[str]) -> str:
    """
    Convert a question into a higher-level frame:
    - Keep STOPWORDS and FRAME_KEYWORDS
    - Replace other content words with <ent>
    - Compress consecutive <ent>
    """
    if not tokens:
        return ""
    framed = []
    for t in tokens:
        if t in STOPWORDS or t in FRAME_KEYWORDS:
            framed.append(t)
        else:
            framed.append("<ent>")
    framed = compress_placeholders(framed, "<ent>")
    return " ".join(framed)


# ---------- Bias / concentration metrics ----------
def shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def herfindahl_hhi(p: np.ndarray) -> float:
    return float((p ** 2).sum())

def effective_num_templates(p: np.ndarray) -> float:
    # 1 / HHI
    hhi = herfindahl_hhi(p)
    return float(1.0 / hhi) if hhi > 0 else float("nan")

def gini_coefficient(counts: np.ndarray) -> float:
    # Gini for non-negative counts
    x = np.array(counts, dtype=float)
    if x.size == 0:
        return float("nan")
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    g = (n + 1 - 2 * (cumx.sum() / cumx[-1])) / n
    return float(g)

def topk_coverage(counts: np.ndarray, k: int) -> float:
    if counts.size == 0:
        return 0.0
    c = np.sort(counts)[::-1]
    return float(c[:k].sum() / c.sum()) if c.sum() > 0 else 0.0


# ---------- Plot helpers ----------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def plot_top_bar(df: pd.DataFrame, label_col: str, count_col: str, title: str, out_path: Path, top_n: int = 20):
    import matplotlib.pyplot as plt

    d = df.head(top_n).copy()
    labels = d[label_col].astype(str).tolist()
    counts = d[count_col].astype(int).tolist()

    # shorten long labels
    short = []
    for s in labels:
        short.append(s[:70] + ("..." if len(s) > 70 else ""))

    plt.figure(figsize=(10, 6))
    plt.barh(list(reversed(short)), list(reversed(counts)))
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_coverage_curve(counts: np.ndarray, title: str, out_path: Path, max_k: int = 100):
    import matplotlib.pyplot as plt

    c = np.sort(counts)[::-1]
    total = c.sum()
    if total <= 0:
        return
    ks = np.arange(1, min(max_k, len(c)) + 1)
    cov = np.cumsum(c[: len(ks)]) / total

    plt.figure(figsize=(7, 4.6))
    plt.plot(ks, cov)
    plt.title(title)
    plt.xlabel("Top-K templates")
    plt.ylabel("Coverage")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def compute_and_save_template_stats(
    name: str,
    templates: pd.Series,
    out_tables: Path,
    out_figs: Path,
    plot_title: str,
    top_n_plot: int = 20
) -> Dict:
    """
    templates: a series of template strings (one per question row)
    """
    vc = templates.value_counts().reset_index()
    vc.columns = ["template", "count"]
    vc.to_csv(out_tables / f"top_{name}_templates.csv", index=False)

    # metrics
    counts = vc["count"].to_numpy(dtype=int)
    p = counts / counts.sum() if counts.sum() > 0 else np.array([])

    summary = {
        "name": name,
        "n_rows": int(templates.shape[0]),
        "n_unique_templates": int(vc.shape[0]),
        "top1_coverage": topk_coverage(counts, 1),
        "top5_coverage": topk_coverage(counts, 5),
        "top10_coverage": topk_coverage(counts, 10),
        "top20_coverage": topk_coverage(counts, 20),
        "hhi": herfindahl_hhi(p) if p.size else None,
        "effective_num_templates": effective_num_templates(p) if p.size else None,
        "entropy_bits": shannon_entropy(p) if p.size else None,
        "gini": gini_coefficient(counts) if counts.size else None,
    }

    plot_top_bar(vc, "template", "count", plot_title, out_figs / f"top_{name}_templates.png", top_n=top_n_plot)
    plot_coverage_curve(counts, f"Coverage curve: {name}", out_figs / f"coverage_curve_{name}.png", max_k=100)

    return summary


def main():
    ap = argparse.ArgumentParser("High-level template bias analysis (prefix + frame)")
    ap.add_argument("--csv", required=True, help="Path to M3D_RefSeg.csv")
    ap.add_argument("--out_dir", default="./tmpl_out")
    ap.add_argument("--prefix_ns", default="3,4,5", help="Comma-separated prefix lengths, e.g. 3,4,5")
    ap.add_argument("--top_plot", type=int, default=20, help="Top-N bars to plot")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_tables = out_dir / "tables"
    out_figs = out_dir / "figs"
    ensure_dir(out_tables)
    ensure_dir(out_figs)

    df = pd.read_csv(args.csv)
    if "Question" not in df.columns:
        raise ValueError("CSV must contain 'Question' column")

    q = df["Question"].fillna("").astype(str)
    tokens_list = q.map(tokenize)

    # 1) prefix templates for several N
    prefix_ns = [int(x.strip()) for x in args.prefix_ns.split(",") if x.strip()]
    summaries = []

    for n in prefix_ns:
        templates = tokens_list.map(lambda toks: to_prefix_template(toks, n))
        templates = templates.replace("", "EMPTY")
        s = compute_and_save_template_stats(
            name=f"prefix_n{n}",
            templates=templates,
            out_tables=out_tables,
            out_figs=out_figs,
            plot_title=f"Top-{args.top_plot} prefix templates (n={n})",
            top_n_plot=args.top_plot
        )
        summaries.append(s)

    # 2) frame templates
    frame_templates = tokens_list.map(to_frame_template).replace("", "EMPTY")
    s_frame = compute_and_save_template_stats(
        name="frame",
        templates=frame_templates,
        out_tables=out_tables,
        out_figs=out_figs,
        plot_title=f"Top-{args.top_plot} frame templates (content -> <ENT>)",
        top_n_plot=args.top_plot
    )
    summaries.append(s_frame)

    # 3) If Question_Type exists, compute per-type concentration (nice for PPT)
    per_type = {}
    if "Question_Type" in df.columns:
        df2 = df.copy()
        df2["frame"] = frame_templates.values
        for t, sub in df2.groupby("Question_Type"):
            vc = sub["frame"].value_counts()
            counts = vc.to_numpy(dtype=int)
            p = counts / counts.sum() if counts.sum() > 0 else np.array([])
            per_type[str(t)] = {
                "n_rows": int(len(sub)),
                "n_unique_frame_templates": int(vc.shape[0]),
                "top10_coverage": topk_coverage(counts, 10),
                "top20_coverage": topk_coverage(counts, 20),
                "hhi": herfindahl_hhi(p) if p.size else None,
                "effective_num_templates": effective_num_templates(p) if p.size else None,
                "entropy_bits": shannon_entropy(p) if p.size else None,
            }

        # plot per-type top20 coverage
        import matplotlib.pyplot as plt
        labels = list(per_type.keys())
        vals = [per_type[k]["top20_coverage"] for k in labels]
        plt.figure(figsize=(6.6, 4.6))
        plt.bar(labels, vals)
        plt.title("Frame template concentration by Question_Type")
        plt.xlabel("Question_Type")
        plt.ylabel("Top-20 coverage (frame)")
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(out_figs / "template_concentration_by_type.png", dpi=220)
        plt.close()

    # save summary
    out_summary = {
        "notes": {
            "prefix_template": "first N tokens after normalization (<NUM>, <LAT>)",
            "frame_template": "keep stopwords + key action words, replace other content with <ENT>",
            "interpretation": "Higher Top-K coverage / higher HHI / lower effective_num indicates stronger templating (potential shortcut risk)."
        },
        "summaries": summaries,
        "per_question_type_frame": per_type
    }
    (out_tables / "template_bias_summary.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")

    print("Done. See:", out_dir.resolve())


if __name__ == "__main__":
    main()
