
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit leakage / overlap risk for M3D-RefSeg when you split data at different granularities.

Why:
- RefSeg CSV has multiple QA rows per (case_id, Mask_ID) region.
- If you split by QA rows, the exact same ROI (mask) can leak across train/test via paraphrased questions.

This script:
1) Reads M3D_RefSeg.csv
2) Creates region_id = case_id__Mask_ID
3) Simulates random splits at 3 levels:
   - row-level (QA row)
   - region-level (unique region_id)
   - case-level (unique case_id)
4) Reports overlaps of case_id and region_id across splits, and template overlap.

Outputs: JSON + CSV in out_dir
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def parse_case_id(image_path: str) -> str:
    parts = str(image_path).replace("\\", "/").split("/")
    return parts[0] if parts else str(image_path)

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


def split_units(units: List[str], fracs: Tuple[float, float, float], seed: int) -> Dict[str, str]:
    rng = np.random.default_rng(seed)
    units = list(units)
    rng.shuffle(units)
    n = len(units)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    n_test = n - n_train - n_val
    # fix rounding if needed
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    splits = {}
    for u in units[:n_train]:
        splits[u] = "train"
    for u in units[n_train:n_train+n_val]:
        splits[u] = "val"
    for u in units[n_train+n_val:]:
        splits[u] = "test"
    return splits


def overlap_report(df: pd.DataFrame, split_col: str) -> Dict:
    rep = {}
    for level in ["case_id", "region_id"]:
        sets = {s: set(df[df[split_col] == s][level].unique().tolist()) for s in ["train", "val", "test"]}
        rep[level] = {
            "n_train": len(sets["train"]),
            "n_val": len(sets["val"]),
            "n_test": len(sets["test"]),
            "train_val_overlap": len(sets["train"] & sets["val"]),
            "train_test_overlap": len(sets["train"] & sets["test"]),
            "val_test_overlap": len(sets["val"] & sets["test"]),
        }
    return rep


def template_overlap(df: pd.DataFrame, split_col: str) -> Dict:
    df = df.copy()
    df["q_template"] = df["Question"].map(normalize_question).map(remove_laterality_tokens)
    sets = {s: set(df[df[split_col] == s]["q_template"].unique().tolist()) for s in ["train", "val", "test"]}
    rep = {
        "n_templates_train": len(sets["train"]),
        "n_templates_val": len(sets["val"]),
        "n_templates_test": len(sets["test"]),
        "train_test_template_overlap": len(sets["train"] & sets["test"]),
        "train_val_template_overlap": len(sets["train"] & sets["val"]),
        "val_test_template_overlap": len(sets["val"] & sets["test"]),
    }
    # Overlap ratios (how much of test templates were already seen in train)
    rep["test_templates_covered_by_train"] = float(len(sets["train"] & sets["test"]) / max(1, len(sets["test"])))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to M3D_RefSeg.csv")
    ap.add_argument("--out_dir", default="./leakage_audit_out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    args = ap.parse_args()

    fracs = (args.train_frac, args.val_frac, args.test_frac)
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError("train/val/test fracs must sum to 1.0")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(args.csv)
    df["case_id"] = df["Image"].map(parse_case_id)
    df["Mask_ID"] = df["Mask_ID"].astype(int)
    df["region_id"] = df["case_id"].astype(str) + "__" + df["Mask_ID"].astype(str)

    results = {}

    # row-level split
    rng = np.random.default_rng(args.seed)
    splits = rng.choice(["train", "val", "test"], size=len(df), p=list(fracs))
    df_row = df.copy()
    df_row["split"] = splits
    results["row_level"] = {
        "overlap": overlap_report(df_row, "split"),
        "template_overlap": template_overlap(df_row, "split"),
    }

    # region-level split
    unit_split = split_units(sorted(df["region_id"].unique().tolist()), fracs, args.seed)
    df_reg = df.copy()
    df_reg["split"] = df_reg["region_id"].map(unit_split)
    results["region_level"] = {
        "overlap": overlap_report(df_reg, "split"),
        "template_overlap": template_overlap(df_reg, "split"),
    }

    # case-level split
    unit_split = split_units(sorted(df["case_id"].unique().tolist()), fracs, args.seed)
    df_case = df.copy()
    df_case["split"] = df_case["case_id"].map(unit_split)
    results["case_level"] = {
        "overlap": overlap_report(df_case, "split"),
        "template_overlap": template_overlap(df_case, "split"),
    }

    save_json(results, out_dir / "split_leakage_audit.json")

    # also write small CSV summaries
    def flatten(prefix: str, d: Dict) -> List[Dict]:
        rows = []
        for level, rep in d["overlap"].items():
            row = {"split_mode": prefix, "level": level}
            row.update(rep)
            rows.append(row)
        return rows

    flat = []
    for mode in ["row_level", "region_level", "case_level"]:
        flat += flatten(mode, results[mode])
    pd.DataFrame(flat).to_csv(out_dir / "overlap_summary.csv", index=False)

    tmp = []
    for mode in ["row_level", "region_level", "case_level"]:
        row = {"split_mode": mode}
        row.update(results[mode]["template_overlap"])
        tmp.append(row)
    pd.DataFrame(tmp).to_csv(out_dir / "template_overlap_summary.csv", index=False)

    print("Done. Saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
