#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RefSeg split generator + leakage audit

Why needed:
- RefSeg has multiple paraphrase rows per region (case_id + Mask_ID).
- Random row split leaks same region into train/test -> inflated results.

This script:
A) Make leak-free split by grouping on case_id or region_id
B) Audit leakage across existing splits (case overlap / region overlap)
C) (Optional) report how many normalized question templates overlap across splits

Inputs:
- full CSV: M3D_RefSeg.csv (must contain Image, Mask_ID, Question)
Outputs:
- splits/train.csv val.csv test.csv
- tables/split_summary.json
- tables/split_leakage_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


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


def normalize_question(q: str, neutralize_laterality: bool = True) -> str:
    q = str(q).lower().strip()
    q = re.sub(r"\d+", "<num>", q)
    q = re.sub(r"[^a-z0-9<>\s]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    if neutralize_laterality:
        q = re.sub(r"\b(left|right|bilateral)\b", "<lat>", q)
    return q


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    if "Image" not in df.columns or "Mask_ID" not in df.columns:
        raise ValueError("CSV must contain columns: Image, Mask_ID")
    out = df.copy()
    out["case_id"] = out["Image"].map(infer_case_id)
    out["Mask_ID"] = out["Mask_ID"].astype(int)
    out["region_id"] = out["case_id"].astype(str) + "__" + out["Mask_ID"].astype(str)
    if "Question" in out.columns:
        out["q_norm"] = out["Question"].map(lambda x: normalize_question(x, neutralize_laterality=True))
    else:
        out["q_norm"] = ""
    return out


def make_group_split(df: pd.DataFrame, split_by: str, ratios: Tuple[float, float, float], seed: int) -> pd.DataFrame:
    train_r, val_r, test_r = ratios
    if not np.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    if split_by not in {"case", "region"}:
        raise ValueError("--split_by must be 'case' or 'region'")

    rng = np.random.default_rng(seed)
    key = "case_id" if split_by == "case" else "region_id"
    groups = sorted(df[key].unique().tolist())
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    # remainder -> test
    g_train = set(groups[:n_train])
    g_val = set(groups[n_train:n_train + n_val])
    g_test = set(groups[n_train + n_val:])

    out = df.copy()
    out["split"] = np.where(out[key].isin(g_train), "train",
                            np.where(out[key].isin(g_val), "val", "test"))
    return out


def split_summary(df: pd.DataFrame) -> Dict:
    out = {}
    for sp, sub in df.groupby("split"):
        out[sp] = {
            "n_rows": int(len(sub)),
            "n_cases": int(sub["case_id"].nunique()),
            "n_regions": int(sub["region_id"].nunique()),
            "n_unique_q_norm": int(sub["q_norm"].nunique()),
        }
    return out


def audit_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict:
    splits = {"train": train, "val": val, "test": test}
    names = ["train", "val", "test"]

    region_sets = {n: set(splits[n]["region_id"].unique().tolist()) for n in names}
    case_sets = {n: set(splits[n]["case_id"].unique().tolist()) for n in names}
    q_sets = {n: set(splits[n]["q_norm"].unique().tolist()) for n in names}

    report = {"status": "ok", "split_sizes": {}}
    for n in names:
        report["split_sizes"][n] = {
            "n_rows": int(len(splits[n])),
            "n_cases": int(splits[n]["case_id"].nunique()),
            "n_regions": int(splits[n]["region_id"].nunique()),
            "n_unique_q_norm": int(splits[n]["q_norm"].nunique()),
        }

    def pairwise_intersection(a: str, b: str, sets: Dict[str, set], k: int = 30):
        inter = sorted(list(sets[a].intersection(sets[b])))
        return {"n": len(inter), "examples": inter[:k]}

    report["case_overlap"] = {
        "train_val": pairwise_intersection("train", "val", case_sets),
        "train_test": pairwise_intersection("train", "test", case_sets),
        "val_test": pairwise_intersection("val", "test", case_sets),
    }
    report["region_overlap"] = {
        "train_val": pairwise_intersection("train", "val", region_sets),
        "train_test": pairwise_intersection("train", "test", region_sets),
        "val_test": pairwise_intersection("val", "test", region_sets),
    }

    # template overlap (not necessarily leakage, but indicates heavy templating)
    report["q_norm_overlap"] = {
        "train_val": pairwise_intersection("train", "val", q_sets),
        "train_test": pairwise_intersection("train", "test", q_sets),
        "val_test": pairwise_intersection("val", "test", q_sets),
    }
    return report


def main():
    ap = argparse.ArgumentParser("RefSeg split maker + leakage audit")
    ap.add_argument("--csv", required=True, help="Path to M3D_RefSeg.csv")
    ap.add_argument("--out_dir", default="./refseg_split_out", help="Output dir")

    ap.add_argument("--make_split", action="store_true", help="Generate a new split")
    ap.add_argument("--split_by", choices=["case", "region"], default="case",
                    help="Group key for leak-free split (recommended: case)")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--audit_only", action="store_true", help="Only audit provided split CSVs")
    ap.add_argument("--train_csv", default="", help="Existing train.csv to audit")
    ap.add_argument("--val_csv", default="", help="Existing val.csv to audit")
    ap.add_argument("--test_csv", default="", help="Existing test.csv to audit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    tables = out_dir / "tables"
    splits_dir = out_dir / "splits"
    ensure_dir(tables)
    ensure_dir(splits_dir)

    full = add_keys(safe_read_csv(Path(args.csv)))

    # audit existing splits
    if args.audit_only:
        if not (args.train_csv and args.val_csv and args.test_csv):
            raise ValueError("--audit_only requires --train_csv --val_csv --test_csv")
        tr = add_keys(safe_read_csv(Path(args.train_csv)))
        va = add_keys(safe_read_csv(Path(args.val_csv)))
        te = add_keys(safe_read_csv(Path(args.test_csv)))
        rep = audit_leakage(tr, va, te)
        save_json(rep, tables / "split_leakage_audit.json")
        print("Done. Audit saved:", (tables / "split_leakage_audit.json").resolve())
        return

    # make split (and audit it)
    if args.make_split:
        df_split = make_group_split(
            full,
            split_by=args.split_by,
            ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
            seed=args.seed,
        )

        # write split CSVs
        for sp in ["train", "val", "test"]:
            sub = df_split[df_split["split"] == sp].copy()
            sub.to_csv(splits_dir / f"{sp}.csv", index=False)

        # summary + audit
        sumry = split_summary(df_split)
        sumry["split_by"] = args.split_by
        sumry["seed"] = args.seed
        save_json(sumry, tables / "split_summary.json")

        tr = df_split[df_split["split"] == "train"]
        va = df_split[df_split["split"] == "val"]
        te = df_split[df_split["split"] == "test"]
        rep = audit_leakage(tr, va, te)
        save_json(rep, tables / "split_leakage_audit.json")

        print("Done. Outputs:", out_dir.resolve())
        print("  Splits:", splits_dir.resolve())
        print("  Audit:", (tables / "split_leakage_audit.json").resolve())
        return

    raise ValueError("Please use either --make_split or --audit_only")


if __name__ == "__main__":
    main()
