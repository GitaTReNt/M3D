#!/usr/bin/env python3
"""
Phase B 3-way analysis: merge BiomedParse direct / BP->MedSAM / GT->MedSAM.

Answers 新计划.md's decision point: is BiomedParse's coarse grounding already
usable for MedSAM refinement, or is the main bottleneck still text→space?

Inputs:
  --bp_direct_csv   results/12_biomedparse_v2_raw<tag>.csv
  --bp_medsam_csv   results/12_biomedparse_medsam_raw<tag>.csv
  --gt_medsam_csv   results/12_medsam_oracle_full/medsam_oracle_bbox_results.csv

Outputs per (mode=raw|structured):
  - merged CSV with dice_bp_direct / dice_bp_medsam / dice_gt_medsam per (case, mask_id)
  - console table: mean / median / Dice>=0.5 for each pipeline
  - paired analysis: #cases where bp_medsam > bp_direct, etc.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_bp_csv(path: Path, pipeline_tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = ["case", "mask_id", "mode", "dice"]
    df = df[keep].rename(columns={"dice": f"dice_{pipeline_tag}"})
    return df


def load_gt_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"case_id": "case"})
    return df[["case", "mask_id", "dice"]].rename(
        columns={"dice": "dice_gt_medsam"}
    )


def summarize(df: pd.DataFrame, col: str) -> dict:
    d = df[col].dropna().values
    if len(d) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, hit=0, empty=0)
    return dict(
        n=len(d),
        mean=float(d.mean()),
        median=float(np.median(d)),
        hit=int((d >= 0.5).sum()),
        empty=int((d == 0.0).sum()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bp_direct_raw",
                   default="results/12_biomedparse_v2_raw_full_bypass.csv")
    p.add_argument("--bp_direct_struct",
                   default="results/12_biomedparse_v2_structured_full_bypass.csv")
    p.add_argument("--bp_medsam_raw",
                   default="results/12_biomedparse_medsam_raw_full.csv")
    p.add_argument("--bp_medsam_struct",
                   default="results/12_biomedparse_medsam_structured_full.csv")
    p.add_argument("--gt_medsam",
                   default="results/12_medsam_oracle_full/medsam_oracle_bbox_results.csv")
    p.add_argument("--out_dir", default="results/12_phase_b_analysis")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for path_arg in ("bp_direct_raw", "bp_direct_struct",
                     "bp_medsam_raw", "bp_medsam_struct"):
        pth = Path(getattr(args, path_arg))
        if not pth.exists():
            missing.append(str(pth))
    if missing:
        print("[!] missing inputs:", *missing, sep="\n  ")
        sys.exit(1)

    gt_path = Path(args.gt_medsam)
    if gt_path.exists():
        gt_df = load_gt_csv(gt_path)
        print(f"[+] oracle (GT box -> MedSAM): {len(gt_df)} rows")
    else:
        gt_df = pd.DataFrame(columns=["case", "mask_id", "dice_gt_medsam"])
        print(f"[!] oracle CSV not found ({gt_path}); skipping GT-box column")

    for mode in ("raw", "structured"):
        direct_p = Path(getattr(args, f"bp_direct_{mode[:6]}"))
        medsam_p = Path(getattr(args, f"bp_medsam_{mode[:6]}"))

        direct = load_bp_csv(direct_p, "bp_direct")
        direct = direct[direct["mode"] == mode][
            ["case", "mask_id", "dice_bp_direct"]
        ]
        medsam = load_bp_csv(medsam_p, "bp_medsam")
        medsam = medsam[medsam["mode"] == mode][
            ["case", "mask_id", "dice_bp_medsam"]
        ]

        m = direct.merge(medsam, on=["case", "mask_id"], how="outer")
        m = m.merge(gt_df, on=["case", "mask_id"], how="left")

        out_csv = out_dir / f"phase_b_merged_{mode}.csv"
        m.to_csv(out_csv, index=False)

        print(f"\n=== mode={mode} (n={len(m)}) -> {out_csv} ===")
        header = f"{'pipeline':22s} {'n':>4s} {'mean':>7s} {'med':>7s} {'>=0.5':>6s} {'==0':>5s}"
        print(header)
        print("-" * len(header))
        for col, tag in [("dice_bp_direct", "bp direct"),
                         ("dice_bp_medsam", "bp box -> medsam"),
                         ("dice_gt_medsam", "gt box -> medsam")]:
            s = summarize(m, col)
            print(f"{tag:22s} {s['n']:4d} {s['mean']:7.4f} {s['median']:7.4f} "
                  f"{s['hit']:6d} {s['empty']:5d}")

        # Paired comparisons on cases with both numbers
        both = m.dropna(subset=["dice_bp_direct", "dice_bp_medsam"])
        if len(both) > 0:
            delta = both["dice_bp_medsam"] - both["dice_bp_direct"]
            print(f"\n  paired (n={len(both)}): bp_medsam vs bp_direct")
            print(f"    medsam > direct:  {int((delta > 0).sum())}")
            print(f"    medsam == direct: {int((delta == 0).sum())}")
            print(f"    medsam < direct:  {int((delta < 0).sum())}")
            print(f"    mean delta: {float(delta.mean()):+.4f}")

        both_gt = m.dropna(subset=["dice_bp_medsam", "dice_gt_medsam"])
        if len(both_gt) > 0:
            ratio = (both_gt["dice_bp_medsam"] / both_gt["dice_gt_medsam"]
                     .replace(0, np.nan))
            print(f"\n  ceiling retention (bp_medsam / gt_medsam, excl. "
                  f"gt_medsam==0): mean={float(ratio.mean()):.3f}  "
                  f"median={float(ratio.median()):.3f}")


if __name__ == "__main__":
    main()
