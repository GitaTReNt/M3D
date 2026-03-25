#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EDA: Adjacent-slice redundancy for RefSeg (ct.npy in each case)

Compute similarity between adjacent z-slices (z=32 after prepare):
- metric=corr (default): fast, no extra dependency
- metric=ssim: optional, requires scikit-image

Outputs:
- tables/slice_redundancy_per_case.csv
- tables/slice_redundancy_all_pairs.csv
- figs/hist_case_mean.png
- figs/hist_all_pairs.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def corr2d(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Pearson correlation of two 2D arrays."""
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(a.std()) * float(b.std()) + eps
    return float((a * b).mean() / denom)


def ssim2d(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM of two 2D arrays. Requires scikit-image."""
    from skimage.metrics import structural_similarity as ssim

    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    if data_range <= 0:
        data_range = 1.0
    return float(ssim(a, b, data_range=data_range))


def load_ct(ct_path: Path) -> np.ndarray:
    """Load ct.npy -> (Z,H,W) float32."""
    vol = np.load(ct_path)
    if vol.ndim == 4 and vol.shape[0] == 1:
        vol = vol[0]
    if vol.ndim != 3:
        raise ValueError(f"Unexpected ct.npy shape: {vol.shape} at {ct_path}")
    return vol.astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser("Adjacent slice redundancy EDA for RefSeg")
    ap.add_argument("--npy_root", required=True, help="Root dir of M3D_RefSeg_npy (contains sXXXX folders)")
    ap.add_argument("--out_dir", default="./slice_redundancy_out")
    ap.add_argument("--metric", choices=["corr", "ssim"], default="corr",
                    help="Similarity metric between adjacent slices")
    ap.add_argument("--stride_xy", type=int, default=1,
                    help="Downsample in-plane by stride for faster compute (1 keeps original)")
    ap.add_argument("--max_cases", type=int, default=-1,
                    help="Process at most N cases (for quick debug). -1 = all.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    npy_root = Path(args.npy_root)
    out_dir = Path(args.out_dir)
    tables = out_dir / "tables"
    figs = out_dir / "figs"
    ensure_dir(tables)
    ensure_dir(figs)

    if args.metric == "ssim":
        try:
            import skimage  # noqa: F401
        except Exception as e:
            raise RuntimeError("metric=ssim requires scikit-image. Install: pip install scikit-image") from e

    # collect case dirs
    case_dirs = sorted([p for p in npy_root.iterdir() if p.is_dir()])
    if args.max_cases and args.max_cases > 0:
        rng = np.random.default_rng(args.seed)
        if len(case_dirs) > args.max_cases:
            idx = rng.choice(len(case_dirs), size=args.max_cases, replace=False)
            case_dirs = [case_dirs[i] for i in sorted(idx)]

    sim_fn = corr2d if args.metric == "corr" else ssim2d

    per_case_rows = []
    pair_rows = []

    for cdir in tqdm(case_dirs, desc=f"Computing {args.metric} redundancy"):
        case_id = cdir.name
        ct_path = cdir / "ct.npy"
        if not ct_path.exists():
            continue

        vol = load_ct(ct_path)  # (Z,H,W)
        if args.stride_xy > 1:
            vol = vol[:, ::args.stride_xy, ::args.stride_xy]

        Z = vol.shape[0]
        sims = []
        for z in range(Z - 1):
            s = sim_fn(vol[z], vol[z + 1])
            sims.append(s)
            pair_rows.append({"case_id": case_id, "z": z, "z_next": z + 1, "sim": s})

        sims = np.array(sims, dtype=np.float32)
        per_case_rows.append({
            "case_id": case_id,
            "Z": int(Z),
            "metric": args.metric,
            "stride_xy": int(args.stride_xy),
            "mean_sim": float(np.mean(sims)) if sims.size else np.nan,
            "median_sim": float(np.median(sims)) if sims.size else np.nan,
            "p10_sim": float(np.percentile(sims, 10)) if sims.size else np.nan,
            "p90_sim": float(np.percentile(sims, 90)) if sims.size else np.nan,
            "min_sim": float(np.min(sims)) if sims.size else np.nan,
            "max_sim": float(np.max(sims)) if sims.size else np.nan,
        })

    df_case = pd.DataFrame(per_case_rows).sort_values("mean_sim", ascending=False)
    df_pair = pd.DataFrame(pair_rows)

    df_case.to_csv(tables / "slice_redundancy_per_case.csv", index=False)
    df_pair.to_csv(tables / "slice_redundancy_all_pairs.csv", index=False)

    # plots (matplotlib only)
    import matplotlib.pyplot as plt

    # per-case mean hist
    plt.figure(figsize=(7, 4.6))
    x = df_case["mean_sim"].to_numpy()
    x = x[np.isfinite(x)]
    plt.hist(x, bins=40)
    plt.title(f"Per-case mean adjacent-slice {args.metric}")
    plt.xlabel("mean_sim")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figs / "hist_case_mean.png", dpi=220)
    plt.close()

    # all pairs hist
    plt.figure(figsize=(7, 4.6))
    y = df_pair["sim"].to_numpy()
    y = y[np.isfinite(y)]
    plt.hist(y, bins=60)
    plt.title(f"All adjacent-slice pairs {args.metric}")
    plt.xlabel("sim")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(figs / "hist_all_pairs.png", dpi=220)
    plt.close()

    print("Done. Outputs:", out_dir.resolve())


if __name__ == "__main__":
    main()
