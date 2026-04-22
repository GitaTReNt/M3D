"""Derive auxiliary supervision labels from GT masks for Phase C.

For each (case, mask_id) we precompute:
    existence       int, mask has any positive voxel in the 3D case
    slice_exist     list[int] length D, per-slice existence
    bbox_3d         [z1,y1,x1,z2,y2,x2] normalised to [0,1]
    centroid_3d     [cz,cy,cx] normalised to [0,1]
    z_range         [z_min, z_max] normalised to [0,1]
    volume          voxels / (D*H*W)
    component_count scipy.ndimage.label on the 3D mask (connected components)

Writes JSON per case to data/M3D_RefSeg_aux/<case>.json:

    {
      "1": {"existence": 1, "slice_exist": [...D...],
            "bbox_3d": [...6...], "centroid_3d": [...3...],
            "z_range": [...2...], "volume": f, "component_count": k},
      "2": {...}
    }

Consumed by `src/datasets/refseg_phase_c.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def mask_to_aux(mask_vol: np.ndarray, mask_id: int) -> dict:
    """mask_vol: (D, H, W) uint/int where pixel == mask_id flags that mask."""
    m = (mask_vol == mask_id).astype(np.uint8)
    D, H, W = m.shape
    existence = int(m.any())
    slice_exist = [int(m[z].any()) for z in range(D)]

    if existence:
        zs, ys, xs = np.where(m)
        z1, z2 = int(zs.min()), int(zs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        bbox_3d = [
            z1 / D, y1 / H, x1 / W,
            (z2 + 1) / D, (y2 + 1) / H, (x2 + 1) / W,
        ]
        cz = float(zs.mean()) / D
        cy = float(ys.mean()) / H
        cx = float(xs.mean()) / W
        centroid_3d = [cz, cy, cx]
        z_range = [z1 / D, (z2 + 1) / D]
        volume = float(m.sum()) / (D * H * W)
        _, n_cc = ndi.label(m)
        component_count = int(n_cc)
    else:
        bbox_3d = [0.0] * 6
        centroid_3d = [0.0] * 3
        z_range = [0.0, 0.0]
        volume = 0.0
        component_count = 0

    return {
        "existence": existence,
        "slice_exist": slice_exist,
        "bbox_3d": bbox_3d,
        "centroid_3d": centroid_3d,
        "z_range": z_range,
        "volume": volume,
        "component_count": component_count,
    }


def process_case(case_dir: Path, out_dir: Path):
    mask = np.load(case_dir / "mask.npy")
    # mask layout: (1, D, H, W) float32 with 0/1/2... integer mask ids
    if mask.ndim == 4:
        mask = mask[0]
    mask = mask.astype(np.int32)
    text_path = case_dir / "text.json"
    with open(text_path, "r", encoding="utf-8") as f:
        text = json.load(f)

    aux = {}
    for mid_str in text.keys():
        mid = int(mid_str)
        aux[mid_str] = mask_to_aux(mask, mid)

    out_path = out_dir / f"{case_dir.name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aux, f)
    return aux


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy_root", default="data/M3D_RefSeg_npy")
    p.add_argument("--out_dir", default="data/M3D_RefSeg_aux")
    p.add_argument("--cases_list", default="",
                   help="optional txt of case_ids to process; default = all")
    args = p.parse_args()

    npy_root = Path(args.npy_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cases_list:
        with open(args.cases_list, "r", encoding="utf-8") as f:
            cases = [ln.strip() for ln in f if ln.strip()]
    else:
        cases = sorted(
            d.name for d in npy_root.iterdir()
            if d.is_dir() and (d / "mask.npy").exists()
        )

    n_empty = 0
    for i, cid in enumerate(cases):
        aux = process_case(npy_root / cid, out_dir)
        for mid_str, a in aux.items():
            if a["existence"] == 0:
                n_empty += 1
        if (i + 1) % 20 == 0 or i == len(cases) - 1:
            print(f"[{i + 1}/{len(cases)}] {cid} -> {out_dir / (cid + '.json')}",
                  flush=True)
    print(f"[done] {len(cases)} cases; empty_masks={n_empty}")


if __name__ == "__main__":
    main()
