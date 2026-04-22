"""Deterministic case-level train/dev/test split for Phase C.

208 cases total. Split 150/30/28 using a hash on case_id so the split is
stable across runs and machines. Writes three files:

    data/M3D_RefSeg_splits/phase_c_train.txt
    data/M3D_RefSeg_splits/phase_c_dev.txt
    data/M3D_RefSeg_splits/phase_c_test.txt

Each line is a bare case_id, e.g. "s0000".
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def hash_rank(case_id: str) -> int:
    """Stable integer derived from case_id (not Python's salted hash())."""
    return int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16], 16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy_root", default="data/M3D_RefSeg_npy")
    p.add_argument("--out_dir", default="data/M3D_RefSeg_splits")
    p.add_argument("--n_train", type=int, default=150)
    p.add_argument("--n_dev", type=int, default=30)
    p.add_argument("--n_test", type=int, default=28)
    args = p.parse_args()

    npy_root = Path(args.npy_root)
    cases = sorted(
        d.name for d in npy_root.iterdir()
        if d.is_dir() and (d / "mask.npy").exists()
    )
    n_total = args.n_train + args.n_dev + args.n_test
    if len(cases) < n_total:
        raise SystemExit(
            f"only {len(cases)} cases under {npy_root}, need {n_total}"
        )

    ordered = sorted(cases, key=hash_rank)
    train = sorted(ordered[: args.n_train])
    dev = sorted(ordered[args.n_train : args.n_train + args.n_dev])
    test = sorted(ordered[args.n_train + args.n_dev : n_total])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", train), ("dev", dev), ("test", test)]:
        path = out_dir / f"phase_c_{name}.txt"
        path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        print(f"[+] {path}  n={len(ids)}  first={ids[0]}  last={ids[-1]}")


if __name__ == "__main__":
    main()
