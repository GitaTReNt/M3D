"""Convert M3D-RefSeg preprocessed .npy cases to BiomedParse .npz format.

Input per case (from data/M3D_RefSeg_npy/sXXXX/):
  ct.npy    (1, 32, 256, 256) float32, normalized to [0, 1]
  mask.npy  (1, 32, 256, 256) float32, multi-class map (0=bg, 1, 2, ...)
  text.json {"1": "...", "2": "..."}  # 1-based mask ids

Output per case (written to out_root/sXXXX.npz):
  imgs          (32, 256, 256) float32 — scaled per --scale_mode
  gts           (32, 256, 256) int32   — same multi-class map
  text_prompts  dict matching text.json, saved as object array

Scale modes (test Hypothesis A from docs/biomedparse_setup_report.md):
  imagenet255  ct[0,1] -> [0,255]. Matches BiomedParse pixel_mean~114/std~58. Default.
  hu           ct[0,1] -> [-1000, 1000], reversing M3D's clip(HU, -1000, 1000)/2000+0.5.
  raw01        ct[0,1] as-is.
"""
import argparse, json
from pathlib import Path
import numpy as np


SCALE_MODES = {
    "imagenet255": lambda ct: (ct * 255.0).clip(0, 255),
    "hu":          lambda ct: (ct * 2000.0 - 1000.0),
    "raw01":       lambda ct: ct,
}


def convert_case(case_dir: Path, out_path: Path, scale_mode: str) -> None:
    ct = np.load(case_dir / "ct.npy")
    mask = np.load(case_dir / "mask.npy")
    with open(case_dir / "text.json", "r", encoding="utf-8") as f:
        text_prompts = json.load(f)

    # Keep float32 because BiomedParse's process_input calls F.interpolate(mode="bicubic"),
    # which is not implemented for integer tensors. The .int() cast happens after resize.
    imgs = SCALE_MODES[scale_mode](ct[0]).astype(np.float32)
    gts = mask[0].astype(np.int32)
    text_prompts = {str(k): str(v) for k, v in text_prompts.items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        imgs=imgs,
        gts=gts,
        text_prompts=np.array(text_prompts, dtype=object),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npy_root", default="data/M3D_RefSeg_npy")
    p.add_argument("--out_root", default="data/M3D_RefSeg_biomedparse")
    p.add_argument("--scale_mode", choices=list(SCALE_MODES.keys()),
                   default="imagenet255",
                   help="how to rescale ct values before saving; see module docstring")
    p.add_argument("--out_suffix", default="",
                   help="append to out_root so multiple scale modes can coexist "
                        "(e.g. '_hu' -> data/M3D_RefSeg_biomedparse_hu/)")
    p.add_argument("--max_cases", type=int, default=0, help="0 = all cases")
    args = p.parse_args()

    npy_root = Path(args.npy_root)
    out_root = Path(args.out_root + args.out_suffix)
    cases = sorted(d for d in npy_root.iterdir() if d.is_dir() and d.name.startswith("s"))
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    print(f"[+] scale_mode={args.scale_mode}  out_root={out_root}  n={len(cases)}",
          flush=True)

    for i, case_dir in enumerate(cases, 1):
        out_path = out_root / f"{case_dir.name}.npz"
        convert_case(case_dir, out_path, args.scale_mode)
        if i % 20 == 0 or i == len(cases):
            print(f"[{i}/{len(cases)}] {case_dir.name} -> {out_path}", flush=True)

    # Quick sanity: load first one back and report value range
    first = np.load(out_root / f"{cases[0].name}.npz", allow_pickle=True)
    im = first["imgs"]
    print(f"Done. imgs dtype={im.dtype}  shape={im.shape}  "
          f"range=[{im.min():.2f}, {im.max():.2f}]  mean={im.mean():.2f}")


if __name__ == "__main__":
    main()
