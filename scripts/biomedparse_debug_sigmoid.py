#!/usr/bin/env python3
"""Diagnostic: dump BiomedParse's object_existence sigmoid distribution.

Doesn't call postprocess — just prints the raw confidence scores per query,
per slice, so we can see whether the model is producing any signal at all
(hidden below threshold) or genuinely dead.
"""
import argparse, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_model(bp_root: Path, ckpt: Path, device):
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    prev_cwd = os.getcwd()
    os.chdir(str(bp_root))
    sys.path.insert(0, str(bp_root))
    try:
        GlobalHydra.instance().clear()
        cfg_dir = str((bp_root / "configs").resolve())
        initialize_config_dir(config_dir=cfg_dir, job_name="bp_debug",
                              version_base=None)
        cfg = compose(config_name="model/biomedparse_3D")
        model_cfg = cfg.model if "model" in cfg else cfg
        model = hydra.utils.instantiate(model_cfg, _convert_="object")
        model.load_pretrained(str(ckpt))
        model.to(device).eval()
    finally:
        os.chdir(prev_cwd)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="data/M3D_RefSeg_biomedparse/s0000.npz")
    p.add_argument("--bp_root", default="third_party/BiomedParse")
    p.add_argument("--ckpt",
                   default="third_party/BiomedParse/model_weights/biomedparse_v2.ckpt")
    p.add_argument("--slice_batch_size", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda")
    bp_root = Path(args.bp_root).resolve()
    sys.path.insert(0, str(bp_root))

    model = load_model(bp_root, Path(args.ckpt).resolve(), device)
    print("[+] model loaded", flush=True)

    data = np.load(args.npz, allow_pickle=True)
    imgs_np = data["imgs"]
    text_prompts = data["text_prompts"].item()
    gts = data["gts"]
    ids = sorted(int(k) for k in text_prompts.keys() if k != "instance_label")

    from utils import process_input

    # Try several prompt formats and report sigmoid distribution for each
    prompt_variants = []
    for i_ in ids:
        raw = text_prompts[str(i_)]
        prompt_variants.append(("raw_full",      i_, raw))
        # short keyword heuristic
        for kw in ["mass", "lymph node", "nodule", "cyst", "tumor", "lesion"]:
            if kw in raw.lower():
                prompt_variants.append(("short_kw", i_, kw))
                break
        # first sentence only
        first_sent = raw.split(".")[0].strip() + "."
        prompt_variants.append(("first_sent", i_, first_sent))

    imgs_t, pad_width, padded_size, valid_axis = process_input(imgs_np, 512)
    imgs_t = imgs_t.to(device).int()
    print(f"[+] imgs_t shape={imgs_t.shape} dtype={imgs_t.dtype}  "
          f"range=[{imgs_t.min().item()},{imgs_t.max().item()}]", flush=True)

    print(f"{'variant':12} {'id':3} {'prompt':45} | "
          f"existence sigmoid: min/mean/max  (> 0.5: n_slices)")
    print("-" * 110)
    with torch.no_grad():
        for variant, i_, text in prompt_variants:
            out = model({"image": imgs_t.unsqueeze(0), "text": [text]},
                        mode="eval", slice_batch_size=args.slice_batch_size)
            oe = out["predictions"]["object_existence"]  # (N, D)
            sig = oe.sigmoid()
            n_high = (sig > 0.5).sum().item()
            n_mid = ((sig > 0.1) & (sig <= 0.5)).sum().item()
            gt_pos = int((gts == i_).sum())
            text_short = text[:42] + "..." if len(text) > 42 else text
            print(f"{variant:12} {i_:3} {text_short:45} | "
                  f"{sig.min().item():.4f}/{sig.mean().item():.4f}/{sig.max().item():.4f}  "
                  f"(>0.5: {n_high}, 0.1-0.5: {n_mid})  gt_pos={gt_pos}", flush=True)


if __name__ == "__main__":
    main()
