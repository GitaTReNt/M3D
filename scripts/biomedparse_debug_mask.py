#!/usr/bin/env python3
"""Diagnostic: inspect pred_gmasks spatial distribution, bypassing object_existence.

If existence is low but mask logits still peak near the GT, we can salvage
predictions by using pred_gmasks.sigmoid() > 0.5 directly (ignoring existence).
"""
import argparse, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_model(bp_root, ckpt, device):
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    prev_cwd = os.getcwd()
    os.chdir(str(bp_root))
    sys.path.insert(0, str(bp_root))
    try:
        GlobalHydra.instance().clear()
        cfg_dir = str((bp_root / "configs").resolve())
        initialize_config_dir(config_dir=cfg_dir, job_name="bp_debug_mask",
                              version_base=None)
        cfg = compose(config_name="model/biomedparse_3D")
        model_cfg = cfg.model if "model" in cfg else cfg
        model = hydra.utils.instantiate(model_cfg, _convert_="object")
        model.load_pretrained(str(ckpt))
        model.to(device).eval()
    finally:
        os.chdir(prev_cwd)
    return model


def dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    s = pred.sum() + gt.sum()
    return float(2 * (pred & gt).sum() / s) if s > 0 else 0.0


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

    from utils import process_input, process_output

    data = np.load(args.npz, allow_pickle=True)
    imgs_np = data["imgs"]
    text_prompts = data["text_prompts"].item()
    gts = data["gts"]
    ids = sorted(int(k) for k in text_prompts.keys() if k != "instance_label")

    imgs_t, pad_width, padded_size, valid_axis = process_input(imgs_np, 512)
    imgs_t = imgs_t.to(device).int()

    prompts = []
    for i_ in ids:
        raw = text_prompts[str(i_)]
        for kw in ["lymph node", "mass", "nodule", "cyst", "tumor", "lesion"]:
            if kw in raw.lower():
                prompts.append((i_, kw))
                break
        else:
            prompts.append((i_, "lesion"))

    print(f"{'id':3} {'prompt':15} | "
          f"{'mask logit max':14} {'sig max':8} {'>0.5':6} {'>0.3':6} "
          f"| direct-mask dice @ sig>0.5 / 0.3 / 0.1")
    print("-" * 120)

    with torch.no_grad():
        for i_, text in prompts:
            out = model({"image": imgs_t.unsqueeze(0), "text": [text]},
                        mode="eval", slice_batch_size=args.slice_batch_size)
            mp = out["predictions"]["pred_gmasks"]           # (1, D, h, w)
            oe = out["predictions"]["object_existence"]      # (1, D)
            mp_up = F.interpolate(mp, size=(512, 512), mode="bicubic",
                                  align_corners=False, antialias=True)
            sig = mp_up.sigmoid()
            logit_max = mp_up.max().item()
            sig_max = sig.max().item()
            n_gt05 = (sig > 0.5).sum().item()
            n_gt03 = (sig > 0.3).sum().item()

            # Direct mask at various thresholds, ignoring existence
            gt_bin = (gts == i_)
            dices = []
            for thr in [0.5, 0.3, 0.1]:
                pred = (sig > thr).squeeze(0).int().cpu()
                pred_3d = process_output(pred, pad_width, padded_size, valid_axis)
                if isinstance(pred_3d, torch.Tensor):
                    pred_3d = pred_3d.numpy()
                dices.append(dice(pred_3d, gt_bin))

            print(f"{i_:3} {text:15} | {logit_max:14.4f} {sig_max:8.4f} "
                  f"{n_gt05:6} {n_gt03:6} "
                  f"| {dices[0]:.4f} / {dices[1]:.4f} / {dices[2]:.4f}  "
                  f"(gt_pos={gt_bin.sum()})", flush=True)


if __name__ == "__main__":
    main()
