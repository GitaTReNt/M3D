#!/usr/bin/env python3
"""
BiomedParse v2 (BoltzFormer, 3D) zero-shot evaluation on M3D-RefSeg.

Input: data/M3D_RefSeg_biomedparse*/sXXXX.npz produced by
       scripts/biomedparse_prepare.py.  Each npz holds
         imgs           (D, H, W) float32 (scale depends on --scale_mode used in prepare)
         gts            (D, H, W) int32 multi-class map (0=bg, 1, 2, ...)
         text_prompts   {"1": "<prompt>", ...}

Runs two prompt modes:
  raw        — full English clinical sentence from text.json
  structured — regex-extracted finding-type keyword ("mass" / "nodule" / ...)

For each case, by default concatenates per-mask prompts with [SEP] and runs the
model once.  With --one_prompt_per_forward, runs N forwards per case so each
prompt stays under CLIP's 77-token limit.

Records one Dice per (case, mask_id, mode) in two CSVs.  Filenames include
--tag so multiple debug runs don't clobber each other.

Usage:
  python scripts/biomedparse_evaluate.py \
      --npz_root data/M3D_RefSeg_biomedparse \
      --max_cases 1  # smoke
"""
import argparse, gc, json, os, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label


FINDING_TYPES = [
    "lymph nodes", "lymph node",
    "cysts", "cyst",
    "masses", "mass",
    "tumors", "tumours", "tumor", "tumour",
    "nodules", "nodule",
    "calcifications", "calcification",
    "stones", "calculi", "calculus",
    "fluid collection", "effusion",
    "hematoma", "hemorrhage", "bleeding",
    "abscess",
    "opacities", "opacity", "infiltrate",
    "emphysema", "atelectasis",
    "fracture",
    "aneurysm",
    "thickening",
    "lesions", "lesion",
]
_NORMALIZE = {
    "lymph nodes": "lymph node",
    "cysts": "cyst",
    "masses": "mass",
    "tumors": "tumor", "tumours": "tumor", "tumour": "tumor",
    "nodules": "nodule",
    "calcifications": "calcification",
    "stones": "stone", "calculi": "calculus",
    "opacities": "opacity",
    "lesions": "lesion",
}


def extract_finding_type(text: str) -> str:
    tl = text.lower()
    for ft in FINDING_TYPES:
        if re.search(r"\b" + re.escape(ft) + r"\b", tl):
            return _NORMALIZE.get(ft, ft)
    return "lesion"


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if gt.sum() == 0 else 0.0
    return float(2.0 * (pred & gt).sum() / denom)


def extract_per_slice_boxes(pred_3d_bin: np.ndarray,
                             margin: int = 5,
                             largest_cc_only: bool = True):
    """Return per-slice list of [x1,y1,x2,y2] or None.

    Feeds downstream MedSAM as pseudo-box prompt (Phase B of docs/新计划.md).
    """
    D, H, W = pred_3d_bin.shape
    out = []
    for d in range(D):
        m = pred_3d_bin[d]
        if not m.any():
            out.append(None)
            continue
        if largest_cc_only:
            labeled, n = cc_label(m)
            if n > 1:
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0  # ignore background
                keep = int(sizes.argmax())
                m = (labeled == keep)
        ys, xs = np.where(m)
        x1 = max(0, int(xs.min()) - margin)
        y1 = max(0, int(ys.min()) - margin)
        x2 = min(W, int(xs.max()) + margin)
        y2 = min(H, int(ys.max()) + margin)
        out.append([x1, y1, x2, y2])
    return out


def load_model(bp_root: Path, ckpt: Path, device: torch.device):
    """Instantiate BiomedParse 3D via hydra and load the v2 checkpoint.

    BiomedParse's own code imports from 'utils' (relative), so we must put
    bp_root on sys.path AND cd into it so hydra resolves 'configs/' correctly.
    """
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    prev_cwd = os.getcwd()
    os.chdir(str(bp_root))
    sys.path.insert(0, str(bp_root))
    try:
        GlobalHydra.instance().clear()
        cfg_dir = str((bp_root / "configs").resolve())
        initialize_config_dir(config_dir=cfg_dir, job_name="biomedparse_eval",
                              version_base=None)
        cfg = compose(config_name="model/biomedparse_3D")
        # composing a sub-path wraps the config under "model:", unwrap it
        model_cfg = cfg.model if "model" in cfg else cfg
        model = hydra.utils.instantiate(model_cfg, _convert_="object")
        model.load_pretrained(str(ckpt))
        model.to(device).eval()
    finally:
        os.chdir(prev_cwd)
    return model


def run_one(model, imgs_np: np.ndarray, text: str, ids: list,
            device: torch.device, slice_batch_size: int,
            score_threshold: float, do_nms: bool) -> np.ndarray:
    """Run BiomedParse on one (case, prompt-string). Returns (D, H, W) int map.

    Pixel values: 0 (bg), or ids[j] for the j-th prompt.
    """
    # Import here so we're under bp_root's sys.path
    from utils import process_input, process_output
    from inference import postprocess, merge_multiclass_masks

    imgs_t, pad_width, padded_size, valid_axis = process_input(imgs_np, 512)
    imgs_t = imgs_t.to(device).int()

    with torch.no_grad():
        out = model({"image": imgs_t.unsqueeze(0), "text": [text]},
                    mode="eval", slice_batch_size=slice_batch_size)

    mp = out["predictions"]["pred_gmasks"]         # (N, D, h, w) logits
    mp = F.interpolate(mp, size=(512, 512), mode="bicubic",
                       align_corners=False, antialias=True)
    mp = postprocess(mp, out["predictions"]["object_existence"],
                     threshold=score_threshold, do_nms=do_nms)
    mp = merge_multiclass_masks(mp, ids)           # (D, 512, 512) int (pixel = id)
    mp = process_output(mp, pad_width, padded_size, valid_axis)

    del out, imgs_t
    return mp


def predict_all_masks(model, imgs, prompts: dict, ids: list, device, args):
    """Return (D, H, W) int map with pixel = mask_id for predicted regions.

    If --one_prompt_per_forward, runs N forwards (one per id) and merges by
    taking the first-hit id at each voxel. Otherwise concatenates with [SEP].
    """
    if args.one_prompt_per_forward:
        merged = None
        for i_ in ids:
            pred_i = run_one(model, imgs, prompts[i_], [i_], device,
                             slice_batch_size=args.slice_batch_size,
                             score_threshold=args.score_threshold,
                             do_nms=not args.no_nms)
            if merged is None:
                merged = np.zeros_like(pred_i)
            # only overwrite where merged is still background — ensures each id
            # keeps its own predicted voxels without one prompt clobbering another
            mask_i = (pred_i == i_) & (merged == 0)
            merged[mask_i] = i_
            del pred_i
            torch.cuda.empty_cache()
        return merged

    text = "[SEP]".join(prompts[i_] for i_ in ids)
    return run_one(model, imgs, text, ids, device,
                   slice_batch_size=args.slice_batch_size,
                   score_threshold=args.score_threshold,
                   do_nms=not args.no_nms)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bp_root = Path(args.bp_root).resolve()
    ckpt = Path(args.ckpt).resolve()
    npz_root = Path(args.npz_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # BiomedParse imports expect its own cwd. Add to sys.path now so we can
    # `from utils import ...` / `from inference import ...` after returning.
    sys.path.insert(0, str(bp_root))

    print(f"[+] npz_root={npz_root}", flush=True)
    print(f"[+] threshold={args.score_threshold}  do_nms={not args.no_nms}  "
          f"one_prompt_per_forward={args.one_prompt_per_forward}  "
          f"tag={args.tag!r}", flush=True)
    print(f"[+] Loading BiomedParse from {bp_root}", flush=True)
    model = load_model(bp_root, ckpt, device)
    print("[+] Model ready. VRAM:",
          f"{torch.cuda.memory_allocated()/1e9:.2f}/"
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB",
          flush=True)

    cases = sorted(p for p in npz_root.glob("s*.npz"))
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    # Report input value range on first case — crucial for debugging scale issues
    first = np.load(cases[0], allow_pickle=True)
    im0 = first["imgs"]
    print(f"[+] first case {cases[0].stem} imgs: dtype={im0.dtype}  "
          f"range=[{im0.min():.2f}, {im0.max():.2f}]  mean={im0.mean():.2f}",
          flush=True)

    raw_rows, struct_rows = [], []
    pseudoboxes = {}  # case_id -> mask_id -> mode -> {prompt, boxes, gt_pos, pred_pos}
    t0 = time.time()

    for i, npz_path in enumerate(cases, 1):
        case = npz_path.stem
        data = np.load(npz_path, allow_pickle=True)
        imgs = data["imgs"]
        gts = data["gts"]
        raw_prompts = data["text_prompts"].item()
        ids = sorted(int(k) for k in raw_prompts.keys() if k != "instance_label")

        for mode_name, extractor, rows in [
            ("raw",        lambda s: s,              raw_rows),
            ("structured", extract_finding_type,     struct_rows),
        ]:
            prompts = {i_: extractor(raw_prompts[str(i_)]) for i_ in ids}

            try:
                pred = predict_all_masks(model, imgs, prompts, ids, device, args)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[{case}/{mode_name}] OOM — retrying w/ slice_batch_size=1",
                      flush=True)
                args.slice_batch_size = 1
                pred = predict_all_masks(model, imgs, prompts, ids, device, args)

            for mid in ids:
                gt_bin = (gts == mid)
                pred_bin = (pred == mid)
                d = dice_score(pred_bin, gt_bin)
                rows.append(dict(
                    case=case, mask_id=mid, mode=mode_name,
                    prompt=prompts[mid], dice=d,
                    gt_pos=int(gt_bin.sum()), pred_pos=int(pred_bin.sum()),
                ))
                if args.save_pseudoboxes:
                    boxes = extract_per_slice_boxes(
                        pred_bin, margin=args.bbox_margin,
                        largest_cc_only=args.largest_cc_only)
                    pseudoboxes.setdefault(case, {}).setdefault(str(mid), {})[mode_name] = {
                        "prompt": prompts[mid],
                        "boxes": boxes,
                        "gt_pos": int(gt_bin.sum()),
                        "pred_pos": int(pred_bin.sum()),
                    }

            del pred
            gc.collect()
            torch.cuda.empty_cache()

        if i % 5 == 0 or i == len(cases):
            elapsed = time.time() - t0
            raw_so_far = [r["dice"] for r in raw_rows]
            st_so_far = [r["dice"] for r in struct_rows]
            raw_empty = sum(1 for r in raw_rows if r["pred_pos"] == 0)
            st_empty = sum(1 for r in struct_rows if r["pred_pos"] == 0)
            print(f"[{i}/{len(cases)}] {case}  "
                  f"raw_mean={np.mean(raw_so_far):.4f} (empty {raw_empty}/{len(raw_rows)})  "
                  f"struct_mean={np.mean(st_so_far):.4f} (empty {st_empty}/{len(struct_rows)})  "
                  f"elapsed={elapsed/60:.1f}m", flush=True)

    # Save CSVs — tag suffix lets multiple debug runs coexist
    suffix = f"_{args.tag}" if args.tag else ""
    raw_csv = out_dir / f"12_biomedparse_v2_raw{suffix}.csv"
    struct_csv = out_dir / f"12_biomedparse_v2_structured{suffix}.csv"
    pd.DataFrame(raw_rows).to_csv(raw_csv, index=False)
    pd.DataFrame(struct_rows).to_csv(struct_csv, index=False)

    if args.save_pseudoboxes:
        pb_path = out_dir / f"12_biomedparse_pseudoboxes{suffix}.json"
        with open(pb_path, "w", encoding="utf-8") as f:
            json.dump(pseudoboxes, f)
        print(f"[+] pseudoboxes saved -> {pb_path}")

    for name, rows, csv_path in [
        ("raw", raw_rows, raw_csv),
        ("structured", struct_rows, struct_csv),
    ]:
        dices = np.array([r["dice"] for r in rows])
        n_empty = sum(1 for r in rows if r["pred_pos"] == 0)
        n_hit = int((dices >= 0.5).sum())
        print(f"  [{name}] n={len(dices)}  mean={dices.mean():.4f}  "
              f"median={np.median(dices):.4f}  Dice>=0.5: {n_hit}/{len(dices)} "
              f"({100*n_hit/max(1,len(dices)):.1f}%)  "
              f"empty_pred: {n_empty}/{len(dices)} -> {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz_root", default="data/M3D_RefSeg_biomedparse")
    p.add_argument("--bp_root", default="third_party/BiomedParse")
    p.add_argument("--ckpt",
                   default="third_party/BiomedParse/model_weights/biomedparse_v2.ckpt")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--max_cases", type=int, default=0, help="0 = all")
    p.add_argument("--slice_batch_size", type=int, default=4,
                   help="slices processed per forward; drop to 1-2 if OOM")
    # Debug knobs for the three hypotheses in docs/biomedparse_setup_report.md
    p.add_argument("--score_threshold", type=float, default=0.5,
                   help="object_existence sigmoid threshold (Hypothesis C: lower to 0.1)")
    p.add_argument("--no_nms", action="store_true",
                   help="disable per-slice NMS (Hypothesis C)")
    p.add_argument("--one_prompt_per_forward", action="store_true",
                   help="run one forward per mask id (Hypothesis B: avoid "
                        "CLIP 77-token truncation from [SEP]-joined prompts)")
    p.add_argument("--tag", default="",
                   help="suffix for output CSV names, e.g. 'hu' or 'thr0.1'")
    # Phase B: dump pseudo-boxes for downstream MedSAM refinement
    p.add_argument("--save_pseudoboxes", action="store_true",
                   help="dump per-slice [x1,y1,x2,y2] bboxes to JSON "
                        "(consumed by scripts/inference_medsam_from_pseudoboxes.py)")
    p.add_argument("--bbox_margin", type=int, default=5,
                   help="pixels to expand each pseudo-box (coordinates in 256x256 space)")
    p.add_argument("--largest_cc_only", action="store_true", default=True,
                   help="keep only the largest connected component per slice before bbox")
    args = p.parse_args()
    evaluate(args)
