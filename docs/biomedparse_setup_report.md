# BiomedParse v2 setup — work log (stopped 2026-04-18)

Ran out of time on a smoke-test debug. Pipeline end-to-end works; predictions
come back empty — probably input-scale or prompt-format mismatch. Pick up from
"Next steps" below.

## What's done

### 1. Conda env `biomedparse_v2`
`C:\Users\63485\miniconda3\envs\biomedparse_v2\python.exe` (Python 3.10.14)

Why a separate env: BiomedParse pins torch 2.6.0+cu124, our main `PBAI` env is
on 2.8+cu126 which MedSAM/VoxTell depend on.

Installed:
- torch==2.6.0+cu124 (verified `torch.cuda.is_available()`)
- transformers 4.40.0, hydra-core, lightning 2.3.0, kornia, einops
- SimpleITK, nibabel
- opencv-python 4.9.0.80 (pinned < 4.10 to keep numpy 1.26.4 ABI)
- numpy 1.26.4 (restored after opencv yanked it to 2.2)
- packaging 24.2 (restored after `pip install ninja` bumped it past lightning's ceiling)

Skipped on purpose:
- `deepspeed==0.14.2` — build fails on Windows; only needed for training.
- `azure-ai-ml`, `azureml-acft-image-components` — training-only.

### 2. detectron2 — python-only in-place install
At `third_party/detectron2/`, cloned from upstream.
- Patched `setup.py`: replaced `ext_modules=get_extensions()` + `cmdclass` with
  `ext_modules=[]` — skips the `detectron2._C` C++/CUDA build (requires MSVC,
  which we don't have).
- Installed with `pip install --no-build-isolation -e third_party/detectron2`.
- Verified BiomedParse's imports all work: `Conv2d`, `ShapeSpec`, `get_norm`,
  `Backbone`, `SEM_SEG_HEADS_REGISTRY` are pure-Python and load fine.
- `DeformConv` imports (lazy resolution) but would crash if *called* — BiomedParse
  imports it in `transformer_encoder_fpn.py:13` but never invokes it. Safe.

### 3. Weights
Gated HF repo `microsoft/BiomedParse`, user provided a read token (trentss).
Downloaded to `third_party/BiomedParse/model_weights/`:
- `biomedparse_v2.ckpt` — 4.46 GB, the 3D BoltzFormer model
- `config.yaml`, `target_dist.json`

Note: the first download went to `I:/i/M3D/...` because I passed `/i/M3D/...`
as a POSIX-style absolute path and hf_hub_download interpreted the leading `/i/`
as a subdirectory of drive `I:`. Moved to the correct location, stray dir
deleted. If you re-download via the script, use Windows-style paths.

### 4. Model import & checkpoint load — working
From `scripts/biomedparse_evaluate.py`:
- `os.chdir(bp_root)` + `sys.path.insert(0, bp_root)` so BiomedParse's
  relative imports (`from utils import ...`, `from inference import ...`) resolve.
- `initialize_config_dir(config_dir=<abs path to configs>)` (instead of
  `initialize(config_path=...)` — the latter is relative to the *script* file).
- `compose(config_name="model/biomedparse_3D")` — the `inference.py` example
  uses `config_name="biomedparse_3D"` but that file only exists under `model/`.
- When composing a sub-path config, hydra wraps the result under a key matching
  the path (`cfg.model.*`). Unwrap before `hydra.utils.instantiate(cfg.model, ...)`.
- `model.load_pretrained(str(ckpt))` → "Checkpoint loaded successfully!"
- VRAM after load: **1.49 / 12.88 GB** — lots of headroom.

### 5. Scripts written
- `scripts/biomedparse_prepare.py` — converts `data/M3D_RefSeg_npy/sXXXX/`
  (ct.npy + mask.npy + text.json) → `data/M3D_RefSeg_biomedparse/sXXXX.npz`
  with `imgs` (D,H,W) float32 [0,255], `gts` int32 multi-class map,
  `text_prompts` dict. Done for 3 cases during smoke test; rerun with
  `--max_cases 0` for all 208.
- `scripts/biomedparse_evaluate.py` — runs two prompt modes (raw / structured)
  per case, saves `results/12_biomedparse_v2_{raw,structured}.csv`.

## Current blocker — predictions are empty

Smoke test on s0000 (2 masks):

```
mask_id=1  prompt="mass"       dice=0.0  gt_pos=9913  pred_pos=0
mask_id=2  prompt="lymph node" dice=0.0  gt_pos=461   pred_pos=0
```

Model runs without error, just emits zero positives — `object_existence.sigmoid()`
is below 0.5 for all slices. Three candidate root causes:

### Hypothesis A — input scale wrong (most likely)
Our adapter maps `ct.npy` (already normalized to [0,1]) → `x * 255` to land in
the [0,255] range that the ImageNet-style `pixel_mean=[123.675,...]` /
`pixel_std=[58.395,...]` implies.

But the original M3D preprocessing was probably `clip(HU, -1000, 1000) / 2000 + 0.5`
→ our [0,1]. BiomedParse v2, if trained on raw HU (int16, [-1024, 3071]), expects
that range, and the ImageNet-style mean/std is just a historical artifact.

**Try next**: feed reverse-scaled HU-like values instead of [0,255]:
```python
imgs = (ct[0] * 2000.0 - 1000.0).astype(np.float32)  # ~HU range
```
or even the raw CT values from `.nii.gz` via nibabel (bypassing the
preprocessed npy). If that fixes `pred_pos`, set a flag in `biomedparse_prepare.py`.

### Hypothesis B — prompt concatenation mis-format
`inference.py` does `"[SEP]".join(prompts)` as a single string. I mirrored that.
But BiomedParse v2 uses CLIP text encoder (`openai/clip-vit-base-patch32`,
downloaded to HF cache during model init) — CLIP tokenizer has **max 77 tokens**.
Two 40-token clinical sentences concatenated with `[SEP]` would get truncated
past the second prompt.

**Try next**: run with a *single* prompt per forward (run_one called once per
mask_id). Slower (N× more forwards per case) but isolates the issue. If per-mask
dice jumps when isolated, the truncation is the culprit — then rethink batching.

### Hypothesis C — threshold / NMS too aggressive
`inference.postprocess` uses `score_threshold=0.5` for NMS. Some models
produce logits that sigmoid low even for correct predictions.

**Try next**: lower threshold to 0.1 and see if pred_pos stops being zero.
Quick test inside `run_one`:
```python
mp = postprocess(mp, out["predictions"]["object_existence"], threshold=0.1, do_nms=False)
```

## Next steps (in order)

All three hypotheses are now CLI-selectable — no code edits needed.

1. **Sanity check with a ground-truth "recognizable" prompt** on s0000:
   - Run BiomedParse's own example npz (if any ship with the HF repo) to
     verify the checkpoint itself produces non-empty masks. If it's also empty,
     we have a wider checkpoint / forward-path issue (check ckpt state_dict
     keys vs model state_dict).
2. **Test Hypothesis A (input scale)**:
   ```bash
   $BP_PY scripts/biomedparse_prepare.py --scale_mode hu --out_suffix _hu --max_cases 3
   $BP_PY scripts/biomedparse_evaluate.py \
       --npz_root data/M3D_RefSeg_biomedparse_hu \
       --max_cases 1 --tag smoke_hu
   ```
   also try `--scale_mode raw01` with `--out_suffix _raw01`.
3. **Test Hypothesis B (CLIP truncation)**:
   ```bash
   $BP_PY scripts/biomedparse_evaluate.py --max_cases 1 \
       --one_prompt_per_forward --tag smoke_per_prompt
   ```
4. **Test Hypothesis C (threshold/NMS)**:
   ```bash
   $BP_PY scripts/biomedparse_evaluate.py --max_cases 1 \
       --score_threshold 0.1 --no_nms --tag smoke_thr0.1_nonms
   ```
   Combinations are fair game (e.g. HU data + no_nms + per-prompt).
5. Once pred_pos > 0 on smoke, prepare all 208 cases with the winning scale:
   ```bash
   $BP_PY scripts/biomedparse_prepare.py --max_cases 0  # or with --scale_mode X
   ```
   then full eval:
   ```bash
   $BP_PY scripts/biomedparse_evaluate.py --max_cases 0 --slice_batch_size 4
   ```
   Expected runtime: ~70-90 min on a 4070 (similar to VoxTell).
6. Append results as row #12 / #13 in README.md table + Exp10 section in
   `docs/project_status.md`. Commit.

### New CLI flags added after pause

`biomedparse_prepare.py`:
- `--scale_mode {imagenet255, hu, raw01}` — default imagenet255 (current behavior)
- `--out_suffix STR` — append to out_root so modes coexist (e.g. `_hu`)

`biomedparse_evaluate.py`:
- `--score_threshold FLOAT` (default 0.5) — Hypothesis C
- `--no_nms` — Hypothesis C
- `--one_prompt_per_forward` — Hypothesis B
- `--tag STR` — CSV filename suffix so debug runs don't clobber each other
- Progress lines now show `empty_pred` counts; summary reports input value range
  of first case (quick sanity check that scale_mode landed as expected).

## Key files touched this session

- `third_party/detectron2/setup.py` — ext_modules=[] patch
- `scripts/biomedparse_prepare.py` — NEW
- `scripts/biomedparse_evaluate.py` — NEW
- `data/M3D_RefSeg_biomedparse/s000[0-2].npz` — 3 smoke cases
- `third_party/BiomedParse/model_weights/` — weights, 4.46 GB

## Tasks state when paused

```
#12 [completed] Install detectron2 (python-only)
#13 [completed] Get HF token for BiomedParse weights
#14 [completed] Write M3D-RefSeg → BiomedParse .npz adapter
#15 [completed] Write BiomedParse eval script
#16 [completed] Smoke-test BiomedParse on 3 cases   -- runs, pred=empty
#17 [pending]   Full 208-case BiomedParse eval      -- blocked on smoke
```

## How to resume quickly next session

```bash
# activate the right python
BP_PY=/c/Users/63485/miniconda3/envs/biomedparse_v2/python.exe

# confirm imports still work
cd /i/M3D
$BP_PY -c "from detectron2.modeling import Backbone; print('ok')"

# smoke test (fast, ~30s per case)
$BP_PY scripts/biomedparse_evaluate.py --max_cases 1 --slice_batch_size 2
```

If `pred_pos > 0` — you're unblocked, go to full run.
If still 0 — work Hypothesis A first (HU-like rescale of imgs).
