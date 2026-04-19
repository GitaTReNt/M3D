# Phase C Implementation Plan — BiomedParse v2 coarse-grounding finetune

Strategic rationale: see `new_plan.md` §4 Phase C. This document is the
concrete *how* — module layout, splits, losses, schedule, and ablations
needed to produce a trained BP v2 that outputs a MedSAM-usable pseudo-box.

---

## 0. Context recap (why this phase)

Phase B 3-way diagnostic on the full 208-case M3D-RefSeg (784 masks) gave:

| Pipeline                          | Mean Dice | Median | Dice≥0.5 | Empty |
| --------------------------------- | --------- | ------ | -------- | ----- |
| BP direct (raw prompt)            | 0.1286    | 0.0017 | 12.6%    | 2/784 |
| BP pseudo-box → MedSAM (raw)      | 0.1259    | —      | —        | —     |
| GT box → MedSAM (oracle, 126-row) | ~1.12×    | —      | high     | 0     |

Paired delta BP→MedSAM vs BP direct = **−0.0027**. Refinement brings
nothing. Ceiling retention (bp_medsam / gt_medsam) ≈ 11%. This confirms
H1 from `new_plan.md`: the bottleneck is text→space grounding, not
boundary decoding. Phase C therefore targets BP's coarse localisation
capability directly, not any new refinement module.

---

## 1. Scope & exit criteria

**In scope**
- LoRA / partial finetune of BP v2 on M3D-RefSeg with main + auxiliary supervision
- Auxiliary label derivation from GT masks
- Dev-split evaluation pipeline (including rerun of Phase B's 3-way)

**Out of scope (this phase)**
- Replacing MedSAM, training MedSAM2, prompt augmentation, prior bank

**Exit criteria (any two of three to declare Phase C done)**
1. BP direct Dice on **test split** ≥ 0.20 (up from 0.1286 baseline)
2. `predicted box → MedSAM` Dice on test split ≥ `bp_direct × 1.3`
   **and** ≥ 0.25 absolute (i.e. refinement starts adding value)
3. Slice-level existence accuracy ≥ 0.80 **and** bbox IoU on
   positive slices ≥ 0.30

If none of these are met after Stage 1+2, we stop and revisit the main
line (per `new_plan.md` §8 decision rule).

---

## 2. Training framework choice

Start from BP's official finetune recipe:
`third_party/BiomedParse/configs/finetune_biomedparse.yaml`
(AdamW, lr=1e-5, 10 epochs, batch=8, Dice+BCE+edge loss).

Our additions:
- A new datamodule wrapping M3D-RefSeg (2D slice sampling from 3D `ct.npy`)
- Aux heads + losses on top of BP's decoder output
- LoRA wrappers on selected modules (prefer `peft` for ViT-style blocks;
  fall back to manual LoRA for the transformer decoder if `peft` can't
  target it)

Why not write from scratch: the BP loss + datamodule + checkpoint loader
are already Hydra-composable; we inherit them and only add what's new.

---

## 3. Architecture — injection points

BP v2 modules (from `third_party/BiomedParse/src/model/`):
- `backbone` — vision encoder (Focal / Swin variant)
- `pixel_decoder`
- `transformer_decoder` — this is where text-image cross-attention lives
- `biomedparse.py` / `biomedparse_3D.py` — top-level wrapper

**Freeze / train plan (Stage 1):**

| Module                        | Stage 1           | Stage 2 (if dev plateau) |
| ----------------------------- | ----------------- | ------------------------ |
| backbone                      | frozen            | top 2 blocks LoRA        |
| pixel_decoder                 | LoRA (rank 16)    | LoRA rank 32             |
| transformer_decoder (X-attn)  | **LoRA (rank 32)**| unfreeze fully           |
| text projection               | trainable         | trainable                |
| aux heads (new)               | trainable         | trainable                |
| CLIP text encoder             | frozen            | frozen (bump lr if FT)   |

**Aux heads (attached after pixel_decoder / pooled features):**

| Head         | Supervision                              | Output shape          | Loss            |
| ------------ | ---------------------------------------- | --------------------- | --------------- |
| existence    | 3D-level: mask has any positive voxel    | (B, 1)                | BCE             |
| slice_exist  | per-slice: slice has any positive voxel  | (B, D)                | BCE, masked     |
| bbox_3d      | (z1,y1,x1,z2,y2,x2) normalised to [0,1]  | (B, 6)                | smooth L1 + GIoU|
| centroid_3d  | centre of mass normalised to [0,1]       | (B, 3)                | L2              |
| z_range      | z_min/z_max normalised                   | (B, 2)                | smooth L1       |

Aux heads are lightweight MLPs (`Linear → GELU → Linear`). Dropped at
inference unless explicitly requested.

---

## 4. Data pipeline

**Dataset**: M3D-RefSeg, 208 cases / 784 masks. Text: English.

**Split** (case-level to avoid leakage across masks of same case):

| Split | Cases | Masks (est.) | Purpose                         |
| ----- | ----- | ------------ | ------------------------------- |
| train | 150   | ~565         | training                        |
| dev   | 30    | ~115         | per-epoch metric tracking       |
| test  | 28    | ~104         | final Phase C evaluation (held) |

Split is derived deterministically from `hash(case_id)` and committed to
`data/M3D_RefSeg_splits/phase_c_{train,dev,test}.txt`.

**Aux label derivation** — `scripts/build_aux_labels.py`:
- Load `data/M3D_RefSeg_npy/<case>/mask.npy`
- For each `mask_id`:
  - `existence = int(mask_volume > 0)`
  - `slice_exist = [int((mask_vol[z]==mid).any()) for z in range(D)]`
  - `bbox_3d = (z1,y1,x1,z2,y2,x2)` of non-zero voxels, /`(D,H,W)`
  - `centroid_3d = COM(mask) / (D,H,W)`
  - `z_range = (z_min, z_max) / D`
  - `volume = voxels / (D*H*W)`
  - `component_count = scipy.ndimage.label(mask).max()`
- Write to `data/M3D_RefSeg_aux/<case>.json`:
  `{mask_id: {existence, slice_exist, bbox_3d, centroid_3d, z_range, volume, component_count}}`

**Prompt variants**: already exist from Phase B:
- `raw` — full clinical sentence (mean 0.1286 baseline)
- `structured` — anatomy/finding short form (mean 0.0056 baseline — much
  weaker; include in training but weight lower)

Training mini-batch draws both variants for the same `(case, mask_id)`
when available (consistency signal).

**Sampling**: 2D slice sampling strategy per BP's datamodule. For each
selected `(case, mask_id)`:
- 70% positive slices (slice_exist=1), 30% negative slices (existence=0)
- Negative slice supervises existence head to emit 0 and bbox head is
  masked out from the loss.

---

## 5. Loss design

```
L_total = L_main + λ_ex·L_existence
                 + λ_sex·L_slice_exist
                 + λ_bb·L_bbox
                 + λ_ct·L_centroid
                 + λ_zr·L_zrange
```

`L_main` = BP's existing loss (Dice + BCE + edge), unchanged.

**Initial λ** (to be tuned on dev):
- λ_ex  = 0.5
- λ_sex = 0.5
- λ_bb  = 0.2  (smooth L1) + 0.1 (GIoU)
- λ_ct  = 0.1
- λ_zr  = 0.1

Mask aux losses with `existence` flag: bbox/centroid/z-range loss is
zero-weighted when existence=0 (avoid teaching bbox on non-existent
targets).

---

## 6. Training stages

**Stage 1 — LoRA + aux heads (baseline run)**
- Config: `configs/phase_c_stage1.yaml`
- 20 epochs, lr=1e-5 for LoRA + 1e-4 for aux heads, batch=8
- Cosine schedule, 1-epoch warmup
- Gradient accumulation if VRAM tight
- Dev eval every epoch; save best by `dev_main_dice`
- Expected wall time: ~6–10 h on single 24 GB GPU

**Stage 2 — escalate (conditional)**
- Triggered if Stage 1 `dev_main_dice` < 0.18 **or** bbox IoU < 0.25
- Unfreeze top 2 backbone blocks, bump transformer_decoder LoRA to
  rank 32, lr=5e-6 on backbone, 20 more epochs
- Continue from Stage 1 best ckpt

**Stage 3 — full FT (last resort)**
- Only if Stage 2 still fails exit criteria
- Full FT with lr=1e-6 on backbone, 1e-5 elsewhere, 10 epochs
- Flag as risky; overfitting on 150 cases is plausible

---

## 7. Evaluation protocol

**Per-epoch on dev**
- `main_dice`, `main_empty_rate`
- `existence_acc`, `slice_exist_f1`
- `bbox_iou_pos` (IoU on positive cases only), `centroid_err_norm`,
  `z_range_recall`

**End of each stage on dev**
- Rerun Phase B pipeline: BP-direct vs BP→MedSAM vs GT→MedSAM
- Report paired delta and ceiling retention

**End of Phase C on test split** (once, after all stages done)
- Full metric table above
- Per-case qualitative viz: sample 5 good / 5 bad cases, save
  `results/phase_c_test/viz/<case>_<mid>.png` (CT slice + pred + GT)

Scripts:
- `scripts/train_biomedparse_phase_c.py` — training driver (Hydra)
- `scripts/eval_phase_c.py` — dev/test eval (wraps existing
  `biomedparse_evaluate.py` + aux metrics)
- `scripts/analyze_phase_c.py` — comparison tables + viz

---

## 8. Slice-level scorer integration (team member's work)

Decision tree based on scorer's training signal:

| Scorer signal           | Best integration route                                 |
| ----------------------- | ------------------------------------------------------ |
| MedCLIP-style sim       | Route 3: ablation only (v3 + scorer vs v3 + cls_filter)|
| Per-slice binary exist. | Route 2: distillation target for `slice_exist` head    |
|                         | — add `L_distill = KL(scorer_prob ‖ slice_exist_prob)` |
|                         | weighted 0.2, as an extra term in Stage 1              |
| Z-range regression      | Route 2 variant: distill into `z_range` head directly  |

Route 1 (BP→MedSAM slice gate at inference) is cheap and independent;
run as an **always-on ablation** regardless of which main route we pick,
to measure "what fraction of the Stage-1 gain is explainable by slice
gating alone".

---

## 9. Ablation grid (after Stage 1 converges)

Run as 1-epoch "finetune-from-stage1" probes:

| Ablation                 | Control  | What it tells us                 |
| ------------------------ | -------- | -------------------------------- |
| no aux losses            | Stage 1  | Is aux supervision the driver?   |
| aux only on existence    | Stage 1  | Cheapest aux signal that works?  |
| LoRA rank 16 vs 32 vs 64 | Stage 1  | Capacity vs overfit trade-off    |
| raw prompt only vs both  | Stage 1  | Is structured prompt net +/-?    |
| scorer distill on/off    | Stage 1  | Does teammate's scorer help BP?  |
| slice-gate at inference  | Stage 1  | Cheap inference-time win?        |

Record all in `results/phase_c_ablations/summary.csv`.

---

## 10. File & module manifest

**New files**
- `docs/phase_c_plan.md` (this file)
- `configs/phase_c_stage1.yaml` (Hydra override of BP finetune config)
- `configs/phase_c_stage2.yaml`
- `scripts/build_aux_labels.py`
- `scripts/make_phase_c_splits.py`
- `scripts/train_biomedparse_phase_c.py`
- `scripts/eval_phase_c.py`
- `scripts/analyze_phase_c.py`
- `src/datasets/refseg_datamodule.py` (if kept local; else under
  `third_party/BiomedParse/src/datasets/`)
- `src/model/aux_heads.py`

**Touched files**
- `third_party/BiomedParse/src/losses/biomedparse_loss.py`
  — extend to forward aux losses
- `third_party/BiomedParse/src/model/biomedparse.py`
  — expose hook for aux head inputs

Keep BP fork clean: prefer subclassing / monkey-patch in our scripts
over editing upstream files where possible.

---

## 11. Risks & fallbacks

| Risk                                  | Mitigation                                 |
| ------------------------------------- | ------------------------------------------ |
| Dataset too small (150 train cases)   | Heavy augmentation + early stop on dev     |
| LoRA not expressive enough            | Escalate to Stage 2 unfreeze               |
| Aux losses destabilise `main_dice`    | λ ramp-up: start aux λ=0, warm over 2 ep   |
| Overfit to raw prompt phrasing        | Dual-track train; Phase D is next anyway   |
| BP text encoder caps at 77 tokens     | Already hit in Phase A; keep one-prompt    |
|                                       | forward, no concat                         |
| Oracle GT→MedSAM baseline drifts      | Pin oracle to committed 126-row reference  |

---

## 12. Timeline — Week 2 mapped

| Day | Tasks                                                               |
| --- | ------------------------------------------------------------------- |
| 6   | `build_aux_labels.py`, `make_phase_c_splits.py`; verify label stats |
| 7   | Datamodule + aux head module + loss hookup; smoke on 5 cases        |
| 8   | Stage 1 full run (20 epochs); dev eval each epoch                   |
| 9   | Dev analysis; decide Stage 2 trigger; ablation grid (parallel)      |
| 10  | Rerun Phase B 3-way on dev using Stage-1 ckpt                       |
| 11  | Stage 2 (if triggered) OR scorer integration (if not)               |
| 12  | Ablation grid completion; consolidate results                       |
| 13  | Test-split evaluation (single shot); viz                            |
| 14  | Write-up; decide Phase D kick-off                                   |

---

## 13. Decision rule at end of Phase C

From `new_plan.md` §8 — the single question Phase C must answer:

> 经过微调后，BiomedParse 是否能稳定输出一个"MedSAM 用得上"的 pseudo-box？

Operationalised: does `predicted box → MedSAM` on test beat
`bp_direct × 1.3` **and** 0.25 absolute?

- **Yes** → proceed to Phase D (prompt augmentation).
- **No** → recheck grounding; do not start Phase D. Consider existence
  head distillation from a stronger teacher, or revisit scorer route 2.
