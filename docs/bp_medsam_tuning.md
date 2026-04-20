# BP → MedSAM Specific Tuning Guide

Companion to `docs/phase_c_plan.md`. Same project, different angle:
**`phase_c_plan.md`** treats Phase C as a training system (splits, loss,
schedule). This file treats it as a tuning problem against **one metric**:
the `BP pseudo-box → MedSAM` Dice, currently **0.1259** (mean over
208 cases / 784 masks, raw prompt).

MedSAM is frozen. Everything here is about making BP's pseudo-box good
enough for MedSAM to actually help, instead of hurt.

---

## 1. Target & baseline state

| Pipeline                       | Dice    | Notes                           |
| ------------------------------ | ------- | ------------------------------- |
| BP direct (raw)                | 0.1286  | paper baseline                  |
| **BP → MedSAM (raw)**          | **0.1259** | **this is what we're moving** |
| GT box → MedSAM                | ~1.12×  | ceiling (oracle)                |
| Paired Δ (bp_medsam − bp_direct)| −0.0027 | refinement currently useless    |
| Ceiling retention              | ~11%    | (bp_medsam / gt_medsam)         |

**Objective**: raise BP→MedSAM Dice to ≥ `bp_direct × 1.3` **and** ≥ 0.25
absolute on test split (Phase C exit criterion #2 from `phase_c_plan.md`).
Equivalently: ceiling retention from 11% → 30%+.

---

## 2. Failure-mode decomposition

Paired Δ ≈ 0 means "MedSAM helps in ~half the cases, hurts in ~half, net
zero." From spot checks of `results/12_phase_b_analysis/phase_b_merged_raw.csv`:

| Failure mode                                        | Approx share | Lever that fixes it         |
| --------------------------------------------------- | ------------ | --------------------------- |
| BP outputs box on a **wrong slice**                 | ~40%         | `slice_exist` head          |
| BP's box is on a right slice but **too large**      | ~25%         | `bbox` regression + GIoU    |
| BP's box **misses centre** of target                | ~20%         | `centroid` head             |
| BP outputs box but target is elsewhere (diffuse/multifocal) | ~10% | `z_range` + component strategy |
| Empty BP output                                     | ~5% (2/784)  | `existence` (already near 0)|

Numbers are rough — re-estimate after Stage 1 from the dev 3-way CSV.

---

## 3. Free wins — inference-time sweep (do BEFORE training)

Three knobs in `scripts/biomedparse_evaluate.py` +
`scripts/inference_medsam_from_pseudoboxes.py` that cost zero training
and must be pinned before we interpret any Stage 1 result.

### Knob 1: BP coarse-mask threshold

Current: `sigmoid > 0.5`. BP's raw logits on M3D-RefSeg skew low in
Phase A (hence the `--score_threshold 0 --no_nms` bypass). The binarisation
threshold on the resulting mask is separate and unswept.

**Sweep on dev split**:

```bash
for thr in 0.30 0.40 0.50 0.60; do
  python scripts/biomedparse_evaluate.py \
    --npy_root data/M3D_RefSeg_npy \
    --cases_list data/M3D_RefSeg_splits/phase_c_dev.txt \
    --score_threshold 0 --no_nms \
    --mask_binarize_thresh $thr \
    --out_csv results/sweep/bp_direct_thr${thr}.csv \
    --dump_boxes results/sweep/boxes_thr${thr}.json
  python scripts/inference_medsam_from_pseudoboxes.py \
    --pseudobox_json results/sweep/boxes_thr${thr}.json \
    --tag thr${thr}
done
```

Pick the threshold maximising BP→MedSAM Dice on dev. Expected gain: 1–3
points in absolute Dice for free.

### Knob 2: pseudo-box margin

Box extraction in Phase B uses a fixed margin. Too small clips target;
too large floods MedSAM's prompt.

**Sweep on dev split** (same loop, vary `--bbox_margin`):

```bash
for m in 0 3 5 8 12; do
  # ... same as above with --bbox_margin $m
done
```

Pick the margin maximising BP→MedSAM Dice on dev.

### Knob 3: connected-component strategy

BP coarse mask often has multiple disconnected blobs. Current behaviour
is per-slice bbox over all pixels combined. Alternatives:

| Strategy                | Implementation                          |
| ----------------------- | --------------------------------------- |
| `all_pixels` (current)  | one bbox covering every positive pixel  |
| `largest_cc`            | keep only largest 3D connected component|
| `top2_cc`               | largest two components, one box each    |
| `per_slice_cc`          | per-slice largest CC only               |

For focal cases, `largest_cc` is usually best; for multifocal / diffuse,
`top2_cc` avoids dropping lesions. Sweep on dev.

**Stop rule for Section 3**: pin the (threshold, margin, CC strategy)
triple that gives the best dev BP→MedSAM Dice with **zero-shot** BP.
That number is the new pre-training baseline. All Stage 1 gains are
measured against this, not 0.1259.

---

## 4. Training levers — ranked by effect on BP→MedSAM

Re-weighted from `phase_c_plan.md` §5. The main-line plan optimises main
Dice; here we optimise BP→MedSAM, which has a different priority:

```
L = L_main (Dice + BCE + edge)       # keeps coarse mask sane
  + 0.5  · L_slice_exist               # rank 1: kills wrong-slice boxes
  + 0.3  · L_bbox (smoothL1 + GIoU)    # rank 1: tightens x/y
  + 0.3  · L_existence                 # rank 2
  + 0.2  · L_zrange                    # rank 3
  + 0.1  · L_centroid                  # rank 3 (bbox subsumes it)
```

**Why this ordering is specific to BP→MedSAM**:

- `slice_exist` is the single biggest lever because MedSAM cannot
  recover from being run on a wrong slice — it produces a mask on garbage
  features.
- `bbox` regression with GIoU directly optimises the object MedSAM
  consumes. Without it, box quality depends on mask→bbox conversion,
  which loses information.
- `centroid` is deprioritised because a well-trained `bbox` head implies
  correct centre. Keep it as a cheap regulariser, not a load-bearing
  signal.
- `existence` has ceiling because BP is already at 2/784 empty — the
  gain here is bounded.

---

## 5. LoRA injection points for this axis

BP has three places where x/y grounding happens:

| Module                          | Plan  | Rationale (for BP→MedSAM)                       |
| ------------------------------- | ----- | ----------------------------------------------- |
| Vision backbone (Focal/Swin)    | frozen| Phase B shows features aren't the bottleneck    |
| `pixel_decoder`                 | LoRA 16| helps mask quality → helps bbox derivation      |
| `transformer_decoder` X-attn    | **LoRA 32** | this is where text ↔ image binds          |
| Text projection                 | trainable | small, cheap, aligns text embedding         |
| Aux heads (new)                 | trainable | required                                    |

Stage 2 escalation (if needed): unfreeze `transformer_decoder` fully,
keep backbone frozen. Only escalate backbone to LoRA if Stage 2 also
stalls.

Rule of thumb: every doubling of LoRA rank buys us 1–2 points of
`bbox_iou_pos` and saturates around rank 64 on a 150-case train split.
Don't go past rank 32 without dev evidence.

---

## 6. Mid-training proxy metrics

These predict BP→MedSAM gain **without** running MedSAM each epoch
(which would add ~10 min/epoch). Log all of them per epoch on dev:

| Metric             | Definition                                         | Target by end of Stage 1 |
| ------------------ | -------------------------------------------------- | ------------------------ |
| `slice_exist_f1`   | F1 on per-slice binary existence                   | ≥ 0.80                   |
| `bbox_iou_pos`     | IoU of predicted vs GT bbox on positive slices     | ≥ 0.35                   |
| `centroid_err_norm`| L2(centroid_pred − centroid_gt) / diag(vol)        | ≤ 0.10                   |
| `z_range_recall`   | fraction of GT z-slab covered by predicted z-slab  | ≥ 0.75                   |
| `main_dice_dev`    | standard Dice                                      | ≥ 0.20                   |

**Rule**: if all five targets are met, BP→MedSAM Dice will almost
certainly hit 0.25+ without further tuning.

If `main_dice` is up but `bbox_iou_pos` is flat → the model is learning
mask shape but not localisation. Increase `λ_bbox` and keep training.

If `bbox_iou_pos` is up but `slice_exist_f1` is flat → model places
good boxes on good slices but doesn't gate wrong ones out. Increase
`λ_slice_exist` or add the scorer distillation (see §8).

---

## 7. Post-Stage-1 decision tree

After Stage 1 converges, run the full 3-way diagnostic on **dev split**
(not test — keep test sealed until Phase C exit):

```bash
python scripts/biomedparse_evaluate.py \
  --npy_root data/M3D_RefSeg_npy \
  --cases_list data/M3D_RefSeg_splits/phase_c_dev.txt \
  --model_ckpt outputs/phase_c_stage1/best.ckpt \
  --out_csv results/phase_c_stage1/bp_direct_dev.csv \
  --dump_boxes results/phase_c_stage1/boxes_dev.json
python scripts/inference_medsam_from_pseudoboxes.py \
  --pseudobox_json results/phase_c_stage1/boxes_dev.json \
  --out_dir results/phase_c_stage1 --tag dev
python scripts/analyze_phase_b.py \
  --bp_direct_raw results/phase_c_stage1/bp_direct_dev.csv \
  --bp_medsam_raw results/phase_c_stage1/12_biomedparse_medsam_raw_dev.csv \
  --out_dir results/phase_c_stage1/analysis
```

Decide based on three numbers:

| BP direct ↑ | BP→MedSAM ↑ | Retention | Action                                   |
| :---------: | :---------: | :-------: | ---------------------------------------- |
| yes         | yes         | ≥ 30%     | proceed to test split; begin Phase D     |
| yes         | yes         | 15–30%    | short Stage 2 on `bbox`+`slice_exist`    |
| yes         | **no**      | any       | `bbox` head underweighted — raise λ_bbox, retrain |
| no          | no          | any       | Stage 2 full (unfreeze X-attn)           |
| direct unchanged, medsam ↑ | — | — | unexpected; inspect per-case deltas |

---

## 8. Teammate's slice scorer — integration paths

The scorer produces per-slice binary existence probability. Two valid
routes on this axis:

### Route A: inference-time gate on BP→MedSAM

Cheapest. No training. Add after BP forward, before MedSAM:

```python
# pseudo-code
for z in range(D):
    if scorer_prob[z] < gate_thresh:
        pseudo_boxes[z] = None  # skip MedSAM call on this slice
```

Tune `gate_thresh` on dev. Expected behaviour: kills 30–60% of
wrong-slice boxes at thresh=0.5, trades ~5% recall for ~20% precision
→ net Dice up if precision dominates current failure mix (which our
§2 decomposition says it does).

Gain is mechanical; should be re-measured each Stage.

### Route B: distillation teacher for `slice_exist` head

Add to Stage 1 loss:

```
L_distill = KL( stop_grad(scorer_prob) || slice_exist_sigmoid )
L = ... + 0.2 · L_distill
```

Scorer teaches BP's `slice_exist` head a well-calibrated per-slice
existence signal, even on training cases where GT is sparse. Cheap to
implement (one extra forward pass through frozen scorer).

**Only valid if scorer's train split is disjoint from Phase C's dev/test
split** — otherwise dev metrics leak. Must re-split or retrain scorer on
Phase C's train-only cases first.

---

## 9. Ablations specific to this axis

Run 1-epoch probes branching off Stage 1 best checkpoint. Record in
`results/phase_c_ablations/bp_medsam_summary.csv`:

| Ablation                         | Expected BP→MedSAM Δ  |
| -------------------------------- | --------------------- |
| `-L_slice_exist` (drop)          | −3 to −6 points       |
| `-L_bbox` (drop)                 | −2 to −4 points       |
| `+scorer distill (Route B)`      | +1 to +3 points       |
| `+scorer gate (Route A)`         | +1 to +4 points       |
| `bbox margin: best vs 0`         | +0.5 to +2 points     |
| `CC strategy: largest vs all`    | ±1 point (case-type dependent) |
| `LoRA rank 16 → 32`              | +1 to +2 points       |
| `raw-prompt only vs dual-prompt` | −1 to +1 point        |

If actual numbers deviate from expected significantly, stop and debug
the training recipe before spending more compute.

---

## 10. Stage 2 trigger conditions (for this axis)

Enter Stage 2 if any is true after Stage 1:

- `bbox_iou_pos_dev` < 0.25 AND `main_dice_dev` ≥ 0.18
  → localisation bottleneck; unfreeze X-attn
- `slice_exist_f1_dev` < 0.75
  → per-slice existence still weak; add scorer distillation if not already
- BP→MedSAM retention on dev < 15%
  → grounding broadly stuck; consider Stage 2 + higher aux weights

**Do not enter Stage 2 just because main Dice plateaued** — main Dice ≥
0.20 with proxy metrics at target is sufficient to move to test split.

---

## 11. Single-line success criterion

Stage 1 is successful on the BP→MedSAM axis iff:

```
dev(BP→MedSAM) ≥ 1.5 × dev(BP→MedSAM zero-shot after §3 sweeps)
AND dev(BP→MedSAM) ≥ 0.22
```

If yes → run §7 decision tree on retention to decide Stage 2.
If no → revisit loss weights or check label quality (aux labels from
`scripts/build_aux_labels.py` can be wrong if GT mask has stray voxels).
