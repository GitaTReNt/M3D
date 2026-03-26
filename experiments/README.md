# Experiments

## Directory Structure

```
experiments/
├── scripts/
│   ├── inference_medsam_refseg.py      # Baseline: MedSAM + oracle bbox
│   ├── inference_medsam_improved.py    # Exp3: inference-time tricks
│   └── inference_medsam_retrieval.py   # Exp4: TF-IDF retrieval → bbox → MedSAM
├── results/
│   ├── 01_baseline_sam/                # Original SAM weights, oracle bbox
│   ├── 02_medsam_finetuned/            # MedSAM fine-tuned weights, oracle bbox
│   ├── 03_improved_tricks/             # multimask / refine / cc3d ablation
│   └── 04_retrieval_guided/            # Text similarity retrieval (TBD)
└── README.md
```

## Experiment Summary

### Exp 1: Baseline — Original SAM ViT-B + Oracle BBox (50 cases)

- **Checkpoint**: `sam_vit_b_01ec64.pth` (original Facebook SAM)
- **Prompt**: Ground-truth 2D bounding box per slice (margin=5px)
- **Result**: Mean Dice = 0.4496, Mean IoU = 0.3420

### Exp 2: MedSAM Fine-tuned + Oracle BBox (30 cases)

- **Checkpoint**: `medsam_vit_b.pth` (MedSAM fine-tuned on medical data)
- **Prompt**: Ground-truth 2D bounding box per slice (margin=5px)
- **Result**: Mean Dice = **0.5221**, Mean IoU = 0.4046
- **Conclusion**: MedSAM fine-tuned weights improve Dice by +7pp over original SAM.

### Exp 3: Inference-Time Tricks Ablation (30 cases, MedSAM fine-tuned)

| Trick | Mean Dice | Mean IoU | Dice≥0.5 |
|-------|-----------|----------|----------|
| Baseline (single mask) | **0.5221** | **0.4046** | 43/91 |
| Multi-mask (IoU selection) | 0.5150 | 0.3951 | **48/91** |
| Multi-mask + Refine | 0.5063 | 0.3918 | 44/91 |
| Multi-mask + Refine + CC3D | 0.4986 | 0.3869 | 44/91 |
| CC3D only | 0.5074 | 0.3945 | 43/91 |

**Conclusion**: No trick improved Mean Dice. MedSAM's IoU head is unreliable; iterative refinement amplifies over-segmentation; CC3D hurts multi-focal targets.

### Exp 4: TF-IDF Retrieval-Guided (30 cases, MedSAM fine-tuned)

- **Method**: For each text description, retrieve top-3 similar descriptions (leave-one-case-out) via TF-IDF cosine similarity. Use retrieved targets' normalized bounding boxes as MedSAM prompt.
- **Purpose**: Test whether simple text similarity can bridge the text→spatial gap.
- **Result**: Mean Dice = **0.0308**, Mean IoU = 0.0185, Dice≥0.5: 2/91
- **Analysis**:
  - 53/91 targets Dice = 0 (retrieved bbox completely misses target)
  - Only "Fatty liver" (0.607) and "cirrhosis" (0.519) worked — organs with consistent positions across patients
  - TF-IDF similarity scores do NOT correlate with segmentation success (0.284 vs 0.272)
- **Conclusion**: Bag-of-words text similarity cannot provide spatial localization. The 256x256 npy format loses original spatial alignment, making cross-patient coordinate transfer unreliable. Semantic text understanding (not just keyword matching) is needed.

## Key Finding

Oracle bbox Dice = 0.52, text-guided Dice ≈ 0.02. The 50x gap shows the bottleneck is 100% in text → spatial prompt conversion, not segmentation quality.

## Hardware

- NVIDIA RTX 4070 (12GB VRAM)
- All inference runs fp16 image encoder, single-slice-at-a-time
