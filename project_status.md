# M3D-RefSeg: Text-Guided 3D Medical Image Segmentation — Project Status

## 1. Project Overview

**Task**: Given a 3D CT volume and a free-text radiology description, segment the corresponding 3D region of interest (Referring Expression Segmentation).

**Dataset**: M3D-RefSeg
- 208 cases, 784 annotated regions
- Data format: `ct.npy` (1, 32, 256, 256) float32 [0,1], `mask.npy` (1, 32, 256, 256), `text.json` {mask_id: description}
- Text descriptions are detailed radiology findings, e.g.:
  - "An irregular-shaped mass in the right inguinal region with enhancement at the edge on enhanced scanning"
  - "Multiple round transparent shadows in both lungs, considering bilateral emphysema"

**Repository**: GitHub (public), with MedSAM cloned into the project directory.

---

## 2. Current Approach: MedSAM with Oracle Bounding Box

### Pipeline

MedSAM is a 2D model (SAM ViT-B fine-tuned on medical images). Our inference pipeline:

1. For each CT volume, iterate over 32 slices
2. For each target region on each slice, extract 2D bounding box from ground-truth mask (oracle bbox, margin=5px)
3. Resize slice to 1024x1024, encode with ViT-B image encoder (fp16)
4. Feed bbox prompt to mask decoder, get binary mask prediction
5. Stack 2D predictions into 3D volume, compute Dice/IoU vs ground truth

**Key implementation**: `inference_medsam_refseg.py` — GPU-based, fp16 image encoder, single-slice-at-a-time to fit in 12GB VRAM (RTX 4070).

### Results: Oracle Bounding Box (30 cases, MedSAM fine-tuned weights)

| Metric | Value |
|--------|-------|
| Mean Dice | 0.5221 |
| Mean IoU | 0.4046 |
| Median Dice | 0.4872 |
| Dice >= 0.5 | 43 / 91 (47.3%) |

### Results: Comparison with Original SAM Weights (50 cases)

| Checkpoint | Mean Dice | Mean IoU | Dice >= 0.5 |
|------------|-----------|----------|-------------|
| SAM ViT-B (original) | 0.4496 | 0.3420 | 40.5% |
| MedSAM fine-tuned | **0.5221** | **0.4046** | **47.3%** |

MedSAM fine-tuned weights improve Dice by ~7 points over original SAM.

---

## 3. Failure Analysis

### 3.1 Over-Segmentation is the Dominant Failure Mode

46% of targets (38/83 non-empty) have prediction volume > 3x ground truth. Mean Dice for these: **0.263**.

### 3.2 Performance by Target Size

| GT Size | Count | Mean Dice | Over-Seg Ratio | Dice >= 0.5 |
|---------|-------|-----------|----------------|-------------|
| < 50 voxels | 12 | 0.329 | 5.3x | 0/12 |
| 50 - 500 | 26 | 0.469 | 4.1x | 11/26 |
| 500 - 5K | 31 | 0.422 | 3.8x | 12/31 |
| 5K - 50K | 12 | 0.756 | 1.4x | 11/12 |
| > 50K | 2 | 0.601 | 2.6x | 1/2 |

Small targets are universally poor. Large, compact structures (>5K voxels) perform well.

### 3.3 Spine/Bone vs Other Structures

| Category | Count | Mean Dice |
|----------|-------|-----------|
| Spine/bone-related | 27 | 0.376 |
| Other structures | 56 | 0.525 |

Diffuse, distributed structures (osteophytes, disc degeneration, vertebral changes) are particularly hard for bbox-based segmentation.

### 3.4 Worst-Performing Targets (Bottom 10)

| Case | Dice | GT Voxels | Pred Voxels | Description |
|------|------|-----------|-------------|-------------|
| s0000-2 | 0.039 | 461 | 8,246 | Enlarged lymph nodes (scattered) |
| s0057-3 | 0.093 | 1,049 | 9,568 | Lumbar spine weight-bearing line |
| s0012-1 | 0.108 | 153 | 2,067 | Abdominal calcification foci |
| s0061-1 | 0.117 | 182 | 2,159 | Lumbar vertebral bone density |
| s0046-1 | 0.122 | 786 | 6,575 | Bilateral emphysema |

Common pattern: scattered/diffuse pathology where bbox is an inherently poor prompt.

---

## 4. Improvement Attempts (Without Retraining)

### 4.1 Inference-Time Tricks

Tested on same 30 cases with MedSAM fine-tuned weights:

| Configuration | Mean Dice | Mean IoU | Dice >= 0.5 |
|---------------|-----------|----------|-------------|
| **Baseline** (single mask) | **0.5221** | **0.4046** | 43/91 |
| Multi-mask (IoU head selection) | 0.5150 | 0.3951 | **48/91** |
| Multi-mask + iterative refinement | 0.5063 | 0.3918 | 44/91 |
| Multi-mask + refinement + CC3D | 0.4986 | 0.3869 | 44/91 |
| CC3D only | 0.5074 | 0.3945 | 43/91 |

**Conclusion**: None of these tricks improved Mean Dice. Reasons:
- MedSAM was trained with single-mask output; its IoU head is unreliable for multi-mask selection
- Iterative refinement amplifies the initial over-segmentation
- Connected component filtering hurts scattered but correct predictions (e.g., multi-focal calcifications)

### 4.2 Implementation

- `inference_medsam_improved.py` — supports configurable tricks via `--tricks multimask,refine,cc3d,tta`

---

## 5. Core Problem: Text-Guided Performance

The real task is **text-guided** segmentation, not oracle bbox. Previous server experiments showed:

| Mode | Mean Dice |
|------|-----------|
| Oracle bounding box | ~0.51 |
| Text-guided | ~0.02 |

The **50x gap** shows that the bottleneck is entirely in converting text descriptions to spatial prompts, not in the segmentation model itself. MedSAM has no text understanding capability — it only accepts visual prompts (bbox, point, mask).

---

## 6. Text-Guided Experiments (No Retraining)

### Exp 4: TF-IDF Retrieval → bbox → MedSAM (30 cases)

- **Method**: For each text description, retrieve top-3 similar descriptions (leave-one-case-out) via TF-IDF cosine similarity, use their normalized bounding boxes as MedSAM prompt
- **Result**: Mean Dice = **0.0308**, 53/91 targets Dice=0
- **Why it failed**: Bag-of-words similarity cannot provide spatial localization. Different patients' same anatomy occupies different pixel coordinates in the (32, 256, 256) npy format

### Exp 5: Prompt Compiler (structured parsing + atlas + retrieval) (30 cases)

- **Method**: Parse text → structured slots (anatomy, side, finding_type, target_form) → atlas spatial prior → structured retrieval refinement → type-aware post-processing → MedSAM
- **Result**: Mean Dice = **0.0235**, 0/91 Dice≥0.5
- **Why it failed**: Same root cause — **spatial coordinates are not transferable across patients** in the npy format. The npy volumes are cropped/resized differently per case, so "right kidney" can be anywhere in pixel space. No amount of text parsing fixes this geometric misalignment

### Key Lesson from Exp 4-5

The fundamental problem is not text understanding — it's that **text cannot be mapped to pixel coordinates without looking at the image**. Any method that tries to predict spatial location from text alone (retrieval, atlas, rules) will fail on this data format. The solution must involve **joint text-image reasoning**: a model that reads both the text AND the image to determine where to segment.

---

## 7. Potential Solutions (Requiring Image-Text Joint Reasoning)

### 6.1 VoxTell (DKFZ, January 2026) — Recommended Priority 1

- **What**: Free-text promptable universal 3D medical segmentation model
- **Why**: Directly accepts clinical free-text as prompt, outputs 3D segmentation
- **Trained on**: 62K+ volumes, 1000+ anatomical and pathological classes
- **Usage**: `pip install voxtell`, CLI: `voxtell-predict -i input.nii.gz -p "liver cyst"`
- **GitHub**: https://github.com/MIC-DKFZ/VoxTell
- **Challenge**: Our data is npy format, needs conversion to NIfTI; GPU requirements TBD
- **Paper**: https://arxiv.org/abs/2511.11450

### 6.2 BiomedParse v2 (Microsoft) — Recommended Priority 2

- **What**: Text-prompted medical image segmentation across 9 modalities
- **Why**: Won 1st place at CVPR 2025 Foundation Models for Text-Guided 3D Segmentation Challenge (DSC 0.7497)
- **Method**: 2.5D approach — encodes 3D context via Fractal Volumetric Encoding
- **GitHub**: https://github.com/microsoft/BiomedParse
- **Challenge**: Complex setup (detectron2, CUDA 12.4), specific preprocessing required

### 6.3 M3D-LaMed (BAAI) — Recommended Priority 3

- **What**: Multi-modal LLM for 3D medical image analysis, with referring expression segmentation
- **Why**: **Designed specifically for M3D-RefSeg dataset** — zero data format conversion needed. Input is exactly our (1, 32, 256, 256) npy files
- **Models**: Phi-3-4B (recommended, ~16GB VRAM) or Llama-2-7B
- **GitHub**: https://github.com/BAAI-DCAI/M3D
- **HuggingFace**: `GoodBaiBai88/M3D-LaMed-Phi-3-4B`
- **Challenge**: 16GB VRAM needed, may be tight on RTX 4070 12GB

### 6.4 SegVol (BAAI, NeurIPS 2024)

- **What**: 3D CT segmentation with text/point/bbox prompts
- **Limitation**: Only supports predefined anatomical labels (200+), not free-form text
- **Workaround**: Use LLM to map free-text descriptions to anatomical label names
- **GitHub**: https://github.com/BAAI-DCAI/SegVol

### 6.5 Hybrid Pipeline (Combine Models)

- **Stage 1**: Use VoxTell/BiomedParse for text-guided coarse localization
- **Stage 2**: Extract bbox from coarse prediction, feed to MedSAM for refined segmentation
- **Rationale**: Leverage text understanding of one model + segmentation quality of another

### 7.6 MedCLIP-SAMv2 (MICCAI 2024 / MedIA 2025)

- **What**: BiomedCLIP + M2IB generates spatial saliency from text+image → derives bbox → SAM segments
- **Architecture**: Three-stage pipeline — text+image → Information Bottleneck saliency map → bbox/point prompts → SAM
- **Text encoder**: BiomedCLIP (PubMedBERT + ViT-B), fine-tuned with DHN-NCE loss
- **Key mechanism**: M2IB optimizes a learnable mask at ViT layer to find image regions most relevant to text (10 gradient steps per image)
- **Limitation**: 2D only, need to process slice-by-slice
- **Best use**: Extract BiomedCLIP+M2IB as text→saliency module, convert saliency to bbox, feed to MedSAM (fully decoupled)
- **GitHub**: https://github.com/HealthX-Lab/MedCLIP-SAMv2
- **GPU**: ~4-6 GB with SAM ViT-B, fits RTX 4070

---

## 8. Recommended Next Steps

1. **Try MedCLIP-SAMv2 pipeline** — BiomedCLIP+M2IB generates per-slice saliency from text+image → bbox → MedSAM. The only approach that does joint text-image reasoning. Fits 12GB.
2. **Try M3D-LaMed** — zero data conversion, designed for our exact task. May need fp16/quantization to fit 12GB.
3. **Try VoxTell** — strongest free-text capability, pip-installable. Convert npy → NIfTI.
4. **Try BiomedParse v2** — CVPR 2025 challenge winner, likely strongest overall performance.
5. Compare all results on same 30 cases as current MedSAM baseline.

---

## 8. Environment

- **Hardware**: NVIDIA RTX 4070, 12GB VRAM
- **Software**: Python (Miniconda PBAI env), PyTorch 2.11.0+cu126, CUDA 13.0
- **OS**: Windows 11

---

## 9. File Structure

```
D:/M3D/
├── inference_medsam_refseg.py          # Baseline MedSAM inference (oracle bbox)
├── inference_medsam_improved.py        # Improved version with tricks
├── MedSAM/                             # Cloned MedSAM repository
│   ├── segment_anything/               # SAM model code
│   └── work_dir/
│       ├── MedSAM/medsam_vit_b.pth            # Original SAM ViT-B weights (358MB)
│       └── MedSAM_finetuned/medsam_vit_b.pth  # MedSAM fine-tuned weights (375MB)
├── M3D_RefSeg_npy/                     # Dataset (208 cases, .gitignored)
│   ├── s0000/ {ct.npy, mask.npy, text.json}
│   ├── s0001/ ...
│   └── ...
├── results_medsam/                     # SAM weights results (50 cases)
├── results_medsam_ft/                  # MedSAM fine-tuned results (30 cases)
├── results_medsam_improved/            # Improved tricks results (30 cases)
└── project_status.md                   # This document
```
