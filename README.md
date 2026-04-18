# M3D-RefSeg: Text-Guided 3D Medical Image Segmentation

Referring expression segmentation on 3D CT volumes — given a radiology text description, segment the corresponding 3D region.

## Task

Given `(ct_volume, text_description)`, output a 3D binary mask.

- **Input**: CT volume `(D, H, W)`, free-form English radiology text (e.g., `"An irregular-shaped mass in the right inguinal region with enhancement at the edge..."`)
- **Output**: Binary mask over the volume

## Dataset: M3D-RefSeg

- 208 cases / 784 annotated regions (BAAI, derived from TotalSegmentator)
- Format: `ct.npy (1, 32, 256, 256)` float32 [0, 1], `mask.npy (1, 32, 256, 256)`, `text.json {mask_id: description}`
- Not included in repo. Download separately:
  - Raw NIfTI: from BAAI M3D release → `data/M3D_RefSeg/`
  - Preprocessed npy (via `m3d_refseg_data_prepare.py`) → `data/M3D_RefSeg_npy/`

## Current results (M3D-RefSeg)

| # | Method | Mean Dice | Notes |
|---|--------|-----------|-------|
| 01 | SAM ViT-B (oracle bbox) | 0.4496 | Off-the-shelf SAM, 50 cases |
| 02 | MedSAM fine-tuned (oracle bbox) | 0.5221 | 30 cases |
| 03 | **MedSAM optimized (oracle bbox, adaptive margin + Otsu)** | **0.5482** | Current best with oracle prompt |
| 04 | MedSAM + inference tricks (multimask / refine / cc3d) | 0.4986 ~ 0.5150 | No stable gain |
| 05 | TF-IDF retrieval → bbox → MedSAM | 0.0308 | Text-guided baseline |
| 06 | Prompt Compiler (rule-based parse + atlas + retrieval) | 0.0235 | Text-guided |
| 07 | MedCLIP-SAMv2 v1 (no slice filter) | 0.0073 | Text-guided, BiomedCLIP saliency → bbox → MedSAM |
| 08 | MedCLIP-SAMv2 v2 (focus-rank slice filter) | 0.0284 | Earlier best text-guided |
| 09 | MedCLIP-SAMv2 v3 (body-mask gating + gamma + percentile) | 0.0117 | Text-guided |
| 10 | **VoxTell zero-shot (raw English clinical prompt)** | **0.1327** | 208 cases / 783 masks; 13.4% reach Dice≥0.5 |
| 11 | VoxTell zero-shot (structured finding-type keyword) | 0.0587 | Same cases; regex-extracted keyword (`mass`/`nodule`/…) |

**Headline gap**: oracle-bbox ceiling 0.55 vs best text-guided **0.13** (VoxTell raw) → text→spatial is still the bottleneck, but VoxTell closes the gap ~4.7× over the prior MedCLIP-SAMv2 best.

**Raw > Structured (surprising)**: on 783 masks, the full clinical sentence beats the single-word keyword by 2.3×. Single-case sanity testing had suggested the opposite — full-dataset statistics tell a different story. VoxTell's Qwen3-Embedding-4B encoder extracts joint signal from *finding type + anatomical location + morphology* that a lone keyword throws away.

## Repository layout

```
M3D/
├── scripts/                     Inference scripts (single source of truth)
│   ├── inference_medsam_*.py        MedSAM oracle-bbox / retrieval / tricks
│   ├── inference_medclip_medsam*.py MedCLIP-SAMv2 text-guided pipelines
│   ├── inference_prompt_compiler.py Prompt Compiler pipeline
│   ├── diagnose_medsam.py           fp16/fp32 + prompt diagnostics
│   └── prompt_compiler/             Structured prompt parsing + retrieval
├── results/                     Experiment outputs (01-09 numbered)
├── docs/
│   ├── project_status.md           Detailed experiment log (Chinese)
│   ├── plan.md                     Data-centric improvement roadmap
│   └── Project_Update_3.pdf        Mid-project presentation
├── third_party/                 Cloned repos (all gitignored)
│   ├── MedSAM/                     https://github.com/bowang-lab/MedSAM
│   ├── MedCLIP-SAMv2/              https://github.com/HealthX-Lab/MedCLIP-SAMv2
│   └── VoxTell/                    https://github.com/MIC-DKFZ/VoxTell
└── data/                        Datasets (gitignored)
    ├── M3D_RefSeg/                 Raw NIfTI (needed for VoxTell, nnUNet-style)
    └── M3D_RefSeg_npy/             Preprocessed npy (used by MedSAM scripts)
```

## Setup

### 1. Clone third-party repos

```bash
git clone https://github.com/bowang-lab/MedSAM third_party/MedSAM
git clone https://github.com/HealthX-Lab/MedCLIP-SAMv2 third_party/MedCLIP-SAMv2
git clone https://github.com/MIC-DKFZ/VoxTell third_party/VoxTell
```

See `third_party/README.md` for model-weight download paths.

### 2. Download model weights

- **MedSAM**: `medsam_vit_b.pth` → `third_party/MedSAM/work_dir/MedSAM/`
- **MedSAM fine-tuned** (on RefSeg, produced by us): → `third_party/MedSAM/work_dir/MedSAM_finetuned/medsam_vit_b.pth`
- **VoxTell v1.1**: via `huggingface_hub.snapshot_download(repo_id="mrokuss/VoxTell", allow_patterns=["voxtell_v1.1/*"])` → `third_party/VoxTell/weights/voxtell_v1.1/`

### 3. Environment

- PyTorch ≥ 2.0 (VoxTell requires < 2.9 due to PyTorch 2.9 OOM bug on 3D convs)
- `huggingface_hub`, `transformers`, `skimage`, `scipy`, `open-clip-torch`, `einops`, `SimpleITK`, `nibabel`
- For VoxTell: also `nnunetv2`, `acvl-utils`, `dynamic-network-architectures`, `positional-encodings`

## Running inference

### MedSAM oracle bbox (baseline)

```bash
python scripts/inference_medsam_refseg.py \
    --npy_root data/M3D_RefSeg_npy \
    --checkpoint third_party/MedSAM/work_dir/MedSAM/medsam_vit_b.pth \
    --mode oracle_bbox \
    --out_dir results/01_medsam_baseline \
    --bbox_margin 5
```

### MedCLIP-SAMv2 text-guided (best text pipeline so far)

```bash
python scripts/inference_medclip_medsam_v3.py \
    --npy_root data/M3D_RefSeg_npy \
    --medsam_ckpt third_party/MedSAM/work_dir/MedSAM_finetuned/medsam_vit_b.pth \
    --out_dir results/09_medclip_medsam_v3
```

### VoxTell zero-shot (requires raw `.nii.gz`, RAS orientation)

Single case (CLI):

```bash
voxtell-predict \
    -i data/M3D_RefSeg/s0000/ct.nii.gz \
    -o results/voxtell/s0000 \
    -m third_party/VoxTell/weights/voxtell_v1.1 \
    -p "right inguinal mass"
```

Full-dataset evaluation (both raw + structured prompts, ~70 min on a 4070):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/voxtell_evaluate.py \
    --data_root data/M3D_RefSeg \
    --model_dir third_party/VoxTell/weights/voxtell_v1.1 \
    --out_dir results \
    --max_cases 0
```

The script pre-encodes all unique prompts in a single Qwen residence, deletes the text backbone, then runs per-case sliding-window inference with cached embeddings — avoiding the repeated GPU↔CPU dance that causes CPU-RAM fragmentation OOM on Windows.

## What to read next

- `docs/project_status.md` — full experiment log, failure analysis, per-case stats
- `docs/plan.md` — data-centric improvement plan: structured prompt bank, train-only prior retrieval, focal/diffuse routing, prior-aware post-processing
- `docs/Project_Update_3.pdf` — mid-project presentation

## Hardware notes

Developed locally on a single RTX 4070 (12GB VRAM). VoxTell inference is tight on 12GB (Qwen3-Embedding-4B frozen text encoder ≈ 8GB in fp16) — may need 4/8-bit quantization for the text encoder, or run with CPU offload.
