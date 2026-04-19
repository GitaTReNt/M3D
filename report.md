# M3D-RefSeg: Text-Guided 3D Medical Image Segmentation
**项目目标、研究路线图与进展报告**

日期：2026-04-18
负责人：Yuntian Wu (trent.wu16@gmail.com)

---

## 0. 项目目标与研究问题

### 0.1 任务定义

给定一个 3D CT volume 和一段自由文本的放射科描述，输出对应的 3D 二值 mask。

- **输入**：`ct (D, H, W)` float CT 体 + 一段英文临床描述
  例："*An irregular-shaped mass in the right inguinal region with enhancement at the edge, considered as metastasis.*"
- **输出**：`mask (D, H, W)` 二值

关键难点：文本不是干净的解剖词汇表，而是**含病灶类型 + 解剖位置 + 形态修饰 + 鉴别诊断**的完整临床句子，文本→空间的映射必须**联合看图像**才能完成。

### 0.2 Project Goal（面向 12 个月）

把 **text-guided 3D referring expression segmentation** 在 M3D-RefSeg 上的 SOTA 从目前的 **Dice 0.1327** 推到 **≥ 0.30**，并形成一个可公开复现的 benchmark。

- Oracle-bbox 上限（MedSAM + GT bbox）= **0.5482**，这是空间 prompt 完美时的理论天花板。
- 当前 text-guided 最好 = **0.1327**（VoxTell, Exp9）。差距 ~**4.1×**。
- 目标把差距缩到 **≤ 1.5-2×**（Dice 0.27-0.37）。
- 学术目标：顶会（CVPR/MICCAI/NeurIPS）workshop 或 short paper 级别的 contribution。

### 0.3 Research Questions

**主问题 (RQ0)**：当前 SOTA 在"自由临床文本 → 3D 像素 mask"这一任务上的能力边界在哪里，如何系统性地推进？

具体分解：

| # | Research Question | 对应实验 | 先验假设 |
|---|-------------------|----------|----------|
| **RQ1** | Zero-shot SOTA 在 RefSeg 上的天花板是多少？ | VoxTell (Exp9) ✓ / BiomedParse v2 (Exp12 阻塞中) / MedSAM2 / SAT-Pro | BiomedParse v2 ≥ VoxTell，0.15-0.25 |
| **RQ2** | Prompt 粒度（full clinical sentence vs. single finding-type keyword）怎么影响性能？ | VoxTell raw vs. structured (Exp9 vs 9b) ✓，要在 BiomedParse 上复现 | raw > structured，但差距依模型而异 |
| **RQ3** | LoRA / full FT 在 166 个训练 case 上能把 zero-shot 推多高？ | BiomedParse / VoxTell + LoRA，5-fold CV | 0.35-0.55，取决于哪个 base |
| **RQ4** | 合成 prompt 扩充（GPT-4V paraphrase、结构化分解）在低数据情境下帮助多大？ | 784 masks × {raw, short, location-only, anatomy+finding, hard-negative} → ~3k 条 (img, text, mask) + LoRA | 稀有类（bony / diffuse）+3-5 pp，常见类（mass / nodule）边际 |
| **RQ5** | Text encoder 的领域匹配性是不是主要瓶颈？ | BiomedParse 替换 CLIP → CT-CLIP / RadBERT / BiomedCLIP / Qwen3-Embedding | 领域 encoder +3-5 pp，对 rare finding 尤其显著 |
| **RQ6** | 训练集构建的 text→spatial prior（z-range, centroid, organ-level ROI）作为后处理插件能否稳定提升？ | Prior bank（train-only）+ prior-aware 后处理，bolt-on 到任一 base segmenter | 模型无关 +3-5 pp，对过分割有效 |
| **RQ7** | 按 focal / multifocal-diffuse / bony 路由到不同 pipeline 能不能改善整体 Dice？ | 样本分类器 → 3 条 post-processing 分支 | 整体 +3-5 pp，diffuse/bony 子集 +10-15 pp |

### 0.4 成功指标与交付物

**定量**：
- **Primary**：text-guided test Dice ≥ **0.30**（5-fold CV mean，或 fixed train/val/test 的 test set）。
- **Stretch**：text-guided test Dice ≥ **0.40**（约为 oracle 的 73%）。
- **Subgroup stretch**：focal-mass 类 Dice ≥ 0.45（接近 oracle，证明 focal 已解决）。

**定性 / 交付物**：
- 三层对比表：**zero-shot / LoRA / full-FT** × **{VoxTell, BiomedParse v2, MedSAM2, SAT-Pro}**
- 按 finding-type（mass / nodule / cyst / lymph node / effusion / fracture / emphysema / …）的分组 Dice breakdown
- 至少 3 个 ablation：text encoder 替换（RQ5）、prior-aware 后处理（RQ6）、prompt 扩充（RQ4）
- 公开 repo + 官方 train/val/test split + baseline 权重
- 每个 mask 的 "prompt quality" 中间指标：解析覆盖率、centroid 落 box 内、slice-range recall、pseudo-box IoU

### 0.5 Non-goals（明确不做）
- ❌ 训练新的 3D vision backbone（算力不够，且 nnUNet/BoltzFormer 已足够）。
- ❌ 追求所有 finding-type 都做好；优先 focal 目标，diffuse/bony 单独做 subgroup 章节。
- ❌ 2D slice-only 方法对比（已在 Exp6-9 证明天花板低于 0.03）。
- ❌ 纯文本检索方法（Exp4-5 已证明"文本不看图像"在 npy 坐标空间必然失败）。

---

## 1. 当前状态快照

### 1.1 已完成的 baseline

| # | Method | Mean Dice | Dice≥0.5 | 备注 |
|---|--------|-----------|----------|------|
| 03 | **MedSAM + oracle bbox (optimized)** | **0.5482** | — | **Ceiling**, 30 cases |
| 05 | TF-IDF → bbox → MedSAM | 0.0308 | 0% | Text-only 必败 |
| 06 | Prompt Compiler (rule+atlas+retrieval) | 0.0235 | 0% | Text-only 必败 |
| 08 | MedCLIP-SAMv2 v2 (focus-rank) | 0.0284 | 0% | Earlier text-guided 最优 |
| 09 | **VoxTell zero-shot (raw)** | **0.1327** | **13.4%** | **当前 text-guided SOTA**，208 cases |
| 11 | VoxTell zero-shot (structured) | 0.0587 | 4.1% | 单词 prompt 差 2.3× |
| 12 | BiomedParse v2 zero-shot | — | — | **阻塞中**，详见 §2 |

**已经回答的 RQ**：
- **RQ2（部分）**：在 VoxTell 上，raw 临床句显著优于 structured 关键词（0.1327 vs 0.0587，2.3×）。Qwen3-Embedding-4B 能从整句中提取 "finding type + anatomy + morphology" 的联合信号。需要在 BiomedParse 上验证这个结论是否普适（BiomedParse 用 CLIP text encoder，可能反而偏好 structured）。

**尚未启动的 RQ**：RQ3 / RQ4 / RQ5 / RQ6 / RQ7（微调、数据扩充、encoder 消融、prior 后处理、路由）。

### 1.2 BiomedParse v2（Exp12）阻塞点

环境、权重（4.46 GB）、prepare / evaluate 两个脚本都就绪，smoke test 端到端跑通但 `pred_pos = 0`（模型输出全空）。三个待验证假设：

- **A. 输入 scale 不对**（最可能）—— 当前 `ct * 255` 走 ImageNet-style mean/std；可能需要 HU 范围或 [0,1]。
- **B. CLIP 77-token 截断** —— `[SEP]` 拼两条长临床句子会被截到第二条之后。
- **C. 阈值过严** —— `score_threshold=0.5` + NMS 都开着。

详细排查见 `docs/biomedparse_setup_report.md`。

### 1.3 本轮代码升级（2026-04-18）

三个假设全部做成 CLI flag，不需要再改代码：

`scripts/biomedparse_prepare.py`
- `--scale_mode {imagenet255, hu, raw01}` + `--out_suffix` → Hypothesis A
- 写完打印首个 case 的 dtype / range / mean 做 sanity check

`scripts/biomedparse_evaluate.py`
- `--score_threshold` + `--no_nms` → Hypothesis C
- `--one_prompt_per_forward` → Hypothesis B（逐 prompt forward，"先到先得"合并）
- `--tag` → CSV 后缀，多次 debug 不互相覆盖
- 进度行加 `empty_pred` 计数；OOM 回退会记住 `slice_batch_size=1`

---

## 2. 近期路障：解除 BiomedParse smoke（回答 RQ1 的前提）

按顺序跑 smoke（每条 ~30s）：

```bash
BP_PY=/c/Users/63485/miniconda3/envs/biomedparse_v2/python.exe

# Hypothesis A — 最可能
$BP_PY scripts/biomedparse_prepare.py --scale_mode hu --out_suffix _hu --max_cases 3
$BP_PY scripts/biomedparse_evaluate.py --npz_root data/M3D_RefSeg_biomedparse_hu \
    --max_cases 1 --tag smoke_hu

# Hypothesis B
$BP_PY scripts/biomedparse_evaluate.py --max_cases 1 \
    --one_prompt_per_forward --tag smoke_per_prompt

# Hypothesis C
$BP_PY scripts/biomedparse_evaluate.py --max_cases 1 \
    --score_threshold 0.1 --no_nms --tag smoke_thr0.1_nonms
```

任一组合 `pred_pos > 0` 即解除阻塞 → 进全量 208 case 评测 → 填 Exp12 入表。

---

## 3. 实验路线图（按 RQ 组织）

### RQ1 — Zero-shot baseline 覆盖

| 模型 | 状态 | ETA |
|------|------|------|
| VoxTell v1.1 | ✓ Done (0.1327) | — |
| BiomedParse v2 | 阻塞 → 预计本周解除 | 1 天 |
| MedSAM2（若 2025Q4 已发布） | 待评估 API | 2-3 天 |
| SAT-Pro（上海 AI Lab） | 待调研权重获取 | 2-3 天 |

**deliverable**：覆盖 4 个 zero-shot baseline 的完整表格 + 按 finding-type 分组的 heatmap。

### RQ2 — Prompt 粒度

复用 `--mode raw / structured` 已有逻辑，在每个新 baseline 上同时跑两种 prompt，验证 "raw > structured" 是不是 VoxTell 特有还是普遍现象。
**deliverable**：所有 baseline × 两种 prompt 的交叉表。

### RQ3 — LoRA / Full FT（核心实验）

**模型**：BiomedParse v2、VoxTell 两条线并行。
**设置**：
- LoRA rank=16，仅 text encoder 最后 4 层 + mask decoder，backbone 冻结
- Loss：DiceCE + boundary
- Optim：AdamW lr=1e-4 for LoRA，1e-5 for decoder
- Scheduler：cosine，20 epoch，warmup 2 epoch
- Batch：每卡 1 case（volumetric）×  grad accum 4
- 5-fold CV，每 fold 166 train / 42 test

**硬件**：H100 80GB（单卡够用）或 5090 32GB（需开 bf16 + grad checkpointing）。

**deliverable**：三层表（zero-shot / LoRA / full-FT）+ learning curve + 训练数据量消融（32 / 64 / 128 / 166 cases）。

### RQ4 — 合成 prompt 扩充

**实现**：
1. 对 784 原始 prompt，用 GPT-4V 或 Claude 4.7 生成 4 类变体：
   - location-only（"right inguinal region"）
   - anatomy + finding（"right inguinal mass"）
   - short canonical（"irregular right inguinal mass"）
   - hard-negative（左右翻转、同病例其他 lesion 的描述）
2. 结构化 metadata 表：`anatomy / side / level / finding_type / focality / bbox_norm / centroid_norm / slice_range / difficulty_tag`
3. 训练时对每个 (image, mask) 采样 prompt 变体作为正样本，hard-negative 作为对比损失。

**deliverable**：消融 — LoRA + {原 prompt} vs. LoRA + {增强 prompt}，按 finding-type 分组看增益分布。

### RQ5 — Text encoder domain match

**实现**：在 BiomedParse 架构内替换 text encoder：
- Baseline: `openai/clip-vit-base-patch32`（原配）
- Candidates: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`、`openai/CT-CLIP`（若可用）、`ncbi/MedCPT-Query-Encoder`、`Qwen/Qwen3-Embedding-4B`
- 修改点：`sem_seg_head.predictor` 的 text projection 维度对齐

**deliverable**：encoder 消融表。如果 CT-CLIP 明显好，写 RQ5 的 positive 结论；如果差异小，故事转为 "text encoder 不是当前瓶颈"。

### RQ6 — Prior-aware 后处理（模型无关）

**实现**：
1. 从 train split 构建 prior bank：每个 mask 的归一化 `(z_range, centroid, volume, bbox_norm)` + 对应 prompt 的结构化 label。
2. 推理时：输入 prompt → 结构化 label → top-k 检索相似 prior → 聚合出候选 `(z_range, centroid, max_volume)`。
3. 对 base segmenter 输出：
   - z_range 外的体素清零
   - 保留离 centroid 最近的 connected component（focal）或 ≥2 个 component（diffuse）
   - 若 volume > max_prior_volume × 1.5，二次阈值收紧

**deliverable**：VoxTell / BiomedParse 在 {zero-shot, +prior} 两种配置下的对比。

### RQ7 — 样本路由

**实现**：
1. 训练一个轻量分类器（LR 或 MLP on BiomedCLIP text feature）把 prompt 分成 `{focal, multifocal-diffuse, bony}`。
2. 每类走不同 post-processing：
   - focal：紧 bbox + largest CC + centroid-aware
   - multifocal-diffuse：multi-box + 保多 CC + organ-level ROI
   - bony：低阈值 + 无 CC 约束 + 沿脊柱方向延展

**deliverable**：整体 + 分子集的对比表，定量回答 "是否 one-size-fits-all pipeline 是错的"。

---

## 4. 数据切分策略

**硬约束**：case 级切分（同一 CT volume 的所有 mask 在同一 split）。

**方案（推荐）**：
1. **检查原 repo**：M3D-Cap / M3D-Seg 有官方 split，RefSeg 可能也有 → 若存在优先用，可比性最好。
2. **5-fold CV（默认）**：每 fold 166 train / 42 test；train 内再切 10% 当 val。报告 mean ± std。
3. **Stratified by finding-type**：mass / nodule / cyst / lymph node / effusion / bony / other 六类在三个 split 中分布均衡。
4. **Leave-one-case-out**（小规模消融用）：当做 prior bank 构建或 RQ6 的 sanity check。

**Deliverable**：`data/splits/refseg_5fold.json` 定稿 + README 更新。

---

## 5. 优先级与时间线

| 优先级 | 任务 | 对应 RQ | ETA | 依赖 |
|---|------|---------|-----|------|
| P0 | 解除 BiomedParse smoke 阻塞 | RQ1 | 0.5 天 | — |
| P0 | BiomedParse 全量 zero-shot 评测，填 Exp12 | RQ1+2 | 0.5 天 | smoke 通过 |
| P0 | 确定 train/val/test split（检查官方 + 自建 5-fold） | 所有 FT | 0.5 天 | — |
| P1 | 代码加 bf16 + slice_batch=32 + Sweep 模式 | 所有 | 1 天 | H100 到手 |
| P1 | MedSAM2 / SAT-Pro zero-shot 接入 | RQ1 | 2-3 天 | 权重获取 |
| P2 | LoRA 微调 BiomedParse（核心实验） | RQ3 | 3-5 天 | split 定稿 |
| P2 | LoRA 微调 VoxTell | RQ3 | 3-5 天 | 同上 |
| P3 | 合成 prompt pipeline + 再训 LoRA | RQ4 | 3 天 | GPT-4V API / MedGemma |
| P3 | Text encoder 替换消融 | RQ5 | 2-3 天 | LoRA baseline done |
| P4 | Prior bank + prior-aware 后处理 | RQ6 | 2-3 天 | split 定稿 |
| P4 | Finding-type 路由分类器 | RQ7 | 1-2 天 | 结构化 metadata 表 |

**预估总时长**：~3-4 周集中工作（H100 可用前提下）。

---

## 6. 风险与 Mitigation

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| BiomedParse smoke 三个假设都不对 | 低 | 高 | 用官方 demo npz（如有）做 minimal reproduction；查 ckpt state_dict key 对齐 |
| 5-fold 后 test set 太小，噪音盖过 LoRA 提升 | 中 | 中 | 用 paired bootstrap 报告 CI；或改 leave-one-out |
| BiomedParse LoRA 过拟合 208 case | 中 | 中 | strong aug + early stop + 合成 prompt 扩增 |
| MedSAM2 / SAT-Pro 权重无法获取 | 中 | 低 | 故事调整为 "BiomedParse + VoxTell 二元对比"，不影响主结论 |
| 合成 prompt 语义漂移（GPT-4V 幻觉） | 中 | 中 | 结构化 slot → 可控模板生成 + LLM 辅助润色，而非纯 LLM 生成 |
| H100 未及时到手 | 中 | 中 | LoRA + bf16 + grad checkpoint 可在 5090 32GB 上跑；仅 full-FT 受限 |

---

## 7. 附：关键文件清单

**本轮新增/修改**
- `scripts/biomedparse_prepare.py` — npy → npz 转换器（支持多 scale mode）
- `scripts/biomedparse_evaluate.py` — 两模式评测 + 三假设 debug flag
- `docs/biomedparse_setup_report.md` — BiomedParse 环境搭建 + 阻塞排查
- `report.md`（本文件） — 研究路线图

**既有资产**
- `README.md` — 项目入口 + 结果表
- `docs/project_status.md` — 完整实验日志（Exp1-9）
- `docs/plan.md` — 数据中心化的改进思路（prompt bank / prior retrieval / routing）
- `results/09_voxtell_{raw,structured}.csv` — VoxTell 最新结果
- `third_party/{MedSAM,MedCLIP-SAMv2,VoxTell,BiomedParse}/` — baseline 代码
- `data/M3D_RefSeg_npy/` — 208 预处理好的 case

**环境**
- 主环境：`PBAI`（torch 2.8+cu126）用于 MedSAM / VoxTell
- BiomedParse 专用：`biomedparse_v2`（torch 2.6+cu124）at `C:\Users\63485\miniconda3\envs\biomedparse_v2\python.exe`
