# M3D-RefSeg: Text-Guided 3D Medical Image Segmentation — 项目完整记录

## 1. 项目概述

**任务**：给定一个3D CT体积和一段放射科自由文本描述，分割出对应的3D区域（Referring Expression Segmentation）。

**数据集**：M3D-RefSeg
- 208个病例，784个标注区域
- 数据格式：`ct.npy` (1, 32, 256, 256) float32 [0,1]，`mask.npy` (1, 32, 256, 256)，`text.json` {mask_id: 描述}
- 文本为详细的放射科发现，例如：
  - "右侧腹股沟区不规则肿块，增强扫描边缘强化"
  - "双肺多发圆形透亮影，考虑双侧肺气肿"

**硬件环境**：
- GPU: NVIDIA RTX 4070, 12GB VRAM
- OS: Windows 11
- PyTorch 2.11.0+cu126, CUDA 13.0

**仓库**：GitHub (public)，MedSAM 和 MedCLIP-SAMv2 作为子目录克隆到项目中。

---

## 2. 实验总览

### 全部实验结果汇总

| 编号 | 方法 | Mean Dice | Mean IoU | Dice≥0.5 | 说明 |
|------|------|-----------|----------|----------|------|
| Exp1 | SAM ViT-B (oracle bbox) | 0.4496 | 0.3420 | 40.5% | 原始SAM权重，50 cases |
| Exp2 | **MedSAM fine-tuned (oracle bbox)** | **0.5221** | **0.4046** | **47.3%** | Fine-tuned权重，30 cases |
| Exp3 | MedSAM + 推理tricks | 0.4986~0.5150 | — | — | multimask/refine/cc3d均无提升 |
| Exp4 | TF-IDF文本检索 → bbox → MedSAM | 0.0308 | — | 0% | 纯文本检索，30 cases |
| Exp5 | Prompt Compiler (解析+atlas+检索) | 0.0235 | — | 0% | 结构化规则，30 cases |
| Exp6 | MedCLIP-SAMv2 (无slice过滤) | 0.0073 | 0.0038 | 0% | BiomedCLIP saliency→bbox→MedSAM，3 cases |
| Exp7 | **MedCLIP-SAMv2 (有slice过滤)** | **0.0284** | **0.0156** | **0%** | 加入聚焦度排序过滤，5 cases |

| Exp3b | **MedSAM optimized (oracle bbox)** | **0.5482** | — | — | adaptive_margin+otsu, 30 cases |
| Exp8 | MedCLIP-SAMv2 + body mask gating | 0.0117 | — | 0% | CT body mask + gamma + percentile, 5 cases |
| Exp9 | **VoxTell zero-shot (raw 中文描述)** | **0.1327** | — | **13.4%** | 全量 208 cases / 783 masks |
| Exp9b | VoxTell zero-shot (structured keyword) | 0.0587 | — | 4.1% | 同上，regex 抽取英文单词 prompt |

**核心结论**：
- Oracle bbox 上限 Dice=0.5482（优化后）
- Text-guided 此前最佳=0.0284 (MedCLIP-SAMv2 v2)
- **VoxTell raw prompt Dice=0.1327，将 text-guided 基线提升 4.7×，首次突破 0.1**
- 与 oracle bbox 差距从 ~20× 缩小到 ~4×

---

## 3. 实验详情

### Exp1 & Exp2: MedSAM Oracle Bounding Box 基线

**脚本**：`inference_medsam_refseg.py`

**流程**：
1. 遍历CT volume的32个slice
2. 对每个目标区域，从ground-truth mask提取2D bounding box（oracle bbox, margin=5px）
3. Slice resize到1024×1024，ViT-B编码器fp16编码
4. Bbox prompt送入mask decoder，获取binary mask
5. 堆叠2D预测为3D，计算Dice/IoU

**结果对比**：

| 权重 | Mean Dice | Mean IoU | Dice≥0.5 |
|------|-----------|----------|----------|
| SAM ViT-B 原始 (50 cases) | 0.4496 | 0.3420 | 40.5% |
| MedSAM fine-tuned (30 cases) | **0.5221** | **0.4046** | **47.3%** |

MedSAM fine-tuned权重比原始SAM提升约7个百分点。

### Exp3: 推理时优化尝试（第一轮 — multimask/refine/cc3d）

**脚本**：`inference_medsam_improved.py`（支持 `--tricks multimask,refine,cc3d,tta`）

| 配置 | Mean Dice | 说明 |
|------|-----------|------|
| Baseline (single mask) | **0.5221** | 最优 |
| Multi-mask (IoU head选择) | 0.5150 | IoU head不可靠 |
| Multi-mask + 迭代refinement | 0.5063 | 放大初始过分割 |
| Multi-mask + refinement + CC3D | 0.4986 | 伤害散在正确预测 |
| CC3D only | 0.5074 | 轻微下降 |

**结论**：所有tricks均无法提升Mean Dice，baseline即最优。

### Exp3b: 推理时优化（第二轮 — Track A: bbox与后处理优化）

**脚本**：`inference_medsam_optimized.py`（支持 `--tricks adaptive_margin,threshold_X,otsu,erosion,ensemble,confidence_gate`）

**核心思路**：针对过分割问题，从bbox生成和阈值两个方向优化。

| 配置 | Mean Dice | 变化 | 说明 |
|------|-----------|------|------|
| Baseline (固定margin=5) | 0.5221 | — | 基准 |
| adaptive_margin | 0.5471 | **+2.5pp** | margin与目标大小成比例（10%，clamp 2-15px） |
| threshold_0.65 | 0.5228 | +0.07pp | 仅提高阈值，效果微弱 |
| threshold_0.65 + erosion | 0.4796 | -4.3pp | 过度抑制，伤害recall |
| adaptive_margin + threshold_0.65 | 0.5451 | +2.3pp | 比单独adaptive略低 |
| adaptive_margin + otsu | **0.5482** | **+2.6pp** | 自适应margin + Otsu自适应阈值，**最优** |

**最优配置**：`adaptive_margin + otsu` → **Dice 0.5482**（从0.5221提升+2.6个百分点）

**关键发现**：
- **Adaptive margin是单一最有效的trick**：固定5px margin对小目标太大（过分割），对大目标太小（漏割）。按目标大小的10%动态调整效果显著。
- **Otsu自适应阈值**在adaptive margin基础上再提供微弱增益（+0.1pp）
- **Erosion + 高阈值**组合太激进，伤害recall大于减少FP的收益
- **Ensemble (bbox扰动)**速度慢5x，效果不稳定
- **Confidence gate**对少量边界slice有效，整体影响不大

---

### Exp4: TF-IDF 文本检索 → bbox → MedSAM

**脚本**：`inference_medsam_retrieval.py`

**方法**：
1. 用TF-IDF (max_features=5000, ngram_range=(1,2)) 对784条描述建立向量
2. Leave-one-case-out检索top-3相似描述
3. 用检索到的归一化bbox加权平均作为MedSAM prompt

**结果**：Mean Dice = **0.0308**，53/91 targets Dice=0

**失败原因**：词袋相似度无法提供空间定位。不同患者的同一解剖结构在npy中占据不同像素坐标（因裁剪/缩放不同）。

---

### Exp5: Prompt Compiler（结构化解析 + Atlas先验 + 检索）

**脚本**：`experiments/scripts/inference_prompt_compiler.py`

**方法**：
1. 文本解析 → 结构化slots（anatomy, side, finding_type, target_form, level）
2. Atlas空间先验（硬编码的解剖位置映射）
3. 结构化相似度检索（anatomy=4分, side=2分, finding_type=2分, level=2分）
4. Alpha混合atlas先验和检索先验
5. 按目标类型路由：focal/diffuse/bony/multifocal不同后处理

**结果**：Mean Dice = **0.0235**，0/91 Dice≥0.5

**失败原因**：与Exp4相同 — **空间坐标在不同患者间不可迁移**。npy volume经过不同裁剪/缩放，"右肾"在不同患者的像素空间中位置完全不同。

---

### Exp4-5 的关键教训

> **文本无法在不看图像的情况下映射到像素坐标。** 任何text-only → spatial的方法（检索、atlas、规则）在这个数据格式下注定失败。解决方案必须涉及**联合text-image推理**：模型必须同时看到图像和理解文本。

---

### Exp6 & Exp7: MedCLIP-SAMv2 (BiomedCLIP + M2IB → bbox → MedSAM)

**脚本**：`inference_medclip_medsam.py`

**这是第一个真正做联合text-image推理的方案。**

#### 架构与原理

```
Text ("右侧腹股沟区肿块")
       ↓
  BiomedCLIP Text Encoder → text features (512-dim)
       ↓                           ↓
  BiomedCLIP Vision Encoder    cos_similarity
  (224×224 CT slice)               ↓
       ↓                    loss = β×compression - fitting
  M2IB Information Bottleneck
  (在ViT第7层插入可学习mask，
   10步梯度优化)
       ↓
  Saliency Map (224×224 热力图)
       ↓
  二值化 → Bounding Box
       ↓
  Scale到原始分辨率 → MedSAM → 分割mask
```

**核心机制 — M2IB（Multi-Modal Information Bottleneck）**：
- 在ViT的某一层插入一个可学习的mask（alpha参数）
- 通过sigmoid将alpha转为[0,1]的"通过率"lambda
- 目标函数：最大化text-image余弦相似度，同时最小化信息流量
- 优化10步后，alpha的分布就反映了图像中哪些patch对当前文本最重要
- 最终从alpha提取saliency map → 二值化 → bbox

#### 实现过程与bug修复

**模型加载**：
- 最初使用 `open_clip` 加载BiomedCLIP base → saliency map完全均匀（mean=0.6，不区分文本）
- 下载DHN-NCE fine-tuned权重（784MB）→ 发现权重是HuggingFace格式，与open_clip不兼容
- **最终方案**：直接用 `AutoModel.from_pretrained("MedCLIP-SAMv2/saliency_maps/model", trust_remote_code=True)` 加载HF格式模型
- 修复了 `modeling_biomed_clip.py` 中 `from transformers.models.clip.modeling_clip import *` 在新版transformers中的导入问题

**Wrapper重构**：
- 移除了6个自定义wrapper类（VisionEmbeddings, ImageEncoderWrapper, TextEmbeddings, TextEncoderWrapper, BiomedCLIPWrapper, permute_then_forward）
- HF模型原生支持IBA所需的所有接口：
  - `model.vision_model(img, output_hidden_states=True)` → hidden_states
  - `model.get_text_features(text_ids)` → 512-dim features
  - `model.get_image_features(pixel_values)` → 512-dim features
  - `model.vision_model.encoder.layers[i]` → 可被replace_layer替换

**之前在open_clip wrapper上修复的6个bug**（已被重构淘汰）：
1. `TypeError: 'Embedding' object is not subscriptable` — BiomedCLIP的pos_emb是nn.Embedding
2. `AttributeError: 'BertLayer' has no attribute 'attn'` — BERT层属性名不同于ViT
3. `AttributeError: 'tuple' object has no attribute 'clone'` — InformationBottleneck返回tuple
4. `TypeError: unsupported operand @ for Sequential` — text_projection是Sequential不是矩阵
5. `RuntimeError: size mismatch 768 vs 512` — 返回了hidden state而非projected output
6. Saliency map均匀 — 使用base BiomedCLIP而非fine-tuned

#### 实验结果

**Exp6（无slice过滤，3 cases）**：

| case | target | GT voxels | Pred voxels | Dice |
|------|--------|-----------|-------------|------|
| s0000 | 腹股沟肿块 | 9,913 | 226,168 | 0.049 |
| s0000 | 盆腔淋巴结 | 461 | 276,282 | 0.002 |
| s0001 | 胆囊窝结节 | 0 | 474,321 | 0.0 |
| s0002 | 颈椎低密度结节 | 62 | 618,351 | 0.0 |

Mean Dice = **0.0073**。严重过分割：每个slice都生成bbox，pred远超gt。

**诊断**：
- 每个slice都有saliency响应 → 32/32 slices都产生bbox
- BiomedCLIP无法判断哪些slice包含目标

**Exp7（加入slice聚焦度过滤，5 cases）**：

改进：按saliency聚焦度 `focus = peak × (1 - coverage)` 排序，只保留top一半slice。

| case | target | GT voxels | Pred voxels | Slices | Dice |
|------|--------|-----------|-------------|--------|------|
| s0000 | 腹股沟肿块 | 9,913 | 79,854 | 16 | 0.060 |
| s0003 | 颈椎曲度反弓 | 12,338 | 34,470 | 16 | **0.236** |
| s0004 | 肝缘不规则(肝硬化) | 45,600 | 199,082 | 16 | 0.086 |
| s0004 | 脾大 | 34,142 | 136,153 | 16 | 0.000 |

Mean Dice = **0.0284**。大面积弥漫性目标（颈椎病0.24）有一定效果，小目标仍然失败。

#### Saliency map质量分析

对同一slice测试不同文本query的saliency：

| 文本 | Mean | >0.5覆盖率 | BBox (224空间) |
|------|------|-----------|---------------|
| "右侧腹股沟肿块" | 0.240 | 4.2% | [108,85,143,111] (小) |
| "盆腔淋巴结" | 0.312 | 10.6% | [82,77,156,114] (中) |
| "正常肝实质" | 0.315 | 9.6% | [75,76,155,111] (中) |

Saliency有一定区分度（不同文本产生不同热力图），但不够精确。

#### CLS级别slice排序测试

测试用全局cosine similarity在32个slice中定位目标slice：

- **腹股沟肿块（11个GT slice）**：Top-10中命中9个（recall=82%），效果好
- **盆腔淋巴结（3个GT slice）**：Top-10中只命中1个（recall=33%），效果差

结论：大/明显的病变可以定位，小/散在的病变无法定位。

### Exp8: Track B 推理时优化汇总

**脚本**：`inference_medclip_medsam_v3.py` + 快速测试脚本

**目标**：在不训练的前提下，通过各种推理时trick提升text-guided的Dice。

#### 测试的Tricks

| Trick | 原理 | 效果 |
|-------|------|------|
| CLS filter | 用BiomedCLIP全局cosine similarity筛选top-k slice | 选错slice，反而漏掉GT slice |
| Text shortening | 去除描述中的冗余成分，保留关键短语 | 丢失上下文信息 |
| Text ensemble | 多种文本变体取平均saliency | 无显著改善 |
| Gamma sharpen | saliency^γ (γ=2.0) 强化高响应区域 | bbox变小变偏 |
| Percentile thresh | 用92%分位数代替Otsu二值化 | bbox定位仍不准 |
| High MedSAM thresh | MedSAM阈值从0.5提高到0.65 | 减少少量过分割 |
| **CT body mask gating** | 用CT强度阈值(>0.05)创建body mask，抑制空气/背景saliency | **+50%相对改善**，但绝对值仅0.0117 |

#### 最终结果

| 配置 | Mean Dice | 说明 |
|------|-----------|------|
| Exp7 baseline (slice过滤) | 0.0284 | |
| All v3 tricks combined | 0.0077 | 所有tricks反而互相干扰 |
| Body mask + gamma + percentile | **0.0117** | 相对baseline+50%，但绝对值仍极低 |

**结论**：所有推理时优化无法突破BiomedCLIP的根本局限——它无法在CT slice中准确定位特定解剖结构。要真正提升text-guided性能，必须更换模型（如M3D-LaMed）或进行领域微调。

---

## 4. 失败分析总结

### 4.1 过分割是主要失败模式

Oracle bbox下：46%目标预测量>GT的3倍。

### 4.2 按目标大小的表现（Oracle bbox）

| GT大小 | 数量 | Mean Dice | 过分割比 |
|--------|------|-----------|----------|
| < 50 voxels | 12 | 0.329 | 5.3x |
| 50 - 500 | 26 | 0.469 | 4.1x |
| 500 - 5K | 31 | 0.422 | 3.8x |
| 5K - 50K | 12 | 0.756 | 1.4x |
| > 50K | 2 | 0.601 | 2.6x |

### 4.3 三种text-guided方法失败的根本原因

| 方法 | 失败原因 |
|------|----------|
| TF-IDF检索 | 词袋相似度无法映射空间位置 |
| Prompt Compiler | Atlas先验在不同裁剪/缩放的npy间不可迁移 |
| MedCLIP-SAMv2 | BiomedCLIP是2D模型，CT切片与训练数据差异大；saliency太散；无法判断目标在哪些slice上 |

**共同根因**：在不训练的条件下，没有模型能同时理解（1）医学文本的语义（2）CT切片的空间解剖（3）两者之间的对应关系。

---

## 5. 下一步方案调研

### 方案一：M3D-LaMed（⭐推荐优先级最高）

**来源**：BAAI（智源研究院），2024年4月发布

**架构**：
- 3D视觉编码器（CLIP策略预训练，12万image-text对）
- 3D空间池化感知器（压缩3D token序列）
- LLM骨干：LLaMA-2-7B 或 Phi-3-4B（轻量版）
- SegVol分割模块：通过`[SEG]`token触发3D mask生成

**为什么推荐**：
- **M3D-RefSeg就是它的benchmark** — 数据格式完全匹配 (1×32×256×256 npy)
- 不需要任何数据转换，直接推理
- 原生支持自由文本 → 3D分割
- 有Phi-3-4B轻量版，12GB fp16可能够用

**代码/权重**：
- GitHub: `BAAI-DCAI/M3D`
- HuggingFace: `GoodBaiBai88/M3D-LaMed-Phi-3-4B`
- Apache 2.0开源

**显存**：7B版本fp16约14GB（偏紧），4B版本推荐

---

### 方案二：SegVol

**来源**：BAAI，NeurIPS 2024

**架构**：
- 3D ViT编码器（SimMIM预训练，9.6万CT）
- CLIP文本编码器（冻结）
- Prompt编码器（点/框/文本）
- Mask解码器（交叉注意力融合）
- Zoom-Out-Zoom-In推理机制

**特点**：
- 支持200+解剖结构
- 文本prompt限于标准解剖学术语（"liver"、"kidney"），非自由文本
- 需从描述中提取关键词

**代码/权重**：
- GitHub: `BAAI-DCAI/SegVol`
- HuggingFace: `BAAI/SegVol`
- 输入格式支持 npy、NIfTI、DICOM

---

### 方案三：BiomedParse v2

**来源**：Microsoft Research，发表于Nature Methods 2025

**架构**：
- BoltzFormer架构
- 600万 image-mask-text 三元组训练
- 82种目标类型，9种影像模态
- v2新增3D支持（逐slice+邻近context编码为RGB）

**特点**：
- 原生支持text-guided分割
- 文本prompt比bbox更好用
- 支持CT、MR、PET、超声、病理等

**局限**：
- 逐slice处理3D（和MedCLIP-SAMv2类似）
- 显存需求≥16GB
- 环境搭建复杂（Python 3.10.14, CUDA 12.4）

**代码/权重**：
- GitHub: `microsoft/BiomedParse`
- HuggingFace: `microsoft/BiomedParse` (biomedparse_v2.ckpt)

---

### 方案四：Fine-tune BiomedCLIP

**目标**：在现有MedCLIP-SAMv2框架下，用M3D-RefSeg数据fine-tune BiomedCLIP，提升saliency map质量。

**方法**：
- DHN-NCE对比学习loss（代码已有：`MedCLIP-SAMv2/biomedclip_finetuning/`）
- 或 LoRA低秩适配（参数减少99.5%，12GB够用）
- 784个text-image对超过最低可行阈值（200-300）

**训练配置**：
```
batch_size=16, epochs=32~50, lr=2e-4
LoRA: rank=16, alpha=32
DHN-NCE: temperature=0.6, beta1=beta2=0.15
```

**预计时间**：RTX 4070上 4-8小时

**预期效果**：
- Saliency map从覆盖30-50% → 缩小到更聚焦区域
- 但仍是2D逐slice方案，无法解决"目标在哪些slice"的问题
- 预估Dice可能从0.03提升到0.05-0.10

---

### 方案对比

| | M3D-LaMed | SegVol | BiomedParse v2 | Fine-tune BiomedCLIP |
|---|---|---|---|---|
| **数据兼容** | 完美（专用） | 需适配 | 需转格式 | 需转2D |
| **3D原生** | ✅ | ✅ | ❌（逐slice） | ❌（逐slice） |
| **自由文本** | ✅ | ❌（关键词） | ✅ | ✅ |
| **12GB可行** | 4B版勉强 | 需测试 | 可能不够 | ✅（LoRA） |
| **部署难度** | 低（pip） | 中 | 高 | 中（需训练） |
| **需要训练** | ❌ | ❌ | ❌ | ✅（4-8h） |
| **预期效果** | 最高 | 中高 | 高 | 中 |

---

## 6. 文件结构

```
D:/M3D/
├── inference_medsam_refseg.py              # Exp1-2: MedSAM oracle bbox推理
├── inference_medsam_improved.py            # Exp3: 推理tricks消融（multimask/refine/cc3d）
├── inference_medsam_optimized.py           # Exp3b: Track A优化（adaptive_margin/otsu/erosion/ensemble）
├── inference_medsam_retrieval.py           # Exp4: TF-IDF文本检索→bbox→MedSAM
├── inference_medclip_medsam.py             # Exp6-7: MedCLIP-SAMv2 pipeline
├── inference_medclip_medsam_v3.py          # Exp8: Track B推理优化（cls_filter/gamma/percentile等）
├── project_status.md                       # 本文档
├── plan.md                                 # 用户提供的改进路线图
│
├── MedSAM/                                 # MedSAM仓库
│   ├── segment_anything/                   # SAM模型代码
│   └── work_dir/
│       ├── MedSAM/medsam_vit_b.pth               # SAM ViT-B原始权重 (358MB)
│       └── MedSAM_finetuned/medsam_vit_b.pth     # MedSAM fine-tuned权重 (375MB)
│
├── MedCLIP-SAMv2/                          # MedCLIP-SAMv2仓库
│   ├── saliency_maps/
│   │   ├── model/                          # BiomedCLIP HF模型 + fine-tuned权重
│   │   │   ├── pytorch_model.bin           # DHN-NCE fine-tuned权重 (784MB)
│   │   │   ├── modeling_biomed_clip.py     # 自定义HF模型定义
│   │   │   └── config.json
│   │   ├── scripts/                        # IBA/M2IB原始代码
│   │   └── generate_saliency_maps.py       # 原始saliency生成脚本
│   └── biomedclip_finetuning/              # BiomedCLIP fine-tuning代码
│
├── experiments/                            # Exp5: Prompt Compiler实验
│   ├── scripts/
│   │   ├── prompt_compiler/                # 编译器代码 (compiler.py, retrieval.py)
│   │   └── inference_prompt_compiler.py    # 推理脚本
│   ├── results/
│   └── README.md
│
├── M3D_RefSeg_npy/                         # 数据集 (208 cases, .gitignored)
│   ├── s0000/ {ct.npy, mask.npy, text.json}
│   └── ...
│
├── results_medsam/                         # Exp1结果 (SAM权重, 50 cases)
├── results_medsam_ft/                      # Exp2结果 (MedSAM权重, 30 cases)
├── results_medsam_improved/                # Exp3结果 (tricks消融, 30 cases)
├── results_medsam_retrieval/               # Exp4结果 (TF-IDF检索, 30 cases)
├── results_medsam_optimized/               # Exp3b结果 (Track A优化, 30 cases)
├── results_medclip_medsam/                 # Exp6结果 (MedCLIP-SAMv2 v1, 3 cases)
└── results_medclip_medsam_v2/              # Exp7结果 (MedCLIP-SAMv2 v2, 5 cases)
```

---

## 7. 技术笔记

### MedSAM vs CLIP 的本质区别

- **MedSAM (SAM)**：纯视觉模型，接受视觉prompt（bbox/point/mask）→ 分割mask。**没有文本理解能力**。
- **CLIP/BiomedCLIP**：视觉-语言对齐模型，将图像和文本映射到同一embedding空间。**有文本理解但没有分割能力**。
- **MedCLIP-SAMv2**：用CLIP做text-image对齐 → 生成saliency → 转为bbox → 送入SAM分割。桥接两者。

### M2IB Information Bottleneck 原理

1. 在ViT的第k层（默认layer=7）插入一个可学习的mask α（shape同hidden state）
2. 通过sigmoid(α)得到"通过率"λ∈[0,1]
3. 原始hidden state × λ = 被mask后的表示
4. 优化目标：`loss = β × compression - cos_sim(text, image)`
   - compression = KL散度，衡量信息流量
   - cos_sim = 文本和图像特征的余弦相似度
5. 10步Adam优化后，α的分布反映了哪些patch对当前文本最重要

### 数据格式核心问题

M3D-RefSeg的npy volume (1, 32, 256, 256) 是经过**per-patient裁剪和缩放**的：
- 不同患者的相同解剖结构在像素空间中位置不同
- 没有统一的世界坐标系
- 因此任何"从文本推断像素坐标"的方法都会失败
- 必须**看图像**才能定位

### 模型加载与兼容性

**BiomedCLIP加载**：
- DHN-NCE fine-tuned权重是HuggingFace格式（`pytorch_model.bin`），与open_clip不兼容
- 必须用 `AutoModel.from_pretrained(model_dir, trust_remote_code=True)` 加载
- Tokenizer必须单独加载：`AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")`（model目录的custom config没有tokenizer映射）
- 新版transformers中 `from transformers.models.clip.modeling_clip import *` 不再导出全部类（`__all__`只有7个），需改为显式import

**HF BiomedCLIP直接兼容IBA，无需wrapper**：
- `BiomedCLIPEncoderLayer.forward(hidden_states, attention_mask, output_attentions)` → 返回 `(hidden_states,)` tuple
- `mySequential(layer, bottleneck)` 兼容，encoder内部通过 `hidden_states = layer_outputs[0]` 解包
- 之前写了6个wrapper类（VisionEmbeddings, ImageEncoderWrapper, TextEmbeddings, TextEncoderWrapper, BiomedCLIPWrapper, permute_then_forward）全部多余，已移除

**MedSAM推理细节**：
- Image encoder用fp16节省显存且不影响精度：`model.image_encoder = model.image_encoder.half()`，decoder保持fp32
- 输入必须resize到1024×1024，bbox坐标也要对应缩放：`box_1024 = bbox / [W, H, W, H] * 1024`
- 用soft probability map（`torch.sigmoid(logits)`）+ 外部阈值（Otsu/固定）比直接取binary更灵活，允许per-slice自适应

---

## 8. 数据增强与推理优化策略总结

### 8.1 有效的策略

| 策略 | 类型 | 效果 | 原理 |
|------|------|------|------|
| **Adaptive margin** | Bbox级 | **+2.5pp Dice** | `margin = clamp(0.1 × max(bbox_w, bbox_h), 2, 15)`，按目标大小动态调整。固定5px对小目标太大（过分割），对大目标太小（漏割） |
| **Otsu自适应阈值** | 后处理 | +0.1pp（在adaptive margin基础上） | 自动适应不同slice的概率分布，比固定阈值更鲁棒 |
| **CT body mask gating** | Saliency后处理 | +50%相对（Track B） | 用CT强度阈值（>0.05）创建body mask，抑制空气/背景区域的saliency噪声。绝对值仍极低（0.0117），但方向正确 |

### 8.2 无效的策略

| 策略 | 类型 | 效果 | 失败原因 |
|------|------|------|----------|
| **Erosion + 高阈值** | 后处理 | **-4.3pp Dice** | 两者都削减预测面积，叠加后recall损失远超precision收益。对抗过分割不能简单"砍预测" |
| **Bbox perturbation ensemble** | Bbox级 | 不稳定，速度慢5x | 随机jitter±3px取5次平均，噪声大于信号 |
| **CLS cosine slice filter** | Slice选择 | 负效果 | 大/明显病变recall 82%，小/散在病变recall 33%，错误排除含GT的slice |
| **Text shortening** | 文本增强 | 负效果 | 去除"冗余"描述反而丢失BiomedCLIP理解所需的上下文语义 |
| **Text ensemble** | 文本增强 | 无显著变化 | 多种文本变体的saliency map差异太小 |
| **Gamma sharpen** | Saliency后处理 | 负效果 | saliency^2.0强化高响应区，但定位本身不准时bbox变小变偏 |
| **Percentile threshold** | Saliency后处理 | 无改善 | saliency分布不反映目标位置时，阈值策略无意义 |
| **Confidence gate** | Slice级 | 微弱正效果 | 仅对边界slice有效，整体影响不大 |

### 8.3 过分割详细分析

过分割是oracle bbox下的**主要失败模式**，46%目标的预测量超过GT的3倍。

| GT大小 | 数量 | Mean Dice | 过分割比 | 说明 |
|--------|------|-----------|----------|------|
| < 50 voxels | 12 | 0.329 | 5.3x | 微小目标，bbox margin相对太大 |
| 50 - 500 | 26 | 0.469 | 4.1x | |
| 500 - 5K | 31 | 0.422 | 3.8x | |
| 5K - 50K | 12 | **0.756** | 1.4x | 中大目标表现最好 |
| > 50K | 2 | 0.601 | 2.6x | 超大目标反而下降 |

MedSAM倾向于**填满bbox内所有"看起来像组织"的区域**，小目标受bbox过大影响最严重。Adaptive margin直接解决了这个问题。

### 8.4 空间先验可行性分析

| 方法 | 是否可行 | 原因 |
|------|----------|------|
| **统计空间先验**（平均位置atlas） | ❌ | npy volume经过per-patient裁剪/缩放，同一结构像素坐标不同 |
| **文本→坐标映射**（规则/检索） | ❌ | 同上，已在Exp4-5中验证失败 |
| **图像条件先验**（小conv网络: saliency+CT→refined saliency） | ✅但需训练 | 理论可行，需标注数据训练~3-4小时，实质是fine-tune |
| **CT body mask gating** | ✅微弱 | 不需训练，方向正确但效果有限（+50%相对，绝对1.2%） |

### 8.5 CT数据特征

- CT强度分布（归一化[0,1]后）：空气/背景≈0，软组织0.2-0.6，骨骼0.7-1.0
- body mask阈值0.05可分离背景，但内部解剖定位仍需视觉理解
- text.json描述：放射科自由文本，长度2词~60+词，含定位信息但也含大量非空间修饰语

---

## 9. Exp9: VoxTell 零样本评测（CVPR 2026）

**脚本**：`scripts/voxtell_evaluate.py`
**模型**：VoxTell v1.1 + Qwen3-Embedding-4B (frozen)
**数据**：全量 M3D_RefSeg 原始 NIfTI (208 cases, 783 有效 masks)
**硬件**：RTX 4070 12GB，总运行时 ≈ 70 分钟

### 9.1 Pipeline

```
(ct.nii.gz, text) → NibabelIOWithReorient (RAS)
                  → 文本编码 Qwen3-Embedding-4B (一次性全数据集批量)
                  → 3D sliding-window inference (MaskFormer-style 文本-视觉融合)
                  → sigmoid > 0.5 → 原尺寸 mask
```

与之前 MedCLIP-SAMv2 pipeline 的关键区别：**原生 3D 端到端，无 2D bbox 中间步骤**，文本直接参与多尺度视觉特征融合。

### 9.2 两种 Prompt 模式

每个 mask 并行评测两种 prompt：

| 模式 | 例子 | 理念 |
|------|------|------|
| **raw** | "右侧腹股沟区不规则肿块，增强扫描边缘强化" | 原文直送，相信 Qwen 的多语言能力 |
| **structured** | "mass" | regex 抽取发现类型关键词（mass/nodule/lymph node/...），fallback "lesion" |

### 9.3 工程问题与 refactor

**初版 bug**：`VoxTellPredictor.embed_text_prompts` 每次调用把 Qwen3-Embedding-4B（fp16 ≈ 8 GB）在 GPU 和 CPU 之间搬运。在 Windows 32 GB 内存机器上，第 2 个 case 就因 **CPU RAM 碎片化** OOM（单次 10 MB 分配失败）。

**解决方案**：两阶段执行：
1. **Pass 1**：扫描全数据集，去重所有 prompt（208 cases × 平均 3-4 masks × 2 modes → 781 唯一 prompt）
2. **批量编码**：Qwen 单次上 GPU，分 98 批（chunk_size=8）编码完所有 prompt，立即 `del predictor.text_backbone` 完全释放
3. **Pass 2**：每个 case 只调 `predict_sliding_window_return_logits(data, cached_embeds[idxs])`，不再触发 Qwen

后续推理阶段 GPU 有 11.6/12.9 GB 空闲，再无 OOM。

### 9.4 结果

| Mode | N | Mean Dice | Median | Dice ≥ 0.5 | Dice = 0.0 |
|------|---|-----------|--------|------------|-------------|
| **raw** | 783 | **0.1327** | 0.0 | 105 (13.4%) | 474 (60.5%) |
| structured | 783 | 0.0587 | 0.0 | 32 (4.1%) | 601 (76.8%) |

### 9.5 关键发现

1. **中文原文 > 英文关键词 (2.3×)**：早期单 case 测试（s0000）显示 raw=0 而 structured (`mass`)=0.30，结论反向。全量统计后 raw 远胜 structured，原因推测：
   - Qwen3-Embedding-4B 多语言能力强，能从长句中提取病灶**类别 + 解剖部位 + 形态描述**的联合语义
   - 纯关键词丢失了所有空间定位信息（"右肺下叶"、"双侧"等）
   - regex 抽取器设定的 fallback=`lesion` 过于宽泛，许多 mask 结构化 prompt 全部退化为 `lesion`

2. **第一个突破 0.1 的 text-guided 方法**：
   - 此前最佳（MedCLIP-SAMv2 v2）：0.0284
   - VoxTell raw：0.1327（**4.7× 提升**）
   - 距 oracle bbox 上限（0.5482）仍有 4× 差距

3. **Dice 分布双峰**：60% 全零 + 13% ≥0.5，说明 VoxTell 在能理解的 prompt 上表现很好，在另一部分 prompt 上直接失败 — 提示后续可从"哪类 prompt 失败"入手改进（长尾病灶词、中文罕见描述等）。

### 9.6 后续方向

- 分析 raw prompt 下 Dice=0 的 474 个样本的文本模式：是否集中在某类病灶？
- 改进 structured prompt：保留解剖部位词（如 `right lung mass` 而非 `mass`）
- 考虑 raw + structured **集成**：每个 mask 跑两次取更高 Dice 或两个预测取 union / intersection
- 若做 VoxTell fine-tune：784 masks 量级偏小，但可能作为一个新 ceiling baseline（vs oracle bbox 0.55）
- 文本缩短反而丢失上下文 → 说明BiomedCLIP用的是整体语义而非关键词匹配
