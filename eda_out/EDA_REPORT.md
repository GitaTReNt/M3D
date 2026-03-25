# M3D-RefSeg EDA 报告（由脚本自动生成）

本报告基于 `eda_out/tables/` 中的统计结果自动汇总。
如果你在 `eda_out/figs/` 下也有对应的图，本报告会引用它们的文件名，方便你在本地打开查看。

## 1. 术语小词典（给初学者）

- **EDA (Exploratory Data Analysis)**：探索性数据分析。目的是在建模前了解数据分布、质量问题、潜在泄漏/偏差。
- **case（病例）**：一个 CT 体数据（例如 `s0001/ct.npy`），通常对应一个病人/一次扫描。
- **ROI (Region of Interest)**：感兴趣区域。这里指某个需要被分割的目标区域。
- **Mask_ID**：mask 中的整数标签 id。当前样本要分割的 ROI = `(mask == Mask_ID)`。
- **region（分割目标单元）**：一个 `case_id + Mask_ID` 组合。它比 CSV 的“行”更接近独立样本。
- **paraphrase（改写/同义问法）**：同一个 region 对应多条不同问句，用来增强提示词（prompt）的多样性。
- **data leakage（数据泄漏）**：同一个 region 同时出现在 train 和 test（哪怕问句不同），会导致评估虚高。
- **laterality（侧别）**：left/right/bilateral（左/右/双侧）等方位线索。
- **uncertainty cue（不确定性措辞）**：如 possible/likely/consider 等“推断语气”。
- **negation（否定）**：如 no/without/negative for 等否定表达。

## 2. 数据规模总览

- QA 行数：**2777**
- 病例数（case）：**193**
- region 数（独立分割目标）：**556**

**Question_Type 分布（问句风格）**：
- Type 0: 1665 行（60.0%）
- Type 1: 1112 行（40.0%）

- Answer 含 `[SEG]` 的比例：**98.78%**
  - 未包含 `[SEG]` 的行数：**34**（建议清洗/过滤）

- Question 命中“不确定性措辞”的比例（启发式）：**40.33%**
- Question 命中“否定措辞”的比例（启发式）：**1.15%**

## 3. 文本长度分布（Question / Answer）

- Question 词数：mean=15.34, median=15, p95=23, min=6, max=36
  - 图：`figs/question_words_hist.png`
- Answer 词数：mean=18.12, median=17, p95=32, min=4, max=57
  - 图：`figs/answer_words_hist.png`

## 4. Question_Type 对比（不同问句风格）

Type 0 通常更像“直接分割/定位”，Type 1 更像“解释/推断 + 分割”。

|   Question_Type |    n |   q_words_mean |   q_words_median |   a_words_mean |   a_words_median |   unc_rate |   neg_rate |   has_seg_rate |
|----------------:|-----:|---------------:|-----------------:|---------------:|-----------------:|-----------:|-----------:|---------------:|
|               0 | 1665 |        13.3922 |               13 |        14.0619 |               13 |   0.202402 |  0.0108108 |       0.988589 |
|               1 | 1112 |        18.2617 |               18 |        24.1996 |               23 |   0.704137 |  0.0125899 |       0.986511 |

## 5. 模板化（Template bias）初步观察

通过统计问句前三个词（prefix），可以看到大量问句来自固定模板（例如 *can you segment* / *are there any*）。
这意味着：prompt 多样性可能有限；模型可能学到“模板”，而不是更强的语言理解。

| prefix(first 3 words)   |   count |
|:------------------------|--------:|
| can you segment         |     433 |
| are there any           |     174 |
| can you identify        |     172 |
| please segment where    |     130 |
| please segment the      |     126 |
| where can we            |     118 |
| where is the            |     117 |
| could you identify      |     115 |
| please identify and     |     113 |
| is there any            |     111 |
| where are the           |      56 |
| can you locate          |      50 |
| where does the          |      43 |
| which part of           |      40 |
| based on the            |      38 |
图：`figs/question_type_bar.png`、`figs/question_anatomy_group_bar.png`（如果你生成了前缀图，也可自行补充）

## 6. region 级别分析（更接近“独立分割样本”）

**每个 region 对应多少条 QA（paraphrase 数）**：
- 4 条/region：3 个 region
- 5 条/region：553 个 region
图：`figs/qa_rows_per_region_hist.png`

**每个 case 有多少个 region（同一病例内多目标）**：
- mean=2.88, median=2, max=11
- 至少 2 个 region 的 case：129/193
- 至少 3 个 region 的 case：88/193
这支持构造“multi-finding 报告”（把同一病例多个发现拼成一段长文本）用于更贴近真实报告场景。

**region 的粗粒度解剖分组（anatomy_group，启发式关键词）**：
| anatomy_group     |   count | pct   |
|:------------------|--------:|:------|
| pulmonary         |     121 | 21.8% |
| other             |     107 | 19.2% |
| renal_urinary     |      89 | 16.0% |
| cardio_vascular   |      77 | 13.8% |
| hepato_biliary    |      68 | 12.2% |
| musculoskeletal   |      50 | 9.0%  |
| gastro_intestinal |      21 | 3.8%  |
| reproductive      |      13 | 2.3%  |
| neuro_headneck    |      10 | 1.8%  |

**region 的侧别（laterality，来自 label_desc/问题文本的启发式）**：
| laterality   |   count | pct   |
|:-------------|--------:|:------|
| none         |     253 | 45.5% |
| left         |     130 | 23.4% |
| right        |     122 | 21.9% |
| bilateral    |      51 | 9.2%  |

## 7. 数据质量与清洗建议（本次输出可直接支持）

- 检测到 **34 条** Answer 不包含 `[SEG]`。这些样本往往更像“是/否问答”而不是分割指令，建议：
  1) 用于纯分割训练时 **过滤掉**；或
  2) 单独作为 VQA/分类子任务；或
  3) 统一修复 Answer（在开头补 `[SEG]`），但要谨慎：如果问题本身不是分割任务，修复会引入噪声。

## 8. 下一步建议（如果你还没跑 image-feature EDA）

如果你已经有 `ct.npy/mask.npy`（prepare 后版本），强烈建议补做：
- **ROI 体积（voxels）/ bbox 尺寸 / slices 覆盖**分布：小目标长尾会显著影响 Dice 和训练稳定性。
- **空 ROI 检测**：prepare 的 resize 可能让极小目标消失（mask==Mask_ID 变 0）。
- **连通域数量（components）**：多发病灶 vs 单发病灶。
- **质检 overlay**：随机/按异常点导出 CT+mask 叠加图。
这些结果会直接指导：采样策略（oversample small ROI）、loss（如 focal/Dice）、以及评估指标（Dice + Hausdorff）。
