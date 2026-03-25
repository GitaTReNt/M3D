下面我按“**新手入门**”的方式，把你这个 **M3D-RefSeg** 数据集讲清楚：**它是什么、里面有什么文件、标签到底是什么、CSV 里每一列代表什么、官方（Hugging Face README）怎么介绍、以及官方给的预处理规范**。

------

## 1) 这个数据集是干什么用的？

**M3D-RefSeg = 3D Medical Image Referring Segmentation Dataset**，核心任务是：

> 给你一段文字描述（referring expression / 指代表达），模型要在 **3D CT** 里把对应的区域 **分割出来**。

官方 README 的关键点是：

- 这是一个 **image-mask-text 三元组** 数据集；
- 他们从 **TotalSegmentator** 数据集中选了 **210 个 3D CT**，再对“文字描述 + 对应区域”进行了重新标注；
- 中文原始描述保存在 `text_zh.txt`，英文翻译保存在 `text.json`（用 Qwen 72B 做自动翻译）；
- 之后又用大模型把“区域描述”转成 **问答（QA）格式**，并保存成 CSV。 ([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))

它支持的任务不止“分割”，官方还提到可以用于 **定位/grounding（比如转 bbox）**：mask 可以转成 bounding box 坐标。 ([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))

------

## 2) 数据集里到底都有什么文件？

在 Hugging Face 仓库的 `Files` 里，官方提供了这些核心文件： ([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/tree/main))

- **`M3D_RefSeg.zip`（约 5.58GB）**：真正的数据主体（CT、mask、文本标注）。([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/tree/main))
- **三个 CSV**（QA 指令数据）
  - `M3D_RefSeg_all.csv`
  - `M3D_RefSeg_train.csv`
  - `M3D_RefSeg_test.csv` ([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/tree/main))
- **`README.md`**：官方说明（就是“官网介绍”的主要来源）([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))
- **`m3d_refseg_data_prepare.py`**：官方预处理脚本（把 nii.gz 统一处理成 npy + 固定体素大小）([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/tree/main))

### 2.1 zip 解压后每个病例（case）长什么样？

官方 README 给了目录结构：([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))

```
M3D_RefSeg/
  s0000/
    ct.nii.gz
    mask.nii.gz
    text.json
    text_zh.txt
  s0001/
    ...
```

也就是说：**每个 sXXXX 是一个 3D CT 病例**，并配套一个 `mask.nii.gz` 和文本描述。

------

## 3) “标签（Label）”到底是什么？

这里要非常明确：**这个数据集的标签不是一个全局分类标签（比如“肺/肝/肾”这种固定类）**，而是 **每个病例里多个“目标区域”的编号（Mask_ID）**。

### 3.1 标签存在哪里？

- `mask.nii.gz`：是一个 **3D 体数据**（与 CT 同空间），里面每个体素是一个数值
  - 0 通常表示背景
  - 1、2、3… 表示不同的“被标注区域”（region）
- `Mask_ID`：告诉你**当前这条样本要分割 mask 里的哪一个编号**（也就是取 `mask == Mask_ID`）。
   官方 README 给的 Dataset 示例代码就是这样二值化的：`seg_array = (seg_array == data["Mask_ID"])` ([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))

### 3.2 关键点：Mask_ID 是“病例内编号”，不是“全局语义类别”

你可以直接从 CSV 里看出来：

- `s0736` 的 `Mask_ID=1` 描述的是冠状动脉区域的异常；
- 但 `s0108` 的 `Mask_ID=1` 描述的是前列腺钙化。
   同一个 `Mask_ID=1` 在不同 CT 上语义不同，说明它不是“全局类别”。([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/raw/main/M3D_RefSeg_all.csv))

**因此：**

- 真正决定“这个 mask_id 对应什么语义”的，是这个病例的 `text.json / text_zh.txt` 或 CSV 的 `Question/Answer` 文本，而不是 `Mask_ID` 数字本身。

------

## 4) CSV 里有哪些“特征（Features/字段）”？每列是什么意思？

以 `M3D_RefSeg_all.csv` 的表头为准，它的列是： ([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/raw/main/M3D_RefSeg_all.csv))

- **`Image`**：CT 路径（相对于数据根目录），例如 `s0736/ct.nii.gz`
- **`Mask`**：mask 路径，例如 `s0736/mask.nii.gz`
- **`Mask_ID`**：本条样本要分割的目标区域编号（见上面标签解释）
- **`Question_Type`**：问题类型（0/1）
- **`Question`**：输入问题文本（指代表达/指令）
- **`Answer`**：答案文本，通常包含占位符 **`[SEG]`**

### 4.1 `Question_Type`（0/1）大概表示什么？

从 CSV 的样例行能看出一个明显区别：([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/raw/main/M3D_RefSeg_all.csv))

- **Type=0**：偏“直接提分割需求/定位需求”的问法
  - 例：`Can you segment ...?`
  - 例：`Where does the abnormality appear ...?`
- **Type=1**：偏“带解释/推理/病因描述 + 同时让你分割”的问法
  - 例：`Please segment and explain the potential cause ...`

（官方 README 没有对 0/1 做非常形式化的定义，但从样本文本分布可以这样理解。）

### 4.2 `[SEG]` 是什么？

**`[SEG]` 不是标签本身**，它只是答案里的一个“占位符”，表示“这里对应模型应该输出的分割区域”。
 真正的 GT segmentation 仍然来自 `mask.nii.gz` + `Mask_ID` 的二值化。官方示例代码是按这个逻辑来取 `seg` 的。([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))

------

## 5) 官方“预处理/输入规格”是怎样的？

官方在仓库里提供了 `m3d_refseg_data_prepare.py`，它做的事情很重要，因为后面很多模型/基线默认输入就是这个规格。

脚本里的关键步骤：([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/m3d_refseg_data_prepare.py))

1. 读取 NIfTI：`ct.nii.gz` 和 `mask.nii.gz`
2. 轴变换：`transpose(2, 0, 1)`（把维度顺序换掉）([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/m3d_refseg_data_prepare.py))
3. 强度归一化：把 CT 强度按 **0.5%~99.5% 分位** 映射到 **[0,1]**，并 clip ([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/m3d_refseg_data_prepare.py))
4. 前景裁剪：`CropForegroundd(keys=["image","seg"], source_key="image")` ([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/m3d_refseg_data_prepare.py))
5. resize 到固定大小：`spatial_size=[32,256,256]`（image 用 trilinear，seg 用 nearest）([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/m3d_refseg_data_prepare.py))
6. 保存为 `ct.npy` 和 `mask.npy`，并把 `text.json` 拷贝过去 ([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/m3d_refseg_data_prepare.py))

> 这也解释了为什么很多 M3D 相关代码里都写死了输入是 **`1×32×256×256`**（官方 README 里的 Dataset 示例也按这种 npy 来读）。([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))

------

## 6) “官网介绍”官方怎么说？（我帮你按要点梳理）

官方 README（Hugging Face 仓库的 `README.md`）核心信息可以概括为：([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))

- 数据集名：**3D Medical Image Referring Segmentation Dataset (M3D-RefSeg)**
- 规模：**210 个 3D 图像**、**2,778 个 masks**、以及文本标注（但 Hugging Face 的表格行数显示为 2,777 行；你会看到 README 的“2,778 masks”和自动统计行数存在轻微不一致，这是常见的小版本差异/统计口径问题）([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))
- 来源：从开源 **TotalSegmentator** 数据集中选子集，并重新标注文字与区域 ([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))
- 标注方式：
  - 医生写中文描述（`text_zh.txt`）
  - Qwen 72B 翻译成英文（`text.json`）
  - 大模型把描述转成 QA，存 CSV ([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))
- 支持任务：3D segmentation、3D positioning（grounding 等）([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))
- 下载方式：git clone / `datasets.load_dataset` / 手动下载 ([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))
- License：Apache-2.0（在 README 元数据里）([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/blob/main/README.md))

------

## 7) 你入门时建议怎么“理解一条样本”？

把一条 CSV 行理解成下面这个“学习样本”就对了：

- 输入（多模态）：
  1. 3D CT 体数据 `ct`（可能是 nii.gz，也可能预处理成 npy）
  2. 文本问题 `Question`（指令/指代表达）
- 输出（监督）：
  - 二值分割 mask：`GT = (mask_volume == Mask_ID)`
- 其他辅助信息：
  - `Answer` 里有 `[SEG]` 占位符（对应“分割区域”这个概念）
  - `Question_Type` 反映问法风格（更直接 vs 带解释）

------

如果你愿意，我下一步可以继续给你做一个“真正面向训练”的入门：

- 如何把它做成 **PyTorch/MONAI Dataset**（按官方脚本的 npy 结构）
- 如何做 **case-level split**（避免同一病例泄漏导致评估虚高）
- 以及最常用的指标（Dice/IoU）+ 可视化质检流程。















从你截图看，你本地的 `M3D_RefSeg/` 目录结构已经是**标准的 M3D-RefSeg 结构**：根目录下是一堆 `sXXXX/` 病例文件夹 + 一个总的 CSV（你这里叫 `M3D_RefSeg.csv`），每个病例文件夹里都有：

- `ct.nii.gz`（3D CT 体数据）
- `mask.nii.gz`（3D 分割标注，整型 label map）
- `text.json`（英文描述，按 label id 对应）
- `text_zh.txt`（中文描述 + 颜色等信息，ITK‑SNAP label 描述文件格式）

下面我按“入门”把这个数据集里**都有啥、标签是什么、特征是什么、官方怎么介绍**讲清楚，并结合你上传的一个病例文件做个具体例子。

------

## 1) 数据集里都有啥：两层结构

### 1.1 病例层（Case-level，存在每个 sXXXX/ 里）

每个 `sXXXX/` 都是一个 3D CT 病例，至少包含 4 个文件：

1. **`ct.nii.gz`（影像）**

- NIfTI 格式的 3D CT（通常强度单位是 HU，范围会覆盖空气到骨皮质等）。
- shape 是 `[X, Y, Z]` 或类似（具体取决于写入方向），但 **`ct` 和 `mask` 的 shape 应当一致**。

1. **`mask.nii.gz`（标注）**

- 也是 NIfTI 3D 体数据，但体素值是**整数标签**：
  - `0` 一般表示背景
  - `1,2,3,...` 表示该病例中不同“目标区域/异常/结构”的编号（注意：这是“病例内编号”）

1. **`text_zh.txt`（中文 label 解释，ITK‑SNAP label 描述文件）**

- 它不是普通“纯文本句子”，而是 ITK‑SNAP 的 label description 格式：每个 label id 附带 RGB 颜色、可见性、以及中文描述。
- 你上传的这个文件里明确写了 label 0~3 的中文含义。

1. **`text.json`（英文 label 解释）**

- key 是 `"1"`, `"2"`, `"3"`…（字符串形式的 id），value 是对应区域的英文描述。

> 这四个文件组合在一起，构成了“影像-标注-文本”的三元组（image-mask-text）。

------

### 1.2 样本层（Sample-level，存在 CSV 里）

根目录的 `M3D_RefSeg.csv`（你上传的这个）是**训练/评估时直接用的样本列表**。
 我读了你这个 CSV：一共 **2777 行、6 列**，列名是：

- `Image`：例如 `s0736/ct.nii.gz`
- `Mask`：例如 `s0736/mask.nii.gz`
- `Mask_ID`：整数（本行要分割的目标 label id）
- `Question_Type`：0/1（问题风格类别）
- `Question`：文本问题
- `Answer`：文本答案，通常包含 `[SEG]` 占位符（表示“这里对应分割结果”）

你可以把 CSV 的每一行理解成一个训练样本：
 **输入 = (CT + Question)**，**监督 = (mask==Mask_ID 的二值分割)**，Answer 只是把任务包装成“指令/问答”形式。

------

## 2) 标签（label）到底是什么？

这里最容易误解，我用一句话讲清：

**M3D‑RefSeg 的“标签”不是全局固定类别（不是“肝=1、脾=2”这种），而是“每个病例里若干个被描述的目标区域”的编号。**

- `mask.nii.gz` 里存的是多标签 label map
- `Mask_ID` 决定“这一条样本要分割 mask 里的哪个 id”
- `text_zh.txt / text.json` 给出 **这个病例中每个 id 的语义描述（中文/英文）**

### 用你上传的这个病例举例（非常直观）

你上传的 `text.json` 里只有 3 个 id（1~3），它们分别是：

- **1**：胆囊窝小结节高密度影 + 胆囊未显示 → 考虑胆囊切除术后表现
- **2**：右侧肾盂扩张向肾门延伸 → 考虑壶腹型肾盂
- **3**：右侧输尿管下段高密度小结节影 → 考虑小结石 + 上段输尿管扩张

`text_zh.txt` 也给了同样的中文版本，并附带颜色（方便在 ITK‑SNAP 里直接加载显示）。

同时，我检查了你上传的这个 `mask.nii.gz`：

- 里面的 label 值确实只有 **0,1,2,3**
- 且三个目标都很小（例如 label 3 只有几个体素），说明它更像“异常/病灶/小结构”的标注，而不是把整个人体器官都分出来的那种粗粒度分割。

> 所以：**“标签是什么” = 每个病例里 `mask` 的整数 id + text 文件里对应的那句医学描述。**

------

## 3) 这个数据集的“特征（features）”都有哪些？

从多模态角度，你可以把它的特征分成三类：

### 3.1 影像特征（Image features）

- 3D CT 体数据（NIfTI）：空间结构 + HU 强度分布
- 体素间距、shape、方向矩阵（affine）等元信息

### 3.2 分割监督特征（Segmentation supervision）

- `mask.nii.gz`：多标签 label map
- 训练时通常会把它变成二值：
   `GT = (mask == Mask_ID)`
   也就是把“这个病例中某个特定 id 的区域”作为当前样本的监督

### 3.3 文本/指令特征（Text/Instruction features）

- `text_zh.txt`：中文原始描述（医生标注的区域描述）
- `text.json`：英文翻译描述（同一个 id 的英文版本）
- CSV 里的 `Question/Answer`：把“描述”转成多种问法（指代分割/推理分割风格），并用 `[SEG]` 作为“分割结果”的占位符

另外，官方也提到：mask 可以很容易转换成 **3D bounding box**，从而支持定位/grounding 类任务。([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/raw/refs%2Fpr%2F3/README.md))

------

## 4) 官方（Hugging Face README）是怎么介绍的？

我按官方 README 的信息给你归纳一下（这部分就是你说的“官网介绍”）：([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/raw/refs%2Fpr%2F3/README.md))

1. **数据集定位**

- 3D Medical Image Referring Segmentation Dataset（M3D‑RefSeg）
- 目标任务是“给文本描述，让模型在 3D 医学图像中分割对应区域”（referring segmentation）

1. **规模（官方描述口径）**

- 官方写的是：**210 个 3D 图像、2,778 个 masks、以及文本标注**。([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/raw/refs%2Fpr%2F3/README.md))

  > 注意：这里的 “masks” 在他们的表述里更像“样本级标注/监督”的数量口径；你本地 CSV 里 2777 行也对应“2777 个训练样本”。

1. **数据来源与标注流程**

- 从开源的 **TotalSegmentator** 数据集中选了 210 个子集
- 医生用中文写原始描述（保存为 `text_zh.txt`）
- 用 **Qwen 72B** 自动翻译成英文（保存为 `text.json`）
- 再用大模型把区域描述转成**问答对（QA）**，存到 CSV。([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/raw/refs%2Fpr%2F3/README.md))

1. **支持的任务**

- 3D Segmentation（文本引导/指代分割等）
- 3D Positioning（visual grounding / referring expression comprehension 等）([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/raw/refs%2Fpr%2F3/README.md))

------

## 5) 官方推荐的预处理规范是什么？

官方给了一个脚本 `m3d_refseg_data_prepare.py`，核心做的是把原始 NIfTI 统一处理成他们训练时用的 `.npy` 规格：([Hugging Face](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg/resolve/main/m3d_refseg_data_prepare.py?download=true&utm_source=chatgpt.com))

- 读取 `ct.nii.gz` 和 `mask.nii.gz`
- 做轴变换：`transpose(2, 0, 1)`（把维度顺序换一下）
- CT 强度归一化：用 **0.5%~99.5% 分位**把强度缩放到 `[0,1]` 并 clip
- 以前景为依据裁剪（CropForeground）
- resize 到固定大小：`[32, 256, 256]`
  - image 用 trilinear
  - mask 用 nearest
- 保存为 `ct.npy` 和 `mask.npy`，并拷贝 `text.json`

这也是很多 M3D 系列代码默认输入是 `1×32×256×256` 的原因。

------

## 6) 你入门理解这个数据集时，建议抓住这 3 个“对应关系”

1. **一个病例 = 一个 ct.nii.gz + 一个 mask.nii.gz + 一套文本解释（json/zh）**
2. **CSV 的每一行 = 选定一个病例 + 选定一个 Mask_ID + 给一个问法**
3. **真正的分割 GT = (mask == Mask_ID)**
    文本答案里的 `[SEG]` 只是“提示模型要输出分割”的符号，不是标注本身。

------

如果你愿意，我下一步可以基于你这份本地数据，带你做“真正的探索式检查（EDA/QC）”，比如：

- 扫描所有病例：每个病例有多少个 label、每个 label 体积多大（小目标特别关键）
- 检查 `text.json` 的 id 是否都在 `mask` 里出现（避免标注/文本不一致）
- 把每个 label 的最大面积切片导出成 PNG（快速人工质检）
- 再决定训练时用原始 NIfTI 还是用官方 npy 预处理版本

你希望我先从哪一步开始：**全量统计**还是先做几个病例的**可视化质检**？







下面是一份**“一目了然、可直接写进 Proposal/PPT”的 EDA 报告**（面向初学者，专业名词都有解释），以及**进一步探索建议 + 完整可运行代码**。

> 说明：你上传的 `eda_out.zip` 里目前包含**文本/QA 与 region 级**的统计表与图（`dataset_text_summary.json / rows_text_features.csv / regions_text_summary.csv / region_paraphrase_counts.csv` 及对应分布图）。影像几何（ROI 体积/bbox/连通域等）、alignment 报告、overlay、影像特征 t-SNE 等文件**不在这个 zip 里**。下面报告会**严格基于 zip 内已有结果**，并在“进一步探索”里给你补齐“影像/ROI 特征 EDA”的建议与代码（你本地有 `M3D_RefSeg_npy/` 就能跑）。
>  同时，我会把结论对齐你们项目目标：**把 free-text report distill 成 grounded prompts 来提升分割**（你们 pitch 的动机）。
>  以及你们 notes 里提到的评估方向（Dice、Hausdorff 等）。
>  Proposal rubric 也明确要求：每种 EDA 方法给图/表、解释、以及“因此做出的决定”。

------

# EDA 报告：M3D-RefSeg（QA + Region 视角）

## 0) 名词速查（初学者友好）

- **Case / 病例**：一个 3D CT 扫描（文件夹 `sXXXX/`）。
- **Mask / 分割标注**：与 CT 同大小的 3D 标签图（每个体素是整数 id）。
- **Voxel / 体素**：3D 图像里的一个“像素”，是 3D 的最小单位。
- **ROI（Region of Interest）/ 目标区域**：我们要分割出来的那块区域（例如“右输尿管结石”）。
- **Mask_ID / label id**：mask 里某个 ROI 的整数编号。**注意：这是“病例内编号”，不是全局类别标签。**
- **QA 行（CSV 一行）**：一个“问题 + 回答 + 指向某个 ROI”的样本。
- **Region 单元**：把同一个 `(case_id, Mask_ID)` 视作一个“独立分割目标”。这比 QA 行更接近真正独立样本。
- **Prompt / 提示词**：这里主要指 `Question`（自然语言指令）。
- **Paraphrase / 改写**：同一个目标 ROI，用不同说法问 5 次（例如 “Can you segment…” / “Where is…”）。
- **Uncertainty / 不确定性**：文本里包含 “possible / consider / may / likely …” 等推断措辞。
- **Negation / 否定**：文本里包含 “no / without / negative …” 等否定措辞。
- **Laterality / 左右侧**：left/right/bilateral 等位置线索。
- **Anatomy group / 解剖粗类别**：用关键词把文本粗分到 lung/kidney/heart 等类别（是启发式，不是严格医学 NLP）。

------

## 1) 数据整体规模与结构（最重要的“现实”）

来自 `dataset_text_summary.json`（你生成的 EDA 输出）：

### 1.1 样本规模

- **QA 行数**：`2777`
- **病例数（case）**：`193`
- **Region 数（独立分割目标）**：`556`

> **关键理解**：2777 并不是 2777 个“独立分割目标”。真正更独立的单位是 **556 个 region**（`case_id + Mask_ID`）。

### 1.2 Question_Type 分布

- **Type 0**：1665（≈ 60.0%）
- **Type 1**：1112（≈ 40.0%）

直观解释（从文本规律看）：

- Type 0 更像“直接让你分割/定位”
- Type 1 更像“带解释/推断语气 + 分割”，更接近临床报告口吻（与你们 pitch 里“报告不是 clean prompt”的动机一致）。

### 1.3 `[SEG]` 占位符一致性

- Answer 含 `[SEG]` 的比例：**98.78%**
- 也就是说约 **34 条（1.22%）** Answer **没有 `[SEG]`**

**为什么重要？**
 如果你们后续用类似 M3D-LaMed 的机制：模型输出 `[SEG]` 触发分割头，那么缺 `[SEG]` 的样本会造成训练监督不一致。你们 pitch 里提到 M3D-LaMed 的流程里 `[SEG]` 是触发分割模块的关键 token。

------

## 2) “Region 重复改写”现象（决定你怎么划分 train/test）

来自 `region_paraphrase_counts.csv` 和 `qa_rows_per_region_hist.png`：

- **553 / 556 个 region 都恰好有 5 条 QA**
- 仅 **3 个 region** 只有 4 条（小缺失）

这意味着：**每个 ROI 基本都被“问 5 次”**（paraphrase augmentation）。

### 结论（Interpretation）

- 数据集把“同一 ROI”变成了“5 个不同问法”，因此**按行随机拆分**会把同一个 ROI 的不同问句分到 train/test，产生**数据泄漏（leakage）** → 评估会虚高。

### 决策（Decisions）

- **强烈建议**：至少做 **region-level split**（按 `case_id+Mask_ID` 分组），更严格的做 **case-level split**（按病例分组）。
- 评估时可以：
  - 每个 region 固定选 1 条问句做 deterministic eval；或
  - 5 条都跑，报告**均值+方差**（衡量 prompt 稳健性），这和你们 notes 里说的“生成很多 prompt 看是否稳定”一致。

------

## 3) 文本分布与噪声（“报告不是 clean prompt”的数据证据）

### 3.1 问题/回答长度（word count）

从 `rows_text_features.csv`（q_words/a_words）与直方图：

- **Question 词数**：中位数约 **15**（整体偏短、模板化较强）
- **Answer 词数**：中位数约 **17**（Type 1 会更长）

按 Question_Type 分组（你 EDA 输出可直接算出）：

- **Type 0**：
  - 平均 Question 长度：**13.39**
  - 平均 Answer 长度：**14.06**
  - Uncertainty 命中率：**20.24%**
- **Type 1**：
  - 平均 Question 长度：**18.26**
  - 平均 Answer 长度：**24.20**
  - Uncertainty 命中率：**70.41%**

### 结论

- Type 1 的“推断语气”非常强（uncertainty 很高），这可以作为你们 pitch 里“报告含不确定/推断信息”的**量化证据**。

### 决策

- 后续实验建议**分 Type 报告结果**（Type0 vs Type1），否则平均分会掩盖“报告式 prompt”下的性能退化。
- 可以做一个很有说服力的 robustness test：
  - **train on Type0，test on Type1**（prompt 风格 shift）

------

## 4) 模板化程度（Prompt 多样性有限的证据）

从你生成的 “前三词前缀”统计（我基于 `rows_text_features.csv` 计算）：

Top 前缀（占 QA 行比例）：

- `can you segment`：**15.6%**
- `are there any`：6.3%
- `can you identify`：6.2%
- `please segment where`：4.7%
- `please segment the`：4.5%
- ……

### 结论

- 问句模板化很明显：大量样本属于固定句式的轻微变体。

### 决策

- 把这 5 条问句看作“prompt augmentation”很合理。
- 但如果你要研究“自由文本报告 → grounded prompt”的价值，仅靠这些短模板问句可能不够逼真 → 建议下一步构造“合成长报告”（见后面的进一步探索建议与代码）。

------

## 5) 解剖与左右侧线索（决定你该用 Question 还是 label_desc）

### 5.1 从 Question 文本提取到的线索（启发式）

- **anatomy_group = other** 占 **47.8%**（说明很多问句根本没写具体器官名）
- laterality：`none` 占 **81.6%**（问句里经常不说 left/right）

### 5.2 从 Region 的 label_desc（更像报告/发现描述）提取到的线索

在 `regions_text_summary.csv`（region 级）：

- laterality（region 级）：
  - none：**45.5%**
  - left：**23.4%**
  - right：**21.9%**
  - bilateral：**9.2%**
- anatomy_group（region 级）：
  - pulmonary：**21.8%**
  - renal_urinary：**16.0%**
  - cardio_vascular：**13.8%**
  - hepato_biliary：**12.2%**
  - musculoskeletal：**9.0%**
  - other：19.2%
  - 其余更少

### 结论

- **Question** 经常不含器官/左右侧，信息密度不够；
- **label_desc**（来自 `text.json`）携带更多“报告式定位线索”，更适合做你们的 **Prompt Distillation**（你们 pitch 的方法图里就是把 report 蒸馏成结构化短 prompt）。
   一个例子（你给的病例）里，`text.json` 就是 label id → 英文报告式描述映射：
   中文 `text_zh.txt` 也是同一套 id 的描述（ITK-SNAP label file）。

### 决策

- 若你要做“grounded prompt”，**优先用 label_desc** 来抽取器官、左右侧、病灶类型；Question 主要用于“指令风格鲁棒性”测试。

------

## 6) 每个病例包含多少个目标（支持你构造 multi-finding report）

从 `regions_text_summary.csv` 汇总（region per case）：

- 平均每个 case 有 **2.88 个 region**
- 有 **129 / 193** 个病例至少 2 个 region
- 有 **88 / 193** 个病例至少 3 个 region
- 最多一个病例有 **11 个 region**

### 结论

- 数据天然支持你们 pitch 中强调的难点：**multi-finding**（同一报告里多个发现）。

### 决策（非常推荐写进 proposal）

- 构造一个更接近真实报告的设定：
   把同一病例多个 label_desc 拼成一段 “full report”（multi-finding），再要求模型分割其中某个 Mask_ID → 这就是“报告→单目标 ROI 的消歧”，与你们 research question 完全对齐。

------

## 7) 数据质量检查（轻量但非常必要）

### 7.1 Answer 缺 `[SEG]` 的样本

- 34 行缺 `[SEG]`，涉及 **24 个病例**（≈ 12.4% 病例至少出现一次）

### 决策

- 如果你做指令微调并依赖 `[SEG]` 触发分割：
  - 训练前把缺 `[SEG]` 的行**过滤掉**或**统一补 `[SEG]`**（最简单的数据清洗规则之一，proposal 很好写）。

------

# 与项目目标对齐的“下一步探索建议”（含对应代码）

你们 notes 里建议：先聚焦一个数据集/一个 body part/一个 aim，并且 t-SNE 可能慢但其他分布可以先做。
 结合这份 EDA，我建议下一步重点做 4 件事：

## A) Prompt Distillation 的“数据可行性检查”

目标：证明 label_desc 可以被抽取成 `{organ, laterality, pathology}` 这种 grounded prompt（你们 pitch 的 funnel）。
 做法：

- 正则/关键词抽取 organ、left/right、病灶词（nodule, dilation, calculus…）
- 输出 condensed prompt
- 检查 condensed prompt 覆盖率（有多少能抽出 organ/side）

## B) 构造 multi-finding synthetic report 数据集

目标：把数据变得更像你们研究问题里的“长报告 + 多发现 + 单 ROI 映射”。
 做法：

- 对每个病例，把所有 label_desc 拼成一个 report
- 针对病例里每个 Mask_ID，生成一条样本：输入是 report + “segment target X”（或再 distill 一下），输出仍然是对应 ROI mask

## C) Region/case 级拆分 + 泄漏检测（必须）

目标：保证评估可信
 做法：

- GroupKFold / GroupShuffleSplit：按 case_id 或 region_id 分组
- 输出 fold 划分文件，固定随机种子，确保可复现

## D) 影像/ROI 特征 EDA（你本地 `M3D_RefSeg_npy/` 可跑）

目标：支撑你们 notes 里的技术验证指标设计（Dice、Hausdorff）
 做法：

- 统计 ROI 体素数长尾、小目标比例、bbox 尺寸、连通域数量
- 检查 prepare 后是否出现 **ROI 变空**（小目标在 resize 后消失）
   你们 prepare 脚本对 CT 做了分位数归一化、前景裁剪、resize 到 `[32,256,256]`。
   这一步很可能让极小 ROI 消失 → 会直接影响训练/评估。

------

------

# 你可以直接写进 Proposal/PPT 的“EDA 决策”模板（给你现成句子）

这部分完全对齐 rubric 要求：图/表 + 解释 + 决策。

1. **样本独立单元选择**

- 发现：每个 ROI（region）基本对应 5 条 paraphrase QA → QA 行非独立。
- 决策：train/test 划分采用 region 或 case 分组，评估报告包含 prompt 方差（鲁棒性）。

1. **报告式文本噪声证据**

- 发现：Type1 的不确定措辞命中率显著更高（~70%），更贴近“临床报告不是 clean prompt”的动机。
- 决策：分 Type 报告性能；做 train(Type0)→test(Type1) 的 prompt shift 实验。

1. **grounded prompt 可行性**

- 发现：Question 常缺器官/左右侧；label_desc 携带更多定位线索。
- 决策：Prompt Distillation 优先基于 label_desc 抽取 organ/side/pathology（你们 pitch 的方法图）。

1. **multi-finding 报告构造**

- 发现：多数病例有多个 region（mean 2.88，最多 11）。
- 决策：构造 synthetic multi-finding report 数据集，验证“长+多发现→单 ROI 消歧”的核心研究问题。

------

如果你把你本地 `eda_followup/tables/empty_roi_after_prepare.csv`（空 ROI 比例）以及 `roi_voxels_hist_log1p.png`（小目标长尾）跑出来，那两张图会非常有说服力，能直接支撑你们 notes 里提到的“Dice + Hausdorff（边界误差）”设计理由。

下面是基于你上传的 **`eda_out.zip`**（我已解压查看）写的一份「一目了然、可直接放进报告/汇报」的 EDA 报告。
 ⚠️注意：这份 zip 里目前只包含**文本 EDA（第 1 阶段）**的输出文件：

- `tables/dataset_text_summary.json`
- `tables/rows_text_features.csv`
- `tables/regions_text_summary.csv`
- `tables/region_paraphrase_counts.csv`
- `figs/question_words_hist.png` / `answer_words_hist.png` / `question_type_bar.png` / `qa_rows_per_region_hist.png` / `question_anatomy_group_bar.png` / `question_laterality_bar.png`

没有看到你之前“影像/ROI 特征阶段”会生成的那些文件（例如 `regions_image_features.csv`、`mask_text_alignment_report.csv`、`roi_voxels_hist.png`、t‑SNE 坐标与图等）。
 所以**本报告的数值结论聚焦在文本与 region 结构**；最后我给了你“进一步探索（含影像/ROI）”建议和完整可跑代码，你可以在本地补齐第二阶段。

我也把这份报告整理成了 Markdown 文件，方便你下载直接用：
 [下载 M3D_RefSeg_EDA_Report.md](sandbox:/mnt/data/M3D_RefSeg_EDA_Report.md)

------



























# M3D‑RefSeg EDA 报告（基于你上传的结果）

## 0) 一句话总结（先抓重点）

- 数据集共有 **193 个病例（case）**、**556 个独立分割目标（region = case_id + Mask_ID）**，但 CSV 有 **2777 行 QA**，因为**每个 region 大多被写成 5 条不同问法（paraphrase）**。
- 问句分两种风格：
  - **Type0**（更像“直接分割/定位”）：1665 行（60.0%）
  - **Type1**（更像“解释/推断 + 分割”）：1112 行（40.0%）
     且 Type1 的“不确定性措辞”命中率非常高（≈70%）。
- Answer 中 `[SEG]` 覆盖率很高（**98.78%**），但仍有 **34 行**不含 `[SEG]`，这些多像“是/否问答”而不是分割指令，建议从分割训练集中剔除或单独作为 VQA 子任务。

------

## 1) 术语解释（给初学者）

- **EDA（Exploratory Data Analysis）探索性数据分析**：在建模前先看数据分布、质量问题、偏差、潜在泄漏（避免训练/测试“穿帮”）。
- **case（病例）**：一个 CT 体数据样本（例如 `s0001`），通常对应一次扫描。
- **mask（分割标注）**：一个与 CT 同形状的整数数组；背景=0，其余整数代表不同目标。
- **Mask_ID**：你要分割的目标标签 id。该目标的真值分割就是 `(mask == Mask_ID)`。
- **ROI（Region of Interest）感兴趣区域**：这里就是某个 `Mask_ID` 对应的目标区域。
- **region（独立分割目标单元）**：`region_id = case_id + Mask_ID`（例如 `s0001__2`）。这比 CSV 的“行”更接近独立样本。
- **paraphrase（同义改写问法）**：同一个 region 被写成多条不同问句，用于提示词增强（prompt augmentation）。
- **data leakage（数据泄漏）**：同一 region 同时出现在 train/test（哪怕问句不同），会导致评估虚高。
- **uncertainty cue（不确定性措辞）**：possible/likely/consider/may… 这类推断语气。
- **negation（否定）**：no/without/negative for… 这类否定表达。
- **laterality（侧别）**：left/right/bilateral（左/右/双侧）等空间线索。

------

## 2) 数据规模与层级结构

### 2.1 全局规模（来自 `dataset_text_summary.json`）

- **QA 行数（CSV 行）**：2777
- **病例数（case）**：193
- **region 数（独立分割目标）**：556

问句风格分布：

- **Type0**：1665（60.0%）clear
- **Type1**：1112（40.0%）uncertainty

文本噪声线索（启发式关键词命中）：

- Question 含 **不确定性措辞**：40.33%
- Question 含 **否定措辞**：1.15%

Answer 结构：

- Answer 含 `[SEG]`：98.78%（=> **34 行不含**）

------

### 2.2 每个 region 有多少条 QA（paraphrase 数）？

（来自 `region_paraphrase_counts.csv`）

| QA条数/region | region数 |
| ------------- | -------- |
| 4             | 3        |
| 5             | 553      |

解释（很重要）：
 **几乎所有 region 都是 5 条 QA**。这意味着：

- CSV 的 2777 行 **不是 2777 个独立分割样本**；
- 真正独立的样本单位更接近 **556 个 region**；
- 如果你按“行”随机拆分 train/test，会出现 **同一 region 的不同问句落入 train/test → 数据泄漏**。

对应图：`figs/qa_rows_per_region_hist.png`

------

### 2.3 每个 case 有多少个 region（同一病例多目标）？

（由 `regions_text_summary.csv` 推导）

- 平均每个 case：**2.88 个 region**
- 中位数：**2**
- 最大：**11**

覆盖情况：

- 至少 2 个 region 的 case：**129/193（66.8%）**
- 至少 3 个 region 的 case：**88/193（45.6%）**
- 至少 5 个 region 的 case：**35/193（18.1%）**

解释：
 很多病例天然是 **multi-finding（多发现）**，这对你们后续想做“长报告→定位到目标 ROI”的任务非常有利：
 你可以把同一 case 的多个 `label_desc` 拼成一段“报告”，再让模型分割其中某一个目标。

------

## 3) 文本分布与可解释结论

### 3.1 Question / Answer 长度分布（词数）

（来自 `rows_text_features.csv`）

**Question 词数：**

- mean 15.34，median 15，p95 23，min 6，max 36

**Answer 词数：**

- mean 18.12，median 17，p95 32，min 4，max 57

解释：

- Question 整体偏短且集中（中位数约 15 词），更像“指令模板”；
- Answer 更长、分散更大（尤其 Type1），解释性内容更多。

对应图：

- `figs/question_words_hist.png`
- `figs/answer_words_hist.png`

------

### 3.2 Type0 vs Type1：两种问句风格差异（非常清晰）

（我按 `Question_Type` 重新汇总）

| Type | 行数 | Q均值词数 | Q中位词数 | A均值词数 | A中位词数 | 不确定词比例 | 否定词比例 | A含[SEG]比例 |
| ---- | ---- | --------- | --------- | --------- | --------- | ------------ | ---------- | ------------ |
| 0    | 1665 | 13.39     | 13        | 14.06     | 13        | 20.2%        | 1.08%      | 98.86%       |
| 1    | 1112 | 18.26     | 18        | 24.20     | 23        | 70.4%        | 1.26%      | 98.65%       |

解释（初学者版）：

- **Type0**：更像“请分割/请定位”——短、直接、不确定性少
- **Type1**：更像“报告式推理+解释”——更长、不确定性非常多（≈70%）

这对你的项目特别有价值，因为 Type1 能作为“更接近真实报告语气”的子域。

建议你后续实验至少做两件事：

1. **分 Type0/Type1 分别汇报性能**（否则平均值会掩盖 Type1 的难度）
2. 做一个非常有说服力的鲁棒性实验：
    **train on Type0 → test on Type1**（prompt 风格 domain shift）

对应图：`figs/question_type_bar.png`

------

### 3.3 模板化（Template bias）：问句前三词统计

我从 `rows_text_features.csv` 统计了“问句前三词（prefix）”的 Top‑15：

| prefix(前三词)       | count |
| -------------------- | ----- |
| can you segment      | 433   |
| are there any        | 174   |
| can you identify     | 172   |
| please segment where | 130   |
| please segment the   | 126   |
| where can we         | 118   |
| where is the         | 117   |
| could you identify   | 115   |
| please identify and  | 113   |
| is there any         | 111   |
| where are the        | 56    |
| can you locate       | 50    |
| where does the       | 43    |
| which part of        | 40    |
| based on the         | 38    |

解释：

- 大量问句来自固定模板（“can you segment…/where is…/are there any…”）
- 这意味着：即使 paraphrase 看似很多，语言变化可能也主要是“模板替换”。

这会直接影响你们后续：

- 如果要做“prompt robustness / prompt distillation”，需要先量化“同一 region 的 5 条问句到底有多不同”（见下一节建议）。

------

## 4) region 级别语义分布（更贴近“分割目标”的视角）

`regions_text_summary.csv` 中有 `label_desc`（目标描述），并基于启发式关键词归了两个标签：

- `anatomy_group`（粗粒度器官系统）
- `laterality`（left/right/bilateral/none）

### 4.1 anatomy_group 分布（region 级）

| anatomy_group     | count | pct   |
| ----------------- | ----- | ----- |
| pulmonary         | 121   | 21.8% |
| other             | 107   | 19.2% |
| renal_urinary     | 89    | 16.0% |
| cardio_vascular   | 77    | 13.8% |
| hepato_biliary    | 68    | 12.2% |
| musculoskeletal   | 50    | 9.0%  |
| gastro_intestinal | 21    | 3.8%  |
| reproductive      | 13    | 2.3%  |
| neuro_headneck    | 10    | 1.8%  |

解释：

- 主要集中在 **肺、泌尿、心血管、肝胆** 等系统；
- `other` 表示关键词规则没覆盖（不是说它“没器官”，而是我们这套简单规则识别不出来）。

入门建议（选子任务更稳）：

- 如果你们想先聚焦一个 body part：从样本量上，**pulmonary / renal_urinary** 更稳。

------

### 4.2 laterality 分布（region 级）

| laterality | count | pct   |
| ---------- | ----- | ----- |
| none       | 253   | 45.5% |
| left       | 130   | 23.4% |
| right      | 122   | 21.9% |
| bilateral  | 51    | 9.2%  |

解释：

- 约一半 region 的描述里能提取出 left/right/bilateral 线索；
- laterality 是很便宜的“grounding 信号”（但严格一致性验证要注意坐标系：如果你用的是 prepare 后 npy，方向信息可能不可靠；如果用 NIfTI affine 会更严谨）。

------

## 5) 数据质量问题与清洗建议（可以直接落地）

### 5.1 `[SEG]` 缺失行（34 行）意味着什么？

- Answer 不含 `[SEG]`：**34/2777（1.22%）**
- 涉及 **25 个 region**（有的 region 里不止 1 行缺失）

我进一步看了这些缺失行最常见的问句模板（Top‑10 prefix）：

| prefix(前三词)      | count |
| ------------------- | ----- |
| is there any        | 9     |
| are there any       | 6     |
| where is the        | 5     |
| could you identify  | 2     |
| please identify and | 2     |
| what abnormality is | 1     |
| is it necessary     | 1     |
| are the enlarged    | 1     |
| what organs might   | 1     |
| which lung has      | 1     |

解释：

- 很多像 “Is there any… / Are there any…” ——更偏 **是/否问答**，不是分割指令；
- 所以这些行即便带着 `Mask_ID`，也可能对“文本引导分割”训练是噪声。

建议（按“分割/文本引导分割”为主）：

1. **训练分割模型时过滤掉 `has_seg_token==0` 的行**（最简单、最安全）
2. 若你们也想扩展任务，可以把它们单独做成 VQA/分类子任务
3. 不建议盲目在 Answer 前补 `[SEG]`：因为问题本身可能不是让你分割









![image-20260217132100486](C:\Users\63485\AppData\Roaming\Typora\typora-user-images\image-20260217132100486.png)

![image-20260217131436708](C:\Users\63485\AppData\Roaming\Typora\typora-user-images\image-20260217131436708.png)

### `t-SNE colored by anatomy_group`（文本→解剖粗分组）

**颜色含义**：我用一个简单的关键词规则，把文本描述粗分为 `renal_urinary / pulmonary / hepato_biliary / cardio_vascular ...` 等组（比如出现 kidney/ureter 就归 renal_urinary）。
 **你怎么读**：

- 如果同色点在 t-SNE 上形成相对明显的簇，说明 **文本本身携带较强的解剖定位信息**（这对你们“报告/文本引导分割”的动机是正证据：报告确实提供 anatomical cue）。Notes_ 2_12
- 如果颜色混得很厉害，可能表示：描述更偏“病理/现象”而不是器官；或关键词规则太粗（可以后续用更专业的 NER/词表改进）。



### `t-SNE colored by anatomy_group`（ROI 数值特征 vs 解剖分组）

**颜色含义**：还是基于文本的 anatomy_group，但点的位置来自 ROI 数值特征。
 **你怎么读**：

- 如果某些 anatomy_group 在 ROI 数值特征空间里天然分开（例如 renal_urinary 的 ROI 常在某些位置/某些尺寸范围），说明“解剖部位”不仅体现在文本里，也体现在 ROI 的几何统计上。
- 这对你们项目方法有启发：你们想做“从报告提取 grounded prompt”，那 anatomy_group 可以作为**中间结构化变量**（先把报告 distill 到器官/侧别/病灶，再做分割）。这正呼应你们 pitch 的“report→grounded prompts”思路。



![image_features_tsne_voxels](D:\M3D\eda_out\figs\image_features_tsne_voxels.png)

### `t-SNE colored by voxels`（ROI 体素体积大小）

**颜色含义**：`voxels = sum(mask==Mask_ID)`，在 resize 后网格上的体素数。
 **你怎么读**：

- 颜色越亮表示 ROI 越大。你这张能看到右侧一列点更偏大体积（亮色更多），说明体积大小是 t-SNE 结构的重要驱动因素之一。
- 这对建模很关键：如果 ROI 体积长尾、小目标占比高，你们后续训练就要考虑 **小 ROI 采样、loss reweight、多尺度 crop**，以及在评估里补充边界指标（你们 notes 里也提到 Hausdorff distance）。















![已上传的图片](https://chat.pilipili.tech/backend-api/estuary/content?id=file_00000000292071fd87c5a70e7d6b3c9a&ts=492042&p=fs&cid=1&sig=2c5b97c43cf7497fc5ef9cde71f412f77fdb647f85546393de6f0b7b750c1d28&v=0)



### `t-SNE colored by laterality`（文本→左右侧线索）文本 t-SNE

**颜色含义**：同样用关键词规则从描述里抓 `left/right/bilateral/none`。
 **你怎么读**：

- 如果 left/right 会形成局部聚类，说明文本里“左右侧”的词对语义空间影响明显。
- 如果完全混在一起，常见原因是：很多描述根本不写侧别（大量 `none`），或者描述里侧别表述多样/隐含（规则没抓全）。
   这张图和你们 pitch/notes 里提到的 “anatomy consistency（报告位置线索）” 是呼应的：这是在**还没训练模型前**对“位置线索是否存在”的快速证据。











![image-20260217131838699](C:\Users\63485\AppData\Roaming\Typora\typora-user-images\image-20260217131838699.png)

### `t-SNE colored by qa_rows`（每个 ROI 的 paraphrase 数）

**颜色含义**：同一个 region（case+Mask_ID）在 CSV 里有几条问句（通常 5 条）。
 **你怎么读**：

- 你这张基本一边倒（几乎全是 5），说明该数据集对每个 ROI 的“问句增广”非常规律。
- 这个图更多是“数据结构核验”：帮助确认你确实是在 region 级建样本，而不是 QA 行。
   对你们 proposal 的 EDA “Decisions made” 也有用：**因为 paraphrase 很规律，所以评估时要避免把同一 ROI 的不同问句泄漏到 train/test。**











![image-20260217131616391](C:\Users\63485\AppData\Roaming\Typora\typora-user-images\image-20260217131616391.png)

### `t-SNE colored by empty`（ROI 是否“空”）

**颜色含义**：

- `empty=1` 表示 **(mask==Mask_ID) 的体素数为 0** —— 也就是 prepare 后这个 ROI 消失了（常见原因：小目标被裁剪/resize 抹没，或 mask/text/csv 对不齐）。
- `empty=0` 正常有体素。

**你怎么读**：

- 你图里 `empty=1` 的点集中在左边一小撮，说明这些“空 ROI”在数值特征空间里很一致（通常就是 bbox/voxels 等特征都变成极端值）。
- 这张图是**数据清洗的硬证据**：训练前应该把这些样本剔除或单独处理，否则会给模型喂“没有 GT”的分割监督。
   这也正对应你们 proposal rubric 要求的“EDA→决策”。











![已上传的图片](https://chat.pilipili.tech/backend-api/estuary/content?id=file_000000000bb071fdae99a0627abc7ed1&ts=492042&p=fs&cid=1&sig=8013cc4e81fcea22829a47f774d12f1c802b5647036640de1656eda7480136a8&v=0)

### 5) `t-SNE colored by laterality`（用 ROI 位置特征推断/对照侧别）

**颜色含义**：laterality 仍然来自文本规则（left/right/bilateral/none），但点的位置来自**ROI 数值特征**（尤其是 centroid_x_norm 这种）。
 **你怎么读**：

- 如果右侧（right）的点在某些区域更集中、左侧（left）在另一些区域更集中，说明 ROI 的空间特征与文本侧别是有一致性的（这是 “anatomy consistency / location cue” 的 proxy）。Notes_ 2_12
- 如果完全混杂，可能是：NIfTI 方向没有统一导致 x 轴左右不一致；或者文本侧别多为 none；或者 ROI 位置特征在 PCA/t-SNE 中被其它特征主导。
   （更严谨可以做：用 centroid_x_norm 对 left/right 做统计检验，而不是只看 t-SNE。）



![image-20260217131905313](C:\Users\63485\AppData\Roaming\Typora\typora-user-images\image-20260217131905313.png)



------

# 6) 进一步探索建议（把 EDA 做到“能指导建模”）

下面这些建议是“下一步最值得做”的 EDA（也最容易写进 proposal/ppt 的解释+决策）：

## A) Prompt Diversity（强烈推荐，和你们任务最贴）

**目标**：量化“同一 region 的 5 条 paraphrase 问句到底有多不同”。
 **方法**：把问句做 embedding（TF‑IDF 或 SBERT），计算每个 region 内的两两 **cosine similarity（余弦相似度）**：

- cosine similarity 越高 → 两句越像 → 多样性越低
- 你可以画出 `mean_cos_sim` 的直方图，找出：
  - “几乎重复”的 region（非常模板化）
  - “差异很大”的 region（更考验模型鲁棒性）

**为什么重要**：
 如果 paraphrase 多样性很低，那它更像“重复数据”；你们做 prompt distillation/robustness 的空间会更小。反之则很适合做鲁棒性展示。

## B) 合成长报告（multi‑finding synthetic reports）

利用你们数据里“一个 case 多个 region”的结构，把同一病例多个 `label_desc` 拼成一段“报告”，然后指定其中某个目标让模型分割。
 这更贴近真实报告难点：**一段报告多个发现 → 需要把 query grounded 到正确 ROI**。

## C) Split 策略与泄漏检查

**必须**按 `case_id` 或至少 `region_id` 进行划分，避免同一 ROI 的不同问句进 train/test。
 进一步可以做“按 anatomy_group 的近似分层”，防止 test 某些器官缺失。

## D) 补齐影像/ROI EDA（如果你确实“都跑完了”，就把第二阶段结果也打包给我；否则用我给的脚本一键跑）

建议至少做：

- ROI 体积（voxels）长尾分布（决定采样/损失/指标）
- bbox 尺寸、slice 覆盖数（判断小目标、薄层目标）
- 连通域数量（单发/多发）
- prepare 后空 ROI 检测（小目标被 resize 抹没）
- overlay 导出质检图（随机 + outlier）

------

# 7) 你要的“完整可直接运行代码”（已给你打包好脚本）

## 代码 1：从已有 `eda_out/` 自动生成一份 Markdown EDA 报告

适合交作业/写 proposal，一键生成 `EDA_REPORT.md`。

[下载 eda_report_from_outputs.py](sandbox:/mnt/data/eda_report_from_outputs.py)

运行：

```bash
python eda_report_from_outputs.py --eda_dir ./eda_out
# 输出：./eda_out/EDA_REPORT.md
```

------

## 代码 2：更深入的 EDA（模板挖掘 / prompt 多样性 / t‑SNE / 合成长报告 / 可选影像特征）

[下载 m3d_refseg_eda_deeper.py](sandbox:/mnt/data/m3d_refseg_eda_deeper.py)

安装依赖：

```bash
pip install pandas numpy scikit-learn matplotlib tqdm
# 如果你要做影像/连通域/边界：
pip install scipy
# 如果你要用 SBERT embedding（可选）：
pip install sentence-transformers
```

常用命令（Windows 路径示例你改成自己的即可）：

1）一键跑“更深入文本 EDA + 合成长报告 + t‑SNE”（不含影像特征）：

```bash
python m3d_refseg_eda_deeper.py ^
  --csv "D:\Data\M3D\M3D_RefSeg.csv" ^
  --npy_root "D:\Data\M3D\M3D_RefSeg_npy" ^
  --out_dir ".\eda_deeper" ^
  --all
```

2）只做最推荐的 prompt 多样性分析（先跑这个）：

```bash
python m3d_refseg_eda_deeper.py ^
  --csv "D:\Data\M3D\M3D_RefSeg.csv" ^
  --out_dir ".\eda_deeper" ^
  --prompt_diversity ^
  --embed_method tfidf
```

3）补齐影像/ROI 特征（ct.npy/mask.npy 必须存在）：

```bash
python m3d_refseg_eda_deeper.py ^
  --csv "D:\Data\M3D\M3D_RefSeg.csv" ^
  --npy_root "D:\Data\M3D\M3D_RefSeg_npy" ^
  --out_dir ".\eda_deeper" ^
  --compute_image_features ^
  --n_overlay 30
```

4）生成“仅保留 `[SEG]` 行”的清洗版 CSV（建议用于分割训练）：

```bash
python m3d_refseg_eda_deeper.py ^
  --csv "D:\Data\M3D\M3D_RefSeg.csv" ^
  --out_dir ".\eda_deeper" ^
  --clean_csv_out ".\eda_deeper\tables\M3D_RefSeg_clean_segonly.csv"
```

------

## 你接下来怎么把它写进 proposal/ppt（模板）

你可以用“发现 → 影响 → 决策”三段式，直接复用：

- **发现（Evidence）**：每个 region 几乎固定 5 条 paraphrase；Type1 的不确定语气显著更强；少量 QA 不含 `[SEG]`。
- **影响（Implication）**：按行拆分会泄漏；Type1 更接近真实报告语气，评估应分域；缺失 `[SEG]` 是噪声样本。
- **决策（Decision）**：按 case/region split；分别报告 Type0/Type1；过滤无 `[SEG]` 行；后续做 prompt diversity 与 multi-finding 合成报告实验。

------

如果你确认你本地确实跑出了“影像/ROI 阶段”的输出（比如 `regions_image_features.csv`、`empty_regions_after_prepare.csv`、t‑SNE 图），但打包时漏了：
 你只要把整个 `eda_out/` 文件夹完整压缩再发我，我可以把**ROI 长尾、小目标比例、空 ROI、连通域、多发现 outlier**这些结论也补进报告里，并给你“按图写结论”的版本。







