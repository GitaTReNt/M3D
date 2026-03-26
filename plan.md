先说结论：你们现在最该优化的，不是分割头，而是“文本怎么稳定地变成空间提示”。

从现状看，你们已经把基线诊断得很清楚了：MedSAM + oracle bbox 已经能做到可用，fine-tuned MedSAM 在 30 个 case 上 Mean Dice 0.5221，明显好于原始 SAM；但 text-guided 结果只有约 0.02，而且 multimask / refine / CC3D 这些 inference-time trick 没有带来稳定提升。再结合失败分析里“过分割主导、小目标差、spine/bone 差、scattered/diffuse 差”，说明当前真正的瓶颈几乎全部在 text→spatial prompt 这一步，不在 mask decoder。

所以这阶段如果要给出一个“不训练也能提升”的方案，我会把方向定成：**数据驱动的 prompt 改造 + 空间先验生成**。传统 image augmentation 只有在训练时才真正起作用；当前更有价值的是 **prompt augmentation、metadata augmentation 和 spatial prior augmentation**。

## 无需训练，最值得先做的 5 件事

1. **把原始 description 结构化，做成 prompt bank**
   每条文本都拆成几个稳定字段：`anatomy / side / vertebral level / finding type / count / focality / diffuse / relation / enhancement`。
   然后给每个 mask 生成 3\~4 类 prompt：

   * location-only：`right inguinal region`
   * anatomy + finding：`right inguinal mass`
   * short canonical：`irregular right inguinal mass`
   * full prompt：`irregular mass in the right inguinal region with peripheral enhancement`

   这里有个关键点：**不是越短越好**。像 `multiple / bilateral / diffuse / scattered` 这些词，对多发或弥漫性病灶是核心信息，不能在简化时删掉。相反，`considering / suggests / on enhanced scanning` 这类不直接提供空间约束的词可以弱化。解剖词可以做安全归一化，病理词的同义改写要保守，避免语义漂移。

2. **用现有标注做一个不训练的 text→prior box / prior slice-range 系统**
   这是我认为最可能真正把 text-guided 拉起来的一步。
   具体做法是：把每个训练样本的 `3D bbox / centroid / z-range / size` 归一化存下来；输入新文本后，先做粗粒度 body region 判断，再按结构化标签或文本相似度，从训练集检索 top-k 相似描述，聚合出候选 `3D box / multi-box / slice range`。
   对 focal case，prior 不只给 box，还可以顺手给一个 **centroid point** 作为附加 prompt；对 diffuse case，prior 更适合给 **organ-level ROI 或多框**。
   唯一要严格注意的是：**prior bank 必须只由 train split 构建**。如果现在没有固定 split，就做 leave-one-case-out，避免把评测样本自己的标注信息漏进去。

3. **把样本分成 focal / multifocal-diffuse / degenerative-bony 三条路由**
   你们现在最大的误区，是默认所有目标都适合“单 bbox + 单一后处理”。但从现有结果看，这对 mass、nodule、cyst 这类 focal object 还行；对 `multiple enlarged lymph nodes`、`bilateral emphysema`、`vertebral degeneration / bone density change` 这类目标天然不适配。
   更合理的路由是：

   * **focal**：单个紧框 + centroid 约束
   * **multifocal / diffuse**：multi-box 或 organ ROI，不要 largest-CC
   * **bony / line-like / ambiguous sign**：单独打 `segmentability` 标签，作为 hard subgroup 处理

   这一步很重要，因为有些描述本身就不是“一个清晰 object”的分割任务，和肿块类样本不是同一种数据分布。

4. **后处理改成 prior-aware，不要继续堆通用 trick**
   你们已经验证过，multimask、iterative refinement、CC3D 都没有稳定收益，尤其 MedSAM 的 IoU head 在 multi-mask 选择上不可靠。
   所以后处理建议改成：

   * 先用 prior 限制 **z-range**
   * 再用相似样本的体积分布限制 **mask size**
   * focal case 保留 **离 prior centroid 最近** 的 component
   * diffuse case 保留 **多个 component**
   * 必要时跑几个阈值候选，选 **体积和位置最符合 prior** 的版本，而不是信 SAM 自己的分数

   这一步是专门针对你们现在最明显的过分割问题。

5. **单独建立一套“prompt 质量”指标**
   现在不要只盯最终 Dice。你们完全可以先证明“prompt 变好了”，再证明“mask 变好了”。
   建议加 4 个中间指标：

   * 文本解析覆盖率
   * GT centroid 是否落在候选 box 内
   * slice-range recall
   * pseudo box 与 GT box 的 IoU / volume ratio

   这样就算最终 Dice 还没立刻大涨，也能很清楚地展示 text→spatial prompt 这层确实在进步。

## 为后续训练，建议现在顺手沉淀的 5 类数据资产

1. **结构化 metadata 表**
   至少包含：`raw_text, normalized_text, anatomy, laterality, level, finding_type, focality, diffuse_flag, segmentability, bbox_norm, centroid_norm, slice_range, size_bin, difficulty_tag`

2. **每个 mask 的 prompt bank**
   原始 prompt、short prompt、location-only、anatomy-only、full prompt、coarse-to-fine prompt 都存下来。
   这一步后面训练时能直接变成多视角正样本。

3. **hard negative prompt**
   例如：

   * 同器官错侧：`left` ↔ `right`
   * 同病种错位置
   * 同病例其他 lesion 的 prompt
     后面做 contrastive 或 localization 时非常有用。

4. **difficulty / segmentability 标签**
   单独标出：

   * small target
   * diffuse / multifocal
   * spine / bone
   * ambiguous sign / line-like finding
     这样后面无论训练还是评测，都不会把本质不同的问题混在一起。

5. **离线增强包，但要明确是为训练准备，不是当前主收益**
   比如：

   * 左右翻转 + `left/right` 文本同步改写（前提是坐标方向一致）
   * 不同 crop scale
   * 如果还能拿到原始 HU，再做多 window 版本
     这些非常适合后续训练，但对当前“不训练”的线上提升不是第一优先。

## 后续训练，我建议的方向

后面真开始训，不建议一上来做 end-to-end text-to-mask。更稳的路线是：

1. **先训 text→box / text→slice-range localizer**
   先把“目标在哪”学出来，再去学边界。你们现在 0.02 和 0.51 之间的断层，本质就是定位没学到。

2. **再做两阶段：localizer + segmentor**
   第一阶段出 coarse ROI，第二阶段用 MedSAM 或 3D segmentor refine。
   这条路线和你们当前已有的 MedSAM 资产是天然兼容的。

3. **做多任务监督**
   把 `anatomy / laterality / focality / segmentability` 一起当辅助任务训，比只盯 mask loss 更容易收敛。

4. **把 prompt bank 和 hard negatives 真正用起来**
   同一个 mask 对应多种 prompt 视角；同时给模型一些“很像但不对”的负样本，提升文本对齐能力。

5. **分开对待 focal 和 diffuse**
   这两类最好不要强行共享同一套 prompt 逻辑和后处理逻辑，甚至训练时都可以考虑分路。

最后压成一句最适合汇报的话：**当前最有价值的无需训练改进，不是继续调 MedSAM 的 decoder trick，而是做一层 data-centric 的 text-to-prior prompt 系统：文本结构化、检索式空间先验、按目标类型路由。** 这套东西既能直接改善当前 text-guided pipeline，也会自然变成后续训练的数据底座。文档里提到的 M3D-LaMed、VoxTell、BiomedParse，本质上也都在解决同一个核心问题：文本和空间提示的对齐。

我建议你们下一步就按这个顺序推进：**先做结构化 prompt bank，再做 train-only 的 prior box 检索，再做 focal/diffuse 路由和 prior-aware 后处理，最后同时看中间指标和 Dice 的 ablation。**
