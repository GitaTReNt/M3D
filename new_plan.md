下面是文档全文：

```md
# M3D-RefSeg 下一步任务计划（BiomedParse v2 主线）

日期：2026-04-18  
项目：M3D-RefSeg  
定位：当前阶段执行版任务计划与分析

---

## 1. 当前判断

### 1.1 任务本质
当前任务的主瓶颈不是“最后怎么把 mask 边界抠得更细”，而是：

**自由临床文本 → 正确空间位置** 这一层还没有建立起来。

这也是为什么现有结果会呈现出很大的上限差：
- 纯 text-guided 最好结果仍然较低；
- 一旦给出正确 bbox，MedSAM 可以把分割质量明显拉高。

因此，下一阶段不应该把主要精力放在“换更大的分割器”上，而应该放在：

**先把文本对应的空间粗定位学出来，再让现有的强分割器做精修。**

---

### 1.2 主线结论
下一步主线建议明确为：

**text → BiomedParse v2 粗定位 → pseudo-box / ROI → MedSAM 精修**

但这里有一个重要判断：

**`BiomedParse → pseudo-box → MedSAM` 本身更像诊断实验，不是最终主要冲分手段。**

它的价值是回答一个关键问题：

> BiomedParse 当前给出的粗定位，是否已经“够 MedSAM 用”？

如果答案是否定的，那么后续主攻方向就非常明确：

**重点放在 finetune BiomedParse 的粗定位能力，而不是继续折腾 MedSAM。**

---

### 1.3 对当前路线的策略调整
结合现在的状态，建议把整体策略调整成下面三层：

1. **先解除 BiomedParse zero-shot 阻塞**  
   先确认模型能不能输出非空预测，拿到可靠 baseline。

2. **用 pseudo-box + MedSAM 做一次诊断实验**  
   判断当前粗定位是否已经具有可利用价值。

3. **把主攻方向转向 BiomedParse finetune**  
   重点训练 existence / coarse localization / coarse mask，而不是一开始就追最终精细边界。

一句话概括：

**BiomedParse v2 负责“先找到大概在哪”，MedSAM 负责“在这个区域里抠清楚边界”，而真正值得投入训练资源的核心，是第一段。**

---

## 2. 核心科研假设

### H1：当前瓶颈在 text-to-space grounding，而不是 boundary decoding
如果这个判断成立，那么：
- 直接分割模型再换更大的 backbone，提升不会特别大；
- 只要粗定位做对，即使用现有 MedSAM，也有明显提升空间。

### H2：BiomedParse v2 更适合作为 coarse grounding backbone，而不是一步到位的最终分割器
也就是说，它更适合先给出：
- coarse mask
- objectness
- pseudo-box
- ROI

再交给后续模块处理。

### H3：当前项目真正值得重点投入的是 BiomedParse 的 finetune
因为 zero-shot 即便跑通，预测框也大概率不会接近 GT box。真正决定后续能否往上走的，是：
- 模型能否判断目标是否存在；
- 能否给出大致正确的中心位置；
- 能否给出可用的 z-range / bbox；
- 能否输出至少可转成 pseudo-box 的 coarse mask。

---

## 3. 研究主线

### 主模型选择
- **主线模型**：BiomedParse v2
- **对照模型**：VoxTell
- **精修模块**：现有 MedSAM（先不要急着换 MedSAM2）

### 为什么这样选
- BiomedParse v2 更接近“文本驱动 + 3D + promptable segmentation”这条主线；
- VoxTell 保留为强 zero-shot baseline；
- MedSAM 在“给对 box”的条件下已经证明很强，因此当前不是优先替换对象。

### 方法定位
最终方法不是：

**“让一个模型直接从 free-text 端到端输出高质量 mask。”**

而是：

**“先把 free-text grounding 做出来，再用稳定的视觉 prompt 分割器做 refinement。”**

---

## 4. 分阶段执行计划

## Phase A：先解除 BiomedParse zero-shot 阻塞

### 目标
先让 BiomedParse v2 至少能在 smoke test 中输出非空预测，建立可分析的 zero-shot baseline。

### 当前最需要检查的三个问题
1. **输入 scale 是否正确**  
   重点检查 HU window、归一化方式、是否真的符合模型预期。

2. **prompt 是否被截断**  
   长临床句、双 prompt 拼接、CLIP token 上限都可能导致有效语义丢失。

3. **阈值 / NMS 是否过严**  
   模型可能有弱响应，但被阈值和 NMS 直接清空。

### 本阶段要做的事
- 跑通多种 scale mode；
- 逐 prompt forward，不做长句拼接；
- 扫 score threshold；
- 比较 with / without NMS；
- 输出每例的 empty_pred 统计；
- 保存原始 coarse prediction 供后续分析。

### 本阶段成功标准
满足任一条件即可视为解除阻塞：
- smoke test 出现稳定的非空预测；
- 全量评测中 empty rate 明显下降；
- 至少能输出可视化上“落在合理位置附近”的 coarse mask。

### 本阶段产出
- `BiomedParse zero-shot` baseline
- 失败案例分析表
- 可视化样例（正确 / 错位 / 空预测）

---

## Phase B：做一次 pseudo-box → MedSAM 诊断实验

### 目标
判断当前 BiomedParse 的粗定位，是否已经足够支撑 MedSAM refinement。

### 具体做法
1. 用 BiomedParse 输出 coarse mask；
2. 对 coarse mask 做阈值化；
3. 取一个或多个 connected components；
4. 从 component 计算 3D pseudo-box；
5. 给 box 加 margin；
6. 把 pseudo-box 输入现有 MedSAM；
7. 得到 refined mask。

### 这里的重点
这一步**不是**做 FID、KL 散度，或者把 box 当分布去拟合。  
这一步本质上只是把：

**GT box → MedSAM**

替换成：

**Predicted box → MedSAM**

也就是一个非常标准的两阶段流程：

**coarse localization → box prompt segmentation**

### 本阶段关键对比
必须并排比较这三条链：

1. **BiomedParse direct mask**
2. **BiomedParse pseudo-box → MedSAM**
3. **GT box → MedSAM**

### 如何解释结果
- 如果 `2` 明显高于 `1`：
  说明 BiomedParse 已经学到了一些可利用的空间定位，只是边界不够好。

- 如果 `2` 和 `1` 都很低：
  说明主问题仍然在粗定位，后续必须把精力转向 finetune BiomedParse。

- 如果 `2` 比 `1` 还差：
  说明当前 coarse mask 连可用 box 都提不出来，zero-shot grounding 还不够成熟。

### 本阶段成功标准
不要求这一步直接成为最终高分方法。  
它的成功标准是：

**帮助我们判断问题到底卡在定位，还是卡在边界。**

---

## Phase C：主攻 BiomedParse finetune

### 目标
把 BiomedParse 从“经常空预测 / 定位不稳”训练成“能给出可用粗定位”的文本定位器。

### 核心判断
后续最值得投入的，不是训练一个更复杂的 refiner，而是训练 BiomedParse 学会这三件事：

1. **目标是否存在**
2. **目标大概在哪**
3. **目标的粗 mask / bbox / z-range 是什么**

### 推荐训练顺序
#### 第一阶段：轻量微调
- 冻结大部分 backbone；
- 只调 decoder、text projection、少量高层模块；
- 优先使用 LoRA 或 partial FT；
- 先不做 full fine-tune。

#### 第二阶段：如果仍明显 underfit
- 再逐步解冻更高层的 cross-modal / visual blocks；
- full FT 放在最后。

### 为什么先不上 full FT
- 目前数据量仍然不大；
- 直接 full FT 风险是过拟合和工程复杂度同时升高；
- 当前更适合先验证“粗定位能力能否通过轻量微调显著改善”。

### 训练目标建议
#### 主监督
- GT mask supervision：Dice + BCE / CE

#### 辅助监督（建议强烈加入）
从 GT mask 自动导出中间标签：
- `bbox`
- `centroid`
- `z-range`
- `volume`
- `component_count`

然后给模型加辅助头或辅助 loss：
- existence loss
- bbox loss
- centroid loss
- z-range loss

### 这样做的意义
这会把原本隐式的“文本 → 空间”过程变成可监督、可分析的中间层。  
就算最终 Dice 提升有限，这部分本身也很有论文价值。

### 本阶段成功标准
- zero-shot 基线被稳定推高；
- empty rate 显著下降；
- pseudo-box IoU、centroid error、z-range recall 有明显改善；
- `predicted box → MedSAM` 开始变得可用。

---

## Phase D：做 prompt augmentation，但不是替代 raw

### 目标
提高文本表达变化下的稳健性，减少模型对 phrasing 的敏感性。

### 原则
不是把 raw clinical sentence 换成结构化短语，  
而是做：

**raw + canonical 双轨训练**

### 建议的 prompt 版本
正样本：
- raw 原始临床句
- canonical 短句
- anatomy + finding
- location-only
- paraphrase

负样本：
- laterality flip
- anatomy swap
- finding swap
- 同病例其他 lesion 的描述

### 训练方式
对同一 `(image, mask)`：
- raw 和 canonical 都作为正样本；
- 做 consistency regularization；
- hard negative 用 existence / contrastive 约束。

### 本阶段定位
这是重要增强项，但不是第一阶段主攻。  
优先级应低于：
- zero-shot 跑通
- pseudo-box 诊断
- BiomedParse finetune 主线

---

## Phase E：再加 prior-aware postprocess 和 routing

### 目标
解决 one-size-fits-all pipeline 对不同 lesion 类型不适配的问题。

### 推荐划分
把样本分成三类：
- **focal**
- **multifocal / diffuse**
- **bony**

### 不同类别的处理思路
#### focal
- 强调单主 component
- 更依赖 centroid 与 pseudo-box
- 更适合接 MedSAM refine

#### multifocal / diffuse
- 不强制只保留 largest CC
- 更强调 z-range 和 multi-component 保留
- 不一定强依赖 MedSAM

#### bony
- bone window 优先
- 阈值可更低
- 形态与连续性规则不同于软组织病灶

### prior bank 的作用
从 train split 建立先验：
- centroid
- bbox_norm
- z-range
- volume
- finding / anatomy / focality 标签

推理时利用 prompt 解析出的结构化槽位去检索相似先验，辅助：
- component 选择
- z-range 裁剪
- volume 约束
- diffuse / focal 路由

### 本阶段定位
这是中后期稳定提升模块，适合在主线模型已经具备基本定位能力之后加入。

---

## 5. 评测设计

## 5.1 主指标
- Dice
- NSD（如果已支持）
- empty prediction rate

## 5.2 中间指标
强烈建议把这些指标正式纳入主表或补充材料：
- pseudo-box IoU
- centroid error
- z-range recall
- component recall
- existence accuracy
- paraphrase robustness
- laterality robustness

## 5.3 为什么必须看中间指标
因为当前项目真正的科学问题，是：

**模型到底有没有把文本正确地落到空间上。**

如果只看 final Dice，很容易看不出来问题到底出在：
- 文本理解
- 空间定位
- 边界细化
- 后处理

---

## 6. 明确优先级

## 第一优先级（马上做）
1. 解除 BiomedParse zero-shot 阻塞；
2. 跑出稳定的 zero-shot baseline；
3. 做 `BiomedParse direct` vs `BiomedParse → MedSAM` vs `GT box → MedSAM` 三路对比。

## 第二优先级（主攻）
4. BiomedParse LoRA / partial FT；
5. 加入 existence / bbox / centroid / z-range 辅助监督；
6. 重跑 pseudo-box → MedSAM 诊断，看是否已经可用。

## 第三优先级（增强）
7. prompt augmentation；
8. hard negatives；
9. prior-aware postprocess；
10. focal / diffuse / bony routing。

## 第四优先级（后续 ablation）
11. text encoder 替换；
12. MedSAM2 / Medical SAM3 refiner 消融；
13. full fine-tune。

---

## 7. 两周执行版

## Week 1
### Day 1-2
- 跑通 BiomedParse smoke；
- 验证输入 scale / one-prompt-forward / threshold / no-NMS；
- 输出可视化与 empty case 统计。

### Day 3
- 跑全量 zero-shot baseline；
- 固定评测协议与日志格式；
- 保存 coarse mask 和候选 component。

### Day 4
- 实现 pseudo-box 提取；
- 接现有 MedSAM inference；
- 跑三路对比实验。

### Day 5
- 分析结果；
- 明确判断主问题是否在 coarse grounding；
- 冻结主线，进入 finetune 准备。

## Week 2
### Day 6-7
- 搭 BiomedParse LoRA / partial FT trainer；
- 从 GT mask 导出 bbox / centroid / z-range 标签；
- 确定训练和验证协议。

### Day 8-9
- 训练第一版轻量微调模型；
- 跑 dev fold；
- 记录 empty rate、pseudo-box IoU、centroid error 变化。

### Day 10
- 重跑 `predicted box → MedSAM`；
- 判断微调后是否已经具备“可用 box”能力。

### Day 11-12
- 如果定位改善明显：开始加 prompt augmentation；
- 如果定位仍差：优先继续加强辅助监督，不急着加复杂后处理。

### Day 13-14
- 汇总第一轮结果；
- 决定是否进入 prior / routing 阶段。

---

## 8. 当前最重要的判断标准

下一阶段不是看某个 fancy 方法名字好不好听，  
而是看下面这个判断是否成立：

> **经过微调后，BiomedParse 是否能稳定输出一个“MedSAM 用得上”的 pseudo-box？**

如果答案是“能”，那整个系统就有机会往上走。  
如果答案是“不能”，那说明当前所有提升都应该继续集中在：

**existence + coarse localization + prompt grounding**

而不是提前把精力分散到更复杂的 refine 模块上。

---

## 9. 一句话结论

这条线最合理的科研表述不是：

**“我们要让 BiomedParse 直接做出最好的最终 mask。”**

而是：

**“我们要把 BiomedParse finetune 成一个可靠的 free-text 粗定位器，再用 MedSAM 把这个定位转成更高质量分割。”**

因此，当前最合理的主线是：

**先修通 BiomedParse zero-shot → 用 pseudo-box + MedSAM 做诊断 → 主攻 BiomedParse finetune → 再加 prompt augmentation 与 prior/routing。**
```

这份内容是根据你当前上传的项目报告整理出来的。