# BP → MedSAM 专项 tuning 指南

`docs/phase_c_plan.md` 的配套文档。同一项目，不同角度：

- **`phase_c_plan.md`**：把 Phase C 当成训练系统来写（数据划分、损失、调度）。
- **本文**：把 Phase C 当成**单一指标优化**来写。要动的指标只有一个——
  `BP pseudo-box → MedSAM` 的 Dice，当前 **0.1259**（208 cases / 784 masks，raw prompt）。

MedSAM 冻结不动。一切改动都是为了让 BP 输出足够好的 pseudo-box，
让 MedSAM 能真的起作用，而不是添乱。

---

## 1. 目标与基线状态

| 流水线                         | Dice    | 备注                             |
| ------------------------------ | ------- | -------------------------------- |
| BP direct (raw)                | 0.1286  | 当前 baseline                    |
| **BP → MedSAM (raw)**          | **0.1259** | **本文要推的指标**            |
| GT box → MedSAM                | ~1.12×  | 天花板（oracle）                 |
| Paired Δ (bp_medsam − bp_direct)| −0.0027 | refinement 目前等于白跑          |
| Ceiling retention              | ~11%    | (bp_medsam / gt_medsam)          |

**目标**：把 BP→MedSAM Dice 在 test split 上推到 ≥ `bp_direct × 1.3`，
且绝对值 ≥ 0.25（对应 `phase_c_plan.md` 的退出标准 #2）。
等价说法：ceiling retention 从 11% 提到 30%+。

---

## 2. 失败模式分解

Paired Δ ≈ 0 的含义是"MedSAM 一半案例帮倒忙、一半帮正忙，净收益为零"。
从 `results/12_phase_b_analysis/phase_b_merged_raw.csv` 抽查看到的
大致分布：

| 失败模式                                    | 占比估计 | 能修它的杠杆             |
| ------------------------------------------- | -------- | ------------------------ |
| BP 在**错误 slice** 上输出了 box            | ~40%     | `slice_exist` 头         |
| slice 对了但 box **过大**                   | ~25%     | `bbox` 回归 + GIoU       |
| BP 的 box **错中心**                        | ~20%     | `centroid` 头            |
| BP 有 box 但目标在别处（diffuse/multifocal）| ~10%     | `z_range` + component 策略|
| BP 直接空输出                               | ~5% (2/784) | `existence`（已接近零）|

以上是粗估。Stage 1 跑完后从 dev 三路 CSV 重新算。

---

## 3. 免费提升——训练前先把推理期旋钮扫掉

`scripts/biomedparse_evaluate.py` + `scripts/inference_medsam_from_pseudoboxes.py`
里有 3 个不训练就能调的旋钮。这些必须在解读 Stage 1 结果之前就**先钉死**，
否则训练产生的提升和推理 trick 的提升混在一起看不清。

### 旋钮 1：BP coarse-mask 阈值

当前：`sigmoid > 0.5`。Phase A 的发现是 BP 在 M3D-RefSeg 上的原始 logits 偏低
（所以才有 `--score_threshold 0 --no_nms` bypass）。产生 mask 后的二值化阈值
是**另一个**旋钮，还没扫过。

**在 dev split 上扫**：

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

选 dev BP→MedSAM Dice 最高的那个。**期望收益：1–3 个绝对点**，纯免费。

### 旋钮 2：pseudo-box margin

Phase B 提 box 时用的是固定 margin。太小会切到目标，太大会稀释 MedSAM 的 prompt。

**扫法**（同循环，改 `--bbox_margin`）：

```bash
for m in 0 3 5 8 12; do
  # 同上，只改 --bbox_margin $m
done
```

### 旋钮 3：connected-component 选择策略

BP 的 coarse mask 经常是多个不连通块。当前是所有正像素一起包一个 box。
备选：

| 策略                   | 实现                                  |
| ---------------------- | ------------------------------------- |
| `all_pixels`（当前）   | 所有正像素合并成一个 box              |
| `largest_cc`           | 只留 3D 最大连通分量                  |
| `top2_cc`              | 留最大两个分量，各出一个 box          |
| `per_slice_cc`         | 每 slice 只保留本 slice 最大连通分量  |

focal 类病灶通常 `largest_cc` 最佳；multifocal/diffuse 用 `top2_cc`
避免丢病灶。在 dev 上扫一遍。

**第 3 节的停止规则**：把 (threshold, margin, CC策略) 这三元组调到
zero-shot BP 在 dev 上 BP→MedSAM Dice 最高的组合**钉死**。
这个数就是新的**训练前 baseline**。之后 Stage 1 的所有提升都和这个比，
不再和 0.1259 比。

---

## 4. 训练杠杆——按对 BP→MedSAM 的作用排序

从 `phase_c_plan.md` §5 的默认权重重新排。那里是针对主 Dice；
这里针对 BP→MedSAM，优先级不同：

```
L = L_main (Dice + BCE + edge)       # 主线不能垮
  + 0.5  · L_slice_exist               # 第一优先：杀掉错 slice 的 box
  + 0.3  · L_bbox (smoothL1 + GIoU)    # 第一优先：压紧 x/y
  + 0.3  · L_existence                 # 第二
  + 0.2  · L_zrange                    # 第三
  + 0.1  · L_centroid                  # 第三（bbox 已隐含它）
```

**为什么这个顺序是 BP→MedSAM 专属**：

- `slice_exist` 是最大杠杆，因为 MedSAM **无法从错 slice 恢复**——
  它会在错特征上刷出一块 mask，直接拖 Dice。
- `bbox` 带 GIoU 直接优化 MedSAM 要消费的对象。不加它的话，box 质量
  就得完全依赖 mask→bbox 转换，损失信息。
- `centroid` 被降级是因为 `bbox` 头学得好就隐含了中心正确。
  留它当便宜的正则，不当主信号。
- `existence` 有天花板：BP 已经只有 2/784 空输出，还能挤出的空间有限。

---

## 5. LoRA 注入层

BP 里负责 x/y grounding 的有三处：

| 模块                            | 策略  | 针对 BP→MedSAM 的理由                      |
| ------------------------------- | ----- | ------------------------------------------ |
| 视觉 backbone (Focal/Swin)      | 冻结  | Phase B 证明瓶颈不在视觉特征               |
| `pixel_decoder`                 | LoRA 16 | 提升 mask 质量 → 提升 bbox 派生质量        |
| `transformer_decoder` X-attn    | **LoRA 32** | text ↔ image 绑定就在这里                |
| text projection                 | 可训  | 小且便宜，对齐文本 embedding               |
| 辅助头（新）                    | 可训  | 必须                                       |

Stage 2 升级（若需要）：把 `transformer_decoder` 完全解冻，backbone 继续冻。
只有 Stage 2 还卡住才考虑让 backbone 上 LoRA。

经验规则：LoRA rank 每翻一倍，`bbox_iou_pos` 涨 1–2 个点，rank 64 开始饱和
（在 150-case 训练集上）。没有 dev 证据不要直接超过 rank 32。

---

## 6. 训练期代理指标

这些是**不用每 epoch 跑 MedSAM** 就能预测 BP→MedSAM 涨幅的指标
（每 epoch 跑 MedSAM 要多花 ~10 分钟）。每个 epoch 在 dev 上全部记录：

| 指标               | 定义                                               | Stage 1 结束时目标 |
| ------------------ | -------------------------------------------------- | ------------------ |
| `slice_exist_f1`   | per-slice 二值 existence 的 F1                     | ≥ 0.80             |
| `bbox_iou_pos`     | 正 slice 上预测 bbox 与 GT bbox 的 IoU             | ≥ 0.35             |
| `centroid_err_norm`| L2(centroid_pred − centroid_gt) / 体积对角线       | ≤ 0.10             |
| `z_range_recall`   | 预测 z 范围覆盖 GT z 范围的比例                    | ≥ 0.75             |
| `main_dice_dev`    | 标准 Dice                                          | ≥ 0.20             |

**判断规则**：5 个目标全命中，BP→MedSAM Dice 几乎一定能过 0.25，
不用再继续调。

如果 `main_dice` 在涨但 `bbox_iou_pos` 不动 → 模型在学 mask 形状但没学定位。
加大 `λ_bbox` 继续训。

如果 `bbox_iou_pos` 在涨但 `slice_exist_f1` 不动 → 模型在对的 slice 放好了 box，
但没学会把错的 slice 拦掉。加大 `λ_slice_exist`，或者加 scorer 蒸馏（见 §8）。

---

## 7. Stage-1 之后的决策树

Stage 1 收敛后，在 **dev split**（不是 test，test 要留到 Phase C 最后验收）
上跑一次完整 3-way 诊断：

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

看三个数做决定：

| BP direct ↑ | BP→MedSAM ↑ | Retention | 动作                                   |
| :---------: | :---------: | :-------: | -------------------------------------- |
| 涨          | 涨          | ≥ 30%     | 跑 test split；启动 Phase D            |
| 涨          | 涨          | 15–30%    | 短 Stage 2（只调 `bbox`+`slice_exist`）|
| 涨          | **没涨**    | 任意      | `bbox` 权重不够——加 λ_bbox 再训        |
| 没涨        | 没涨        | 任意      | Stage 2 全量（解冻 X-attn）            |
| direct 不动，medsam 涨 | — | — | 反常，逐 case 对比看发生了什么 |

---

## 8. 组员 slice scorer 的接入路径

Scorer 产出 per-slice 二值 existence 概率。本轴上有两种合理用法：

### 路径 A：推理期 gate，架在 BP→MedSAM 上

最便宜，不训练。BP forward 后、MedSAM 前加一层：

```python
# 伪代码
for z in range(D):
    if scorer_prob[z] < gate_thresh:
        pseudo_boxes[z] = None  # 本 slice 跳过 MedSAM
```

在 dev 上扫 `gate_thresh`。期望行为：thresh=0.5 时能砍掉 30–60% 的错 slice box，
用 ~5% 的 recall 换 ~20% 的 precision。失败模式分解（§2）显示当前问题里
precision 不足占主导，所以这笔交易净收益为正。

收益是机械的。每次换 Stage 都要重测。

### 路径 B：当 `slice_exist` 头的蒸馏老师

在 Stage 1 损失里加：

```
L_distill = KL( stop_grad(scorer_prob) || slice_exist_sigmoid )
L = ... + 0.2 · L_distill
```

Scorer 教 BP 的 `slice_exist` 头学到一个 calibrate 过的 per-slice existence 信号，
即使训练样本稀疏时也有软标签可用。实现便宜（冻结 scorer 加一次 forward）。

**前提**：scorer 的训练 split 必须和 Phase C dev/test split **不相交**——
否则 dev 指标会泄漏。必须先重做 split，或者在 Phase C 的 train-only cases 上
把 scorer 重训一遍。

---

## 9. 本轴专属 ablation

从 Stage 1 best ckpt 分叉做 1-epoch 探针，结果记到
`results/phase_c_ablations/bp_medsam_summary.csv`：

| Ablation                         | 预期 BP→MedSAM Δ      |
| -------------------------------- | --------------------- |
| `-L_slice_exist`（去掉）         | −3 到 −6 个点         |
| `-L_bbox`（去掉）                | −2 到 −4 个点         |
| `+scorer 蒸馏（路径 B）`         | +1 到 +3 个点         |
| `+scorer gate（路径 A）`         | +1 到 +4 个点         |
| `bbox margin: 最优 vs 0`         | +0.5 到 +2 个点       |
| `CC 策略: largest vs all`        | ±1 个点（依病灶类型） |
| `LoRA rank 16 → 32`              | +1 到 +2 个点         |
| `raw prompt only vs dual prompt` | −1 到 +1 个点         |

实际数和预期偏差过大就停下来，检查训练配方，别再烧算力。

---

## 10. Stage 2 触发条件（专门针对本轴）

Stage 1 跑完后任一条件成立就进 Stage 2：

- `bbox_iou_pos_dev` < 0.25 且 `main_dice_dev` ≥ 0.18
  → 定位瓶颈；解冻 X-attn
- `slice_exist_f1_dev` < 0.75
  → per-slice existence 仍然弱；如果还没加 scorer 蒸馏就加上
- BP→MedSAM retention on dev < 15%
  → grounding 整体卡死；Stage 2 + 提升辅助权重

**不要只因为主 Dice 停滞就进 Stage 2**——主 Dice ≥ 0.20 且代理指标达标，
就足以跑 test split。

---

## 11. 单行成功标准

Stage 1 在 BP→MedSAM 轴上成功当且仅当：

```
dev(BP→MedSAM) ≥ 1.5 × dev(BP→MedSAM zero-shot after §3 sweeps)
AND dev(BP→MedSAM) ≥ 0.22
```

达成 → 按 §7 决策树看 retention 判断是否进 Stage 2。

未达成 → 复查损失权重，或者检查辅助标签质量
（`scripts/build_aux_labels.py` 的输出在 GT mask 有杂散 voxel 时会出错）。
