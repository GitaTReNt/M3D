
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a beginner-friendly EDA report (Markdown) from an existing eda_out folder.

This script is designed for M3D-RefSeg EDA outputs produced by our EDA pipeline.
It reads the tables/ directory (and optionally figs/) and writes a single report:

  <eda_out>/EDA_REPORT.md

Usage:
  python eda_report_from_outputs.py --eda_dir ./eda_out

Optional:
  python eda_report_from_outputs.py --eda_dir ./eda_out --report_name MY_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def pct(x: float, total: float) -> float:
    return 0.0 if total == 0 else (100.0 * x / total)


def prefix(text: str, n: int = 3) -> str:
    toks = re.findall(r"[A-Za-z']+", str(text).lower())
    return " ".join(toks[:n]) if toks else ""


def safe_read_json(p: Path) -> Dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def describe_numeric(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(s) == 0:
        return {}
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p05": float(np.percentile(s, 5)),
        "p25": float(np.percentile(s, 25)),
        "p50": float(np.percentile(s, 50)),
        "p75": float(np.percentile(s, 75)),
        "p95": float(np.percentile(s, 95)),
        "max": float(s.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda_dir", required=True, help="Path to eda_out folder")
    ap.add_argument("--report_name", default="EDA_REPORT.md", help="Output report filename under eda_dir/")
    args = ap.parse_args()

    eda_dir = Path(args.eda_dir)
    tables = eda_dir / "tables"
    figs = eda_dir / "figs"

    # Required files (text EDA)
    summary_path = tables / "dataset_text_summary.json"
    rows_path = tables / "rows_text_features.csv"
    regions_path = tables / "regions_text_summary.csv"
    reg_counts_path = tables / "region_paraphrase_counts.csv"

    for p in [summary_path, rows_path, regions_path, reg_counts_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required EDA file: {p}")

    summary = safe_read_json(summary_path)
    rows = safe_read_csv(rows_path)
    regions = safe_read_csv(regions_path)
    reg_counts = safe_read_csv(reg_counts_path)

    # Core metrics
    n_rows = int(summary.get("n_rows", len(rows)))
    n_cases = int(summary.get("n_cases", rows["case_id"].nunique() if "case_id" in rows else -1))
    n_regions = int(summary.get("n_regions", regions["region_id"].nunique() if "region_id" in regions else -1))

    q_words_stats = describe_numeric(rows.get("q_words", pd.Series(dtype=float)))
    a_words_stats = describe_numeric(rows.get("a_words", pd.Series(dtype=float)))

    # Group by Question_Type if available
    type_table = None
    if "Question_Type" in rows.columns:
        type_table = (
            rows.groupby("Question_Type")
            .agg(
                n=("Question_Type", "size"),
                q_words_mean=("q_words", "mean"),
                q_words_median=("q_words", "median"),
                a_words_mean=("a_words", "mean"),
                a_words_median=("a_words", "median"),
                unc_rate=("q_has_unc", "mean") if "q_has_unc" in rows.columns else ("Question_Type", lambda x: float("nan")),
                neg_rate=("q_has_neg", "mean") if "q_has_neg" in rows.columns else ("Question_Type", lambda x: float("nan")),
                has_seg_rate=("has_seg_token", "mean") if "has_seg_token" in rows.columns else ("Question_Type", lambda x: float("nan")),
            )
            .reset_index()
        )

    # Missing [SEG]
    no_seg_rows = pd.DataFrame()
    if "has_seg_token" in rows.columns:
        no_seg_rows = rows[rows["has_seg_token"] == 0].copy()

    # Template prefixes
    prefix_counts = Counter()
    if "Question" in rows.columns:
        prefix_counts = Counter(rows["Question"].map(lambda t: prefix(t, 3)).tolist())

    # Region stats
    regions_per_case = regions.groupby("case_id").size() if "case_id" in regions.columns else pd.Series(dtype=int)

    anatomy_counts = regions["anatomy_group"].value_counts(dropna=False) if "anatomy_group" in regions.columns else pd.Series(dtype=int)
    laterality_counts = regions["laterality"].value_counts(dropna=False) if "laterality" in regions.columns else pd.Series(dtype=int)

    # Paraphrase stats
    qa_rows_dist = reg_counts["qa_rows"].value_counts().sort_index() if "qa_rows" in reg_counts.columns else pd.Series(dtype=int)

    # Compose report
    out_path = eda_dir / args.report_name
    lines = []

    lines.append("# M3D-RefSeg EDA 报告（由脚本自动生成）\n")
    lines.append("本报告基于 `eda_out/tables/` 中的统计结果自动汇总。")
    lines.append("如果你在 `eda_out/figs/` 下也有对应的图，本报告会引用它们的文件名，方便你在本地打开查看。\n")

    lines.append("## 1. 术语小词典（给初学者）\n")
    lines.append("- **EDA (Exploratory Data Analysis)**：探索性数据分析。目的是在建模前了解数据分布、质量问题、潜在泄漏/偏差。")
    lines.append("- **case（病例）**：一个 CT 体数据（例如 `s0001/ct.npy`），通常对应一个病人/一次扫描。")
    lines.append("- **ROI (Region of Interest)**：感兴趣区域。这里指某个需要被分割的目标区域。")
    lines.append("- **Mask_ID**：mask 中的整数标签 id。当前样本要分割的 ROI = `(mask == Mask_ID)`。")
    lines.append("- **region（分割目标单元）**：一个 `case_id + Mask_ID` 组合。它比 CSV 的“行”更接近独立样本。")
    lines.append("- **paraphrase（改写/同义问法）**：同一个 region 对应多条不同问句，用来增强提示词（prompt）的多样性。")
    lines.append("- **data leakage（数据泄漏）**：同一个 region 同时出现在 train 和 test（哪怕问句不同），会导致评估虚高。")
    lines.append("- **laterality（侧别）**：left/right/bilateral（左/右/双侧）等方位线索。")
    lines.append("- **uncertainty cue（不确定性措辞）**：如 possible/likely/consider 等“推断语气”。")
    lines.append("- **negation（否定）**：如 no/without/negative for 等否定表达。\n")

    lines.append("## 2. 数据规模总览\n")
    lines.append(f"- QA 行数：**{n_rows}**")
    lines.append(f"- 病例数（case）：**{n_cases}**")
    lines.append(f"- region 数（独立分割目标）：**{n_regions}**\n")

    qtc = summary.get("question_type_counts", {})
    if qtc:
        lines.append("**Question_Type 分布（问句风格）**：")
        for k, v in qtc.items():
            lines.append(f"- Type {k}: {v} 行（{pct(float(v), float(n_rows)):.1f}%）")
        lines.append("")

    if "has_seg_token_rate" in summary:
        lines.append(f"- Answer 含 `[SEG]` 的比例：**{summary['has_seg_token_rate']*100:.2f}%**")
        if len(no_seg_rows) > 0:
            lines.append(f"  - 未包含 `[SEG]` 的行数：**{len(no_seg_rows)}**（建议清洗/过滤）")
        lines.append("")

    if "q_has_unc_rate" in summary:
        lines.append(f"- Question 命中“不确定性措辞”的比例（启发式）：**{summary['q_has_unc_rate']*100:.2f}%**")
    if "q_has_neg_rate" in summary:
        lines.append(f"- Question 命中“否定措辞”的比例（启发式）：**{summary['q_has_neg_rate']*100:.2f}%**")
    lines.append("")

    lines.append("## 3. 文本长度分布（Question / Answer）\n")
    if q_words_stats:
        lines.append(f"- Question 词数：mean={q_words_stats['mean']:.2f}, median={q_words_stats['p50']:.0f}, p95={q_words_stats['p95']:.0f}, min={q_words_stats['min']:.0f}, max={q_words_stats['max']:.0f}")
        lines.append("  - 图：`figs/question_words_hist.png`")
    if a_words_stats:
        lines.append(f"- Answer 词数：mean={a_words_stats['mean']:.2f}, median={a_words_stats['p50']:.0f}, p95={a_words_stats['p95']:.0f}, min={a_words_stats['min']:.0f}, max={a_words_stats['max']:.0f}")
        lines.append("  - 图：`figs/answer_words_hist.png`")
    lines.append("")

    if type_table is not None:
        lines.append("## 4. Question_Type 对比（不同问句风格）\n")
        lines.append("Type 0 通常更像“直接分割/定位”，Type 1 更像“解释/推断 + 分割”。\n")
        lines.append(type_table.to_markdown(index=False))
        lines.append("")

    lines.append("## 5. 模板化（Template bias）初步观察\n")
    lines.append("通过统计问句前三个词（prefix），可以看到大量问句来自固定模板（例如 *can you segment* / *are there any*）。")
    lines.append("这意味着：prompt 多样性可能有限；模型可能学到“模板”，而不是更强的语言理解。\n")
    top_prefix = prefix_counts.most_common(15)
    if top_prefix:
        prefix_df = pd.DataFrame(top_prefix, columns=["prefix(first 3 words)", "count"])
        lines.append(prefix_df.to_markdown(index=False))
    lines.append("图：`figs/question_type_bar.png`、`figs/question_anatomy_group_bar.png`（如果你生成了前缀图，也可自行补充）\n")

    lines.append("## 6. region 级别分析（更接近“独立分割样本”）\n")
    if not qa_rows_dist.empty:
        lines.append("**每个 region 对应多少条 QA（paraphrase 数）**：")
        for k, v in qa_rows_dist.items():
            lines.append(f"- {int(k)} 条/region：{int(v)} 个 region")
        lines.append("图：`figs/qa_rows_per_region_hist.png`\n")

    if len(regions_per_case) > 0:
        lines.append("**每个 case 有多少个 region（同一病例内多目标）**：")
        lines.append(f"- mean={regions_per_case.mean():.2f}, median={regions_per_case.median():.0f}, max={regions_per_case.max():.0f}")
        lines.append(f"- 至少 2 个 region 的 case：{int((regions_per_case>=2).sum())}/{int(regions_per_case.shape[0])}")
        lines.append(f"- 至少 3 个 region 的 case：{int((regions_per_case>=3).sum())}/{int(regions_per_case.shape[0])}")
        lines.append("这支持构造“multi-finding 报告”（把同一病例多个发现拼成一段长文本）用于更贴近真实报告场景。\n")

    if not anatomy_counts.empty:
        lines.append("**region 的粗粒度解剖分组（anatomy_group，启发式关键词）**：")
        anatomy_df = anatomy_counts.rename_axis("anatomy_group").reset_index(name="count")
        anatomy_df["pct"] = anatomy_df["count"].map(lambda x: f"{pct(float(x), float(anatomy_df['count'].sum())):.1f}%")
        lines.append(anatomy_df.to_markdown(index=False))
        lines.append("")

    if not laterality_counts.empty:
        lines.append("**region 的侧别（laterality，来自 label_desc/问题文本的启发式）**：")
        lat_df = laterality_counts.rename_axis("laterality").reset_index(name="count")
        lat_df["pct"] = lat_df["count"].map(lambda x: f"{pct(float(x), float(lat_df['count'].sum())):.1f}%")
        lines.append(lat_df.to_markdown(index=False))
        lines.append("")

    lines.append("## 7. 数据质量与清洗建议（本次输出可直接支持）\n")
    if len(no_seg_rows) > 0:
        lines.append(f"- 检测到 **{len(no_seg_rows)} 条** Answer 不包含 `[SEG]`。这些样本往往更像“是/否问答”而不是分割指令，建议：")
        lines.append("  1) 用于纯分割训练时 **过滤掉**；或")
        lines.append("  2) 单独作为 VQA/分类子任务；或")
        lines.append("  3) 统一修复 Answer（在开头补 `[SEG]`），但要谨慎：如果问题本身不是分割任务，修复会引入噪声。")
        lines.append("")

    lines.append("## 8. 下一步建议（如果你还没跑 image-feature EDA）\n")
    lines.append("如果你已经有 `ct.npy/mask.npy`（prepare 后版本），强烈建议补做：")
    lines.append("- **ROI 体积（voxels）/ bbox 尺寸 / slices 覆盖**分布：小目标长尾会显著影响 Dice 和训练稳定性。")
    lines.append("- **空 ROI 检测**：prepare 的 resize 可能让极小目标消失（mask==Mask_ID 变 0）。")
    lines.append("- **连通域数量（components）**：多发病灶 vs 单发病灶。")
    lines.append("- **质检 overlay**：随机/按异常点导出 CT+mask 叠加图。")
    lines.append("这些结果会直接指导：采样策略（oversample small ROI）、loss（如 focal/Dice）、以及评估指标（Dice + Hausdorff）。\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("Saved report to:", out_path.resolve())


if __name__ == "__main__":
    main()
