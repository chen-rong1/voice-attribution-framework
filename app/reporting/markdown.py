"""Markdown report generation for benchmark runs."""

from __future__ import annotations

from pathlib import Path

from app.benchmark.models import BenchmarkRunResult
from app.reporting.models import ReportArtifact


def write_benchmark_markdown(
    result: BenchmarkRunResult,
    output_path: Path,
) -> ReportArtifact:
    """Write a Chinese Markdown report for one benchmark run."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = result.summary
    lines = [
        f"# {result.config.run_name} 正式报告",
        "",
        "## 一句话结论",
        "",
        (
            f"- 当前运行在 `{summary['dataset_name']}` 上共评测 `{summary['total']}` 条，"
            f"总体准确率 `{summary['accuracy']:.2%}`，"
            f"本人召回 `{summary['positive_recall']:.2%}`，"
            f"`UNKNOWN` 拒识 `{summary['unknown_reject_rate']:.2%}`。"
        ),
        "",
        "## 测试口径",
        "",
        f"- 运行名称：`{result.config.run_name}`",
        f"- 数据集：`{result.config.dataset_name}` / `{result.config.dataset_version}`",
        f"- Backend：`{result.config.backend_name}`",
        f"- 打分策略：`{result.config.scoring_strategy.value}`",
        f"- 阈值：`{result.config.threshold_value:.4f}`",
        f"- 排除 `MIXED`：`{result.config.exclude_mixed}`",
        "",
        "## 汇总指标",
        "",
        f"- 总样本数：`{summary['total']}`",
        f"- 总正确数：`{summary['correct']}`",
        f"- 总体准确率：`{summary['accuracy']:.2%}`",
        f"- 本人样本数：`{summary['positive_total']}`",
        f"- 本人命中数：`{summary['positive_correct']}`",
        f"- 本人召回：`{summary['positive_recall']:.2%}`",
        f"- 非本人样本数：`{summary['unknown_total']}`",
        f"- 非本人拒识数：`{summary['unknown_correct']}`",
        f"- 非本人拒识率：`{summary['unknown_reject_rate']:.2%}`",
        "",
        "## 逐条明细",
        "",
        "| 片段编号 | 预期标签 | 最终标签 | 决策 | 最高分 | 是否正确 |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for item in result.items:
        lines.append(
            f"| {item.clip_id} | {item.expected_label} | {item.final_label} | "
            f"{item.decision} | {item.best_score:.4f} | {'是' if item.is_correct else '否'} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ReportArtifact(artifact_type="markdown", path=output_path)
