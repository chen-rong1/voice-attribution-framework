"""Markdown report generation for benchmark runs."""

from __future__ import annotations

import json
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
            f"`UNKNOWN` 拒识 `{summary['unknown_reject_rate']:.2%}`，"
            f"`REVIEW` `{summary['review_count']}` 条。"
        ),
        "",
        "## 测试口径",
        "",
        f"- 运行名称：`{result.config.run_name}`",
        f"- 数据集：`{result.config.dataset_name}` / `{result.config.dataset_version}`",
        f"- 数据集角色：`{summary['dataset_role']}`",
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
        f"- 平均时延：`{summary['average_latency_ms']:.3f} ms`",
        f"- 最大时延：`{summary['max_latency_ms']:.3f} ms`",
        f"- 本人样本数：`{summary['positive_total']}`",
        f"- 本人命中数：`{summary['positive_correct']}`",
        f"- 本人召回：`{summary['positive_recall']:.2%}`",
        f"- 非本人样本数：`{summary['unknown_total']}`",
        f"- 非本人拒识数：`{summary['unknown_correct']}`",
        f"- 非本人拒识率：`{summary['unknown_reject_rate']:.2%}`",
        f"- 接受数：`{summary['accept_count']}`",
        f"- 拒识数：`{summary['reject_count']}`",
        f"- 复核数：`{summary['review_count']}`",
        f"- 误接收数：`{summary['false_accept_count']}`",
        f"- 误拒识数：`{summary['false_reject_count']}`",
        f"- 外部已知样本数：`{summary['external_known_total']}`",
        f"- 外部已知 Top1 准确率：`{summary['external_known_top1_accuracy']:.2%}`",
        f"- 外部未知样本数：`{summary['external_unknown_total']}`",
        f"- 外部未知拒识率：`{summary['external_unknown_reject_rate']:.2%}`",
        f"- 高风险误接收数：`{summary['high_risk_false_accept_count']}`",
        f"- 接受原因分布：`{summary['accept_reason_counts']}`",
        f"- 拒识原因分布：`{summary['reject_reason_counts']}`",
        f"- 复核原因分布：`{summary['review_reason_counts']}`",
        f"- 校准状态分布：`{summary['calibration_status_counts']}`",
        f"- Heldout 校准样本数：`{summary['heldout_calibrated_count']}`",
        f"- 判决理由统计：`{summary['decision_reason_stats']}`",
        "",
        "## 逐条明细",
        "",
        "| 片段编号 | 预期标签 | 分组 | 最终标签 | 决策 | 判决原因 | 校准状态 | 判决证据 | 最高分 | 时延(ms) | 是否正确 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in result.items:
        evidence = _format_evidence_preview(item.metadata.get("decision_evidence", {}))
        lines.append(
            f"| {item.clip_id} | {item.expected_label} | {item.evaluation_group} | {item.final_label} | "
            f"{item.decision} | {item.metadata.get('decision_reason', '')} | "
            f"{item.metadata.get('calibration_status', '')} | "
            f"{evidence} | {item.best_score:.4f} | {item.latency_ms:.3f} | "
            f"{'是' if item.is_correct else '否'} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ReportArtifact(artifact_type="markdown", path=output_path)


def _format_evidence_preview(evidence: object) -> str:
    return json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
