"""Chinese TSV exporters for benchmark result tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.benchmark.models import BenchmarkRunResult
from app.reporting.models import ReportArtifact


def write_benchmark_tsv(result: BenchmarkRunResult, output_path: Path) -> ReportArtifact:
    """Write a Chinese TSV summary table for one benchmark run."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(
            [
                "运行名称",
                "数据集",
                "片段编号",
                "音频路径",
                "真实标签",
                "预期标签",
                "评测分组",
                "最终标签",
                "决策",
                "判决原因",
                "校准状态",
                "判决证据",
                "最高分",
                "阈值",
                "时延(ms)",
                "是否正确",
                "分数明细",
                "判决元数据",
            ]
        )
        for item in result.items:
            writer.writerow(
                [
                    result.config.run_name,
                    result.config.dataset_name,
                    item.clip_id,
                    str(item.audio_path),
                    item.truth_label,
                    item.expected_label,
                    item.evaluation_group,
                    item.final_label,
                    item.decision,
                    str(item.metadata.get("decision_reason", "")),
                    str(item.metadata.get("calibration_status", "")),
                    json.dumps(
                        item.metadata.get("decision_evidence", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    f"{item.best_score:.4f}",
                    f"{item.threshold_value:.4f}",
                    f"{item.latency_ms:.3f}",
                    "是" if item.is_correct else "否",
                    json.dumps(item.score_breakdown, ensure_ascii=False, sort_keys=True),
                    json.dumps(item.metadata, ensure_ascii=False, sort_keys=True),
                ]
            )
    return ReportArtifact(artifact_type="tsv", path=output_path)
