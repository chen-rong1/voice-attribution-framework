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
                "最终标签",
                "决策",
                "最高分",
                "阈值",
                "是否正确",
                "分数明细",
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
                    item.final_label,
                    item.decision,
                    f"{item.best_score:.4f}",
                    f"{item.threshold_value:.4f}",
                    "是" if item.is_correct else "否",
                    json.dumps(item.score_breakdown, ensure_ascii=False, sort_keys=True),
                ]
            )
    return ReportArtifact(artifact_type="tsv", path=output_path)
