"""JSON summary export helpers for benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path

from app.benchmark.models import BenchmarkRunResult
from app.reporting.models import ReportArtifact


def write_benchmark_json_summary(
    result: BenchmarkRunResult,
    output_path: Path,
) -> ReportArtifact:
    """Write a machine-readable JSON summary for one benchmark run."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": result.summary,
        "items": [
            {
                "clip_id": item.clip_id,
                "audio_path": str(item.audio_path),
                "truth_label": item.truth_label,
                "expected_label": item.expected_label,
                "final_label": item.final_label,
                "decision": item.decision,
                "best_score": item.best_score,
                "threshold_value": item.threshold_value,
                "is_correct": item.is_correct,
                "score_breakdown": item.score_breakdown,
            }
            for item in result.items
        ],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ReportArtifact(artifact_type="json", path=output_path)
