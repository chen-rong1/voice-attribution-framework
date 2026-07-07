"""Load flattened business benchmark datasets from truth tables."""

from __future__ import annotations

import csv
from pathlib import Path

from app.benchmark.models import BenchmarkClip

DEFAULT_BUSINESS_TRUTH_FILENAME = "merged_truth.tsv"
DEFAULT_BUSINESS_PURE_LIST_FILENAME = "pure_test_files.txt"
MIXED_LABEL = "MIXED"


def load_business_benchmark_clips(
    dataset_dir: Path,
    *,
    truth_tsv_path: Path | None = None,
    pure_list_path: Path | None = None,
    exclude_mixed: bool = True,
) -> list[BenchmarkClip]:
    """Load a flattened business dataset from a TSV truth table."""

    resolved_truth_tsv = truth_tsv_path or dataset_dir / DEFAULT_BUSINESS_TRUTH_FILENAME
    resolved_pure_list = pure_list_path or dataset_dir / DEFAULT_BUSINESS_PURE_LIST_FILENAME
    pure_clip_names = (
        _load_pure_clip_names(resolved_pure_list) if resolved_pure_list.exists() else None
    )

    clips: list[BenchmarkClip] = []
    with resolved_truth_tsv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            clip_filename = row["测试片段"].strip()
            expected_label = row["预期标签"].strip()
            if exclude_mixed and expected_label == MIXED_LABEL:
                continue
            if pure_clip_names is not None and clip_filename not in pure_clip_names:
                continue

            audio_path = dataset_dir / clip_filename
            truth_label = row.get("真实身份集合", "").strip() or expected_label
            clips.append(
                BenchmarkClip(
                    clip_id=audio_path.stem,
                    audio_path=audio_path,
                    truth_label=truth_label,
                    expected_label=expected_label,
                    metadata={
                        "source_segment_count": _parse_int(row.get("原始片段数", "")),
                        "source_segments": row.get("原始片段", "").strip(),
                        "truth_table": resolved_truth_tsv.name,
                    },
                )
            )
    return clips


def _load_pure_clip_names(pure_list_path: Path) -> set[str]:
    with pure_list_path.open("r", encoding="utf-8-sig") as file:
        return {line.strip() for line in file if line.strip()}


def _parse_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
