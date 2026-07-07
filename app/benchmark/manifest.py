"""CSV manifest loaders for real benchmark datasets."""

from __future__ import annotations

import csv
from pathlib import Path

from app.benchmark.models import BenchmarkClip
from app.common.constants import DEFAULT_REJECT_LABEL
from app.services.identification import EnrollmentRecord


def load_from_manifest(
    *,
    dataset_root: Path,
    manifest_path: Path,
) -> tuple[list[EnrollmentRecord], list[BenchmarkClip]]:
    """Load enrollments and benchmark clips from a manifest CSV."""

    enrollment_map: dict[str, list[Path]] = {}
    clip_rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            kind = row["kind"]
            audio_path = dataset_root / row["output_rel_path"]
            if kind == "enroll":
                enrollment_map.setdefault(row["agent"], []).append(audio_path)
            elif kind == "attribution":
                clip_rows.append(row)

    enrolled_agents = set(enrollment_map.keys())
    enrollments = [
        EnrollmentRecord(speaker_id=speaker_id, audio_paths=audio_paths)
        for speaker_id, audio_paths in sorted(enrollment_map.items())
    ]
    clips: list[BenchmarkClip] = []
    for row in clip_rows:
        agent = row["agent"]
        expected_label = agent if agent in enrolled_agents else DEFAULT_REJECT_LABEL
        audio_path = dataset_root / row["output_rel_path"]
        clips.append(
            BenchmarkClip(
                clip_id=audio_path.stem,
                audio_path=audio_path,
                truth_label=agent,
                expected_label=expected_label,
                metadata={
                    "trial_label": row.get("trial_label", ""),
                    "source_file": row.get("source_file", ""),
                    "note": row.get("note", ""),
                },
            )
        )
    return enrollments, clips
