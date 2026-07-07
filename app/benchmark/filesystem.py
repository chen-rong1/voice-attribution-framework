"""Filesystem dataset loaders for enrollment and benchmark clips."""

from __future__ import annotations

import re
from pathlib import Path

from app.benchmark.models import BenchmarkClip
from app.services.identification import EnrollmentRecord

STRICT_ENROLL_PATTERN = re.compile(r"^enroll_[^_]+_([A-Za-z0-9]+)_\d+$")
STRICT_CLIP_PATTERN = re.compile(r"^clip_[^_]+_([A-Za-z0-9]+)_(.+)$")


def _iter_audio_files(directory: Path) -> list[Path]:
    patterns = ("*.wav", "*.flac", "*.mp3", "*.m4a")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(directory.glob(pattern)))
    return files


def load_enrollments_from_directory(enroll_root: Path) -> list[EnrollmentRecord]:
    """Load enrollments from `speaker_id/*.wav` style directory structure."""

    subdirectories = sorted(path for path in enroll_root.iterdir() if path.is_dir())
    if subdirectories:
        return _load_nested_enrollments(subdirectories)
    return _load_flat_enrollments(enroll_root)


def _load_nested_enrollments(speaker_dirs: list[Path]) -> list[EnrollmentRecord]:
    enrollments: list[EnrollmentRecord] = []
    for speaker_dir in speaker_dirs:
        audio_paths = _iter_audio_files(speaker_dir)
        if not audio_paths:
            continue
        enrollments.append(
            EnrollmentRecord(
                speaker_id=speaker_dir.name,
                audio_paths=audio_paths,
            )
        )
    return enrollments


def load_benchmark_clips_from_directory(test_root: Path) -> list[BenchmarkClip]:
    """Load benchmark clips from `label/*.wav` style directory structure."""

    label_dirs = sorted(path for path in test_root.iterdir() if path.is_dir())
    if label_dirs:
        return _load_nested_benchmark_clips(label_dirs)
    return _load_flat_benchmark_clips(test_root)


def _load_nested_benchmark_clips(label_dirs: list[Path]) -> list[BenchmarkClip]:
    clips: list[BenchmarkClip] = []
    for label_dir in label_dirs:
        expected_label = label_dir.name
        for audio_path in _iter_audio_files(label_dir):
            clips.append(
                BenchmarkClip(
                    clip_id=audio_path.stem,
                    audio_path=audio_path,
                    truth_label=expected_label,
                    expected_label=expected_label,
                    metadata={"label_dir": label_dir.name},
                )
            )
    return clips


def _load_flat_enrollments(enroll_root: Path) -> list[EnrollmentRecord]:
    grouped_audio_paths: dict[str, list[Path]] = {}
    for audio_path in _iter_audio_files(enroll_root):
        match = STRICT_ENROLL_PATTERN.match(audio_path.stem)
        if match is None:
            continue
        speaker_id = match.group(1)
        grouped_audio_paths.setdefault(speaker_id, []).append(audio_path)
    return [
        EnrollmentRecord(speaker_id=speaker_id, audio_paths=sorted(audio_paths))
        for speaker_id, audio_paths in sorted(grouped_audio_paths.items())
    ]


def _load_flat_benchmark_clips(test_root: Path) -> list[BenchmarkClip]:
    clips: list[BenchmarkClip] = []
    for audio_path in _iter_audio_files(test_root):
        match = STRICT_CLIP_PATTERN.match(audio_path.stem)
        if match is None:
            continue
        truth_label = match.group(1)
        trial_suffix = match.group(2)
        expected_label = "UNKNOWN" if "unknown" in trial_suffix else truth_label
        clips.append(
            BenchmarkClip(
                clip_id=audio_path.stem,
                audio_path=audio_path,
                truth_label=truth_label,
                expected_label=expected_label,
                metadata={"trial_suffix": trial_suffix, "loader": "flat_strict"},
            )
        )
    return clips
