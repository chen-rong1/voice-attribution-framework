from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from app.benchmark.business import load_business_benchmark_clips


def _write_tone(audio_path: Path, frequency: float, *, duration_sec: float = 1.5) -> None:
    sample_rate = 16000
    timeline = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * frequency * timeline).astype(np.float32)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, waveform, sample_rate)


def _write_truth_and_pure_list(dataset_dir: Path) -> tuple[Path, Path]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    truth_tsv_path = dataset_dir / "merged_truth.tsv"
    pure_list_path = dataset_dir / "pure_test_files.txt"
    with truth_tsv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["测试片段", "预期标签", "原始片段数", "真实身份集合", "原始片段"])
        writer.writerow(["clip_x.wav", "speaker_a", "1", "speaker_a", "clip_x.wav"])
        writer.writerow(["clip_u.wav", "UNKNOWN", "1", "UNKNOWN", "clip_u.wav"])
        writer.writerow(["clip_m.wav", "MIXED", "2", "UNKNOWN,speaker_a", "a.wav | b.wav"])
    pure_list_path.write_text("clip_x.wav\nclip_u.wav\n", encoding="utf-8")
    return truth_tsv_path, pure_list_path


def test_business_loader_filters_mixed_and_uses_pure_list(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "business_dataset"
    truth_tsv_path, pure_list_path = _write_truth_and_pure_list(dataset_dir)
    _write_tone(dataset_dir / "clip_x.wav", 220.0)
    _write_tone(dataset_dir / "clip_u.wav", 660.0)
    _write_tone(dataset_dir / "clip_m.wav", 330.0)

    clips = load_business_benchmark_clips(
        dataset_dir,
        truth_tsv_path=truth_tsv_path,
        pure_list_path=pure_list_path,
    )

    assert [clip.clip_id for clip in clips] == ["clip_x", "clip_u"]
    assert clips[0].expected_label == "speaker_a"
    assert clips[0].truth_label == "speaker_a"
    assert clips[0].evaluation_group == "internal_known"
    assert clips[1].expected_label == "UNKNOWN"
    assert clips[1].evaluation_group == "external_unknown"
    assert clips[1].metadata["source_segment_count"] == 1


def test_business_loader_supports_explicit_external_known_group(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "business_external_known"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    truth_tsv_path = dataset_dir / "merged_truth.tsv"
    with truth_tsv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["测试片段", "预期标签", "真实身份集合", "评测分组"])
        writer.writerow(["clip_known.wav", "speaker_b", "speaker_b", "external_known"])
    _write_tone(dataset_dir / "clip_known.wav", 330.0)

    clips = load_business_benchmark_clips(dataset_dir, truth_tsv_path=truth_tsv_path)

    assert len(clips) == 1
    assert clips[0].expected_label == "speaker_b"
    assert clips[0].truth_label == "speaker_b"
    assert clips[0].evaluation_group == "external_known"


def test_business_cli_benchmark_script_runs_with_direct_enroll_files(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = tmp_path / "business_dataset"
    output_root = tmp_path / "outputs"
    truth_tsv_path, pure_list_path = _write_truth_and_pure_list(dataset_dir)

    enroll_path = tmp_path / "enroll_x.wav"
    _write_tone(enroll_path, 220.0)
    _write_tone(dataset_dir / "clip_x.wav", 220.0)
    _write_tone(dataset_dir / "clip_u.wav", 660.0)
    _write_tone(dataset_dir / "clip_m.wav", 330.0)

    command = [
        sys.executable,
        str(project_root / "scripts" / "run_benchmark.py"),
        "--business-dataset-dir",
        str(dataset_dir),
        "--business-truth-tsv",
        str(truth_tsv_path),
        "--business-pure-list",
        str(pure_list_path),
        "--enroll-speaker",
        "speaker_a",
        "--enroll-file",
        str(enroll_path),
        "--output-dir",
        str(output_root),
        "--run-name",
        "business_smoke",
        "--dataset-name",
        "business_demo",
        "--project-root",
        str(project_root),
        "--threshold",
        "0.5",
    ]
    result = subprocess.run(
        command,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "business_demo" in result.stdout
    assert (output_root / "business_smoke_测试总表.tsv").exists()
    assert (output_root / "business_smoke_正式报告.md").exists()
    assert (output_root / "business_smoke_摘要.json").exists()


def test_business_cli_benchmark_script_runs_with_enroll_list(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = tmp_path / "business_dataset"
    output_root = tmp_path / "outputs"
    truth_tsv_path, pure_list_path = _write_truth_and_pure_list(dataset_dir)

    enroll_path = tmp_path / "enroll_x.wav"
    enroll_list = tmp_path / "enroll_list.txt"
    _write_tone(enroll_path, 220.0)
    _write_tone(dataset_dir / "clip_x.wav", 220.0)
    _write_tone(dataset_dir / "clip_u.wav", 660.0)
    enroll_list.write_text(f"{enroll_path}\n", encoding="utf-8")

    command = [
        sys.executable,
        str(project_root / "scripts" / "run_benchmark.py"),
        "--business-dataset-dir",
        str(dataset_dir),
        "--business-truth-tsv",
        str(truth_tsv_path),
        "--business-pure-list",
        str(pure_list_path),
        "--enroll-speaker",
        "speaker_a",
        "--enroll-list",
        str(enroll_list),
        "--output-dir",
        str(output_root),
        "--run-name",
        "business_enroll_list",
        "--dataset-name",
        "business_demo",
        "--project-root",
        str(project_root),
        "--threshold",
        "0.5",
    ]
    result = subprocess.run(
        command,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "business_demo" in result.stdout
    assert (output_root / "business_enroll_list_摘要.json").exists()
