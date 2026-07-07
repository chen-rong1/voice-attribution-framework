from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from app.benchmark.manifest import load_from_manifest


def _write_tone(audio_path: Path, frequency: float, *, duration_sec: float = 1.5) -> None:
    sample_rate = 16000
    timeline = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * frequency * timeline).astype(np.float32)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, waveform, sample_rate)


def _write_manifest(manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["kind", "output_rel_path", "source_file", "start_sec", "end_sec", "agent", "trial_label", "note"]
        )
        writer.writerow(
            ["enroll", "processed/enroll/eval/enroll_A_1.wav", "raw/a.wav", "0", "1.5", "A", "enroll", "demo"]
        )
        writer.writerow(
            ["enroll", "processed/enroll/eval/enroll_B_1.wav", "raw/b.wav", "0", "1.5", "B", "enroll", "demo"]
        )
        writer.writerow(
            ["attribution", "processed/attribution/eval/clip_A.wav", "raw/a.wav", "0", "1.5", "A", "pos", "demo"]
        )
        writer.writerow(
            ["attribution", "processed/attribution/eval/clip_C.wav", "raw/c.wav", "0", "1.5", "C", "unknown", "demo"]
        )


def test_manifest_loader_and_cli_script(tmp_path: Path) -> None:
    dataset_root = tmp_path / "strict_dataset"
    manifest_path = dataset_root / "meta" / "manifest.csv"
    _write_manifest(manifest_path)
    _write_tone(dataset_root / "processed" / "enroll" / "eval" / "enroll_A_1.wav", 220.0)
    _write_tone(dataset_root / "processed" / "enroll" / "eval" / "enroll_B_1.wav", 440.0)
    _write_tone(dataset_root / "processed" / "attribution" / "eval" / "clip_A.wav", 220.0)
    _write_tone(dataset_root / "processed" / "attribution" / "eval" / "clip_C.wav", 660.0)

    enrollments, clips = load_from_manifest(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
    )
    assert len(enrollments) == 2
    assert len(clips) == 2
    assert clips[1].expected_label == "UNKNOWN"

    project_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "outputs"
    command = [
        sys.executable,
        str(project_root / "scripts" / "run_benchmark.py"),
        "--output-dir",
        str(output_root),
        "--run-name",
        "manifest_smoke",
        "--dataset-name",
        "strict_demo",
        "--manifest-path",
        str(manifest_path),
        "--dataset-root",
        str(dataset_root),
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
    assert "strict_demo" in result.stdout
    assert (output_root / "manifest_smoke_测试总表.tsv").exists()
    assert (output_root / "manifest_smoke_正式报告.md").exists()
    assert (output_root / "manifest_smoke_摘要.json").exists()
