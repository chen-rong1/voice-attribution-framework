from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def _write_tone(audio_path: Path, frequency: float, *, duration_sec: float = 1.5) -> None:
    sample_rate = 16000
    timeline = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * frequency * timeline).astype(np.float32)
    sf.write(audio_path, waveform, sample_rate)


def test_cli_benchmark_script_generates_all_outputs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    enroll_root = tmp_path / "enrollments"
    test_root = tmp_path / "testset"
    output_root = tmp_path / "outputs"
    (enroll_root / "alice").mkdir(parents=True)
    (test_root / "alice").mkdir(parents=True)
    (test_root / "UNKNOWN").mkdir(parents=True)

    _write_tone(enroll_root / "alice" / "a1.wav", 220.0)
    _write_tone(test_root / "alice" / "q1.wav", 220.0)
    _write_tone(test_root / "UNKNOWN" / "u1.wav", 660.0)

    command = [
        sys.executable,
        str(project_root / "scripts" / "run_benchmark.py"),
        "--enroll-dir",
        str(enroll_root),
        "--test-dir",
        str(test_root),
        "--output-dir",
        str(output_root),
        "--run-name",
        "cli_smoke",
        "--dataset-name",
        "cli_dataset",
        "--dataset-version",
        "v1",
        "--threshold",
        "0.5",
        "--project-root",
        str(project_root),
    ]
    result = subprocess.run(
        command,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "accuracy" in result.stdout
    assert (output_root / "cli_smoke_测试总表.tsv").exists()
    assert (output_root / "cli_smoke_正式报告.md").exists()
    assert (output_root / "cli_smoke_摘要.json").exists()


def test_cli_benchmark_script_supports_scoring_config(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    enroll_root = tmp_path / "enrollments"
    test_root = tmp_path / "testset"
    output_root = tmp_path / "outputs"
    scoring_config = tmp_path / "scoring.yaml"
    (enroll_root / "alice").mkdir(parents=True)
    (test_root / "alice").mkdir(parents=True)
    (test_root / "UNKNOWN").mkdir(parents=True)

    _write_tone(enroll_root / "alice" / "a1.wav", 220.0)
    _write_tone(test_root / "alice" / "q1.wav", 220.0)
    _write_tone(test_root / "UNKNOWN" / "u1.wav", 660.0)
    scoring_config.write_text(
        "\n".join(
            [
                "strategy: max",
                "threshold: 0.50",
                "profile_aggregation_strategy: center",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(project_root / "scripts" / "run_benchmark.py"),
        "--enroll-dir",
        str(enroll_root),
        "--test-dir",
        str(test_root),
        "--output-dir",
        str(output_root),
        "--run-name",
        "cli_scoring_config",
        "--dataset-name",
        "cli_dataset",
        "--dataset-version",
        "v1",
        "--scoring-config",
        str(scoring_config),
        "--project-root",
        str(project_root),
    ]
    result = subprocess.run(
        command,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "max" in result.stdout
    assert (output_root / "cli_scoring_config_测试总表.tsv").exists()


def test_cli_benchmark_script_supports_flat_strict_layout(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    enroll_root = tmp_path / "enroll_eval"
    test_root = tmp_path / "attr_eval"
    output_root = tmp_path / "outputs"
    enroll_root.mkdir(parents=True)
    test_root.mkdir(parents=True)

    _write_tone(enroll_root / "enroll_ES2005_A_1.wav", 220.0)
    _write_tone(enroll_root / "enroll_ES2005_B_1.wav", 440.0)
    _write_tone(test_root / "clip_ES2005_A_pos_1.wav", 220.0)
    _write_tone(test_root / "clip_ES2005_C_unknown_1.wav", 660.0)

    command = [
        sys.executable,
        str(project_root / "scripts" / "run_benchmark.py"),
        "--enroll-dir",
        str(enroll_root),
        "--test-dir",
        str(test_root),
        "--output-dir",
        str(output_root),
        "--run-name",
        "cli_flat_strict",
        "--dataset-name",
        "strict_flat_demo",
        "--dataset-version",
        "eval",
        "--threshold",
        "0.5",
        "--project-root",
        str(project_root),
    ]
    result = subprocess.run(
        command,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "strict_flat_demo" in result.stdout
    assert (output_root / "cli_flat_strict_测试总表.tsv").exists()
