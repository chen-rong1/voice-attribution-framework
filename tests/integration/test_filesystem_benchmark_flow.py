from pathlib import Path

import numpy as np
import soundfile as sf

from app.benchmark.filesystem import (
    load_benchmark_clips_from_directory,
    load_enrollments_from_directory,
)
from app.benchmark.models import BenchmarkRunConfig
from app.benchmark.runner import BenchmarkRunner
from app.reporting.markdown import write_benchmark_markdown
from app.reporting.tsv import write_benchmark_tsv
from app.scoring.models import ScoringStrategy
from app.services.bootstrap import register_default_backends
from app.services.container import FrameworkContainer
from app.services.identification import IdentificationService


def _write_tone(audio_path: Path, frequency: float, *, duration_sec: float = 1.5) -> None:
    sample_rate = 16000
    timeline = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * frequency * timeline).astype(np.float32)
    sf.write(audio_path, waveform, sample_rate)


def test_filesystem_benchmark_flow_generates_reports(tmp_path: Path) -> None:
    enroll_root = tmp_path / "enrollments"
    test_root = tmp_path / "testset"
    (enroll_root / "alice").mkdir(parents=True)
    (enroll_root / "bob").mkdir(parents=True)
    (test_root / "alice").mkdir(parents=True)
    (test_root / "UNKNOWN").mkdir(parents=True)

    _write_tone(enroll_root / "alice" / "a1.wav", 220.0)
    _write_tone(enroll_root / "bob" / "b1.wav", 440.0)
    _write_tone(test_root / "alice" / "q1.wav", 220.0)
    _write_tone(test_root / "UNKNOWN" / "u1.wav", 660.0)

    project_root = Path(__file__).resolve().parents[2]
    container = FrameworkContainer()
    register_default_backends(project_root=project_root, registry=container.backend_registry)
    backend = container.backend_registry.get("wespeaker-ecapa1024-lm-onnx")
    backend.load()

    runner = BenchmarkRunner(IdentificationService(backend))
    result = runner.run(
        config=BenchmarkRunConfig(
            run_name="filesystem-smoke",
            dataset_name="filesystem-demo",
            dataset_version="v1",
            backend_name=backend.backend_name,
            scoring_strategy=ScoringStrategy.CENTER,
            threshold_value=0.5,
        ),
        enrollments=load_enrollments_from_directory(enroll_root),
        clips=load_benchmark_clips_from_directory(test_root),
    )

    tsv_artifact = write_benchmark_tsv(result, tmp_path / "result.tsv")
    markdown_artifact = write_benchmark_markdown(result, tmp_path / "report.md")

    assert result.total == 2
    assert "运行名称" in tsv_artifact.path.read_text(encoding="utf-8")
    markdown_text = markdown_artifact.path.read_text(encoding="utf-8")
    assert "一句话结论" in markdown_text
    assert "总体准确率" in markdown_text


def test_filesystem_loader_supports_flat_strict_layout(tmp_path: Path) -> None:
    enroll_root = tmp_path / "enroll"
    test_root = tmp_path / "eval"
    enroll_root.mkdir(parents=True)
    test_root.mkdir(parents=True)

    _write_tone(enroll_root / "enroll_ES2005_A_1.wav", 220.0)
    _write_tone(enroll_root / "enroll_ES2005_A_2.wav", 230.0)
    _write_tone(enroll_root / "enroll_ES2005_B_1.wav", 440.0)
    _write_tone(test_root / "clip_ES2005_A_pos_1.wav", 220.0)
    _write_tone(test_root / "clip_ES2005_B_neg_for_A_1.wav", 440.0)
    _write_tone(test_root / "clip_ES2005_C_unknown_1.wav", 660.0)

    enrollments = load_enrollments_from_directory(enroll_root)
    clips = load_benchmark_clips_from_directory(test_root)

    assert [record.speaker_id for record in enrollments] == ["A", "B"]
    assert len(enrollments[0].audio_paths) == 2
    assert len(clips) == 3
    assert clips[0].truth_label == "A"
    assert clips[0].expected_label == "A"
    assert clips[1].truth_label == "B"
    assert clips[1].expected_label == "B"
    assert clips[2].truth_label == "C"
    assert clips[2].expected_label == "UNKNOWN"
