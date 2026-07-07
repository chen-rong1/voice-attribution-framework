from pathlib import Path

import numpy as np
import soundfile as sf

from app.benchmark.models import BenchmarkClip, BenchmarkRunConfig
from app.benchmark.runner import BenchmarkRunner
from app.reporting.tsv import write_benchmark_tsv
from app.scoring.models import ScoringStrategy
from app.services.bootstrap import register_default_backends
from app.services.container import FrameworkContainer
from app.services.identification import EnrollmentRecord, IdentificationService


def _write_tone(audio_path: Path, frequency: float, *, duration_sec: float = 1.5) -> None:
    sample_rate = 16000
    timeline = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * frequency * timeline).astype(np.float32)
    sf.write(audio_path, waveform, sample_rate)


def test_benchmark_runner_and_tsv_export(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    container = FrameworkContainer()
    register_default_backends(project_root=project_root, registry=container.backend_registry)
    backend = container.backend_registry.get("wespeaker-ecapa1024-lm-onnx")
    backend.load()

    alice = tmp_path / "alice.wav"
    bob = tmp_path / "bob.wav"
    query_alice = tmp_path / "query_alice.wav"
    query_bob = tmp_path / "query_bob.wav"
    _write_tone(alice, 220.0)
    _write_tone(bob, 440.0)
    _write_tone(query_alice, 220.0)
    _write_tone(query_bob, 440.0)

    service = IdentificationService(backend)
    runner = BenchmarkRunner(service)
    result = runner.run(
        config=BenchmarkRunConfig(
            run_name="smoke",
            dataset_name="synthetic",
            dataset_version="v1",
            backend_name="wespeaker-ecapa1024-lm-onnx",
            scoring_strategy=ScoringStrategy.CENTER,
            threshold_value=0.5,
        ),
        enrollments=[
            EnrollmentRecord(speaker_id="alice", audio_paths=[alice]),
            EnrollmentRecord(speaker_id="bob", audio_paths=[bob]),
        ],
        clips=[
            BenchmarkClip(
                clip_id="clip-1",
                audio_path=query_alice,
                truth_label="alice",
                expected_label="alice",
            ),
            BenchmarkClip(
                clip_id="clip-2",
                audio_path=query_bob,
                truth_label="bob",
                expected_label="bob",
            ),
        ],
    )

    assert result.total == 2
    assert result.correct == 2
    assert result.accuracy == 1.0

    artifact = write_benchmark_tsv(result, tmp_path / "benchmark.tsv")
    text = artifact.path.read_text(encoding="utf-8")
    assert "运行名称" in text
    assert "clip-1" in text
    assert "alice" in text
