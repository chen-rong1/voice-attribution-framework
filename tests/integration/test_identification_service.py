from pathlib import Path

import numpy as np
import soundfile as sf

from app.profiles.calibration import HeldoutCalibrationTrial
from app.benchmark.models import BenchmarkClip
from app.scoring.models import DecisionLabel, ScoringStrategy
from app.services.identification import EnrollmentRecord, IdentificationService
from app.services.bootstrap import register_default_backends
from app.services.container import FrameworkContainer


def _write_tone(audio_path: Path, frequency: float, *, duration_sec: float = 1.5) -> None:
    sample_rate = 16000
    timeline = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * frequency * timeline).astype(np.float32)
    sf.write(audio_path, waveform, sample_rate)


def test_identification_service_accepts_matching_query(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    container = FrameworkContainer()
    register_default_backends(project_root=project_root, registry=container.backend_registry)
    backend = container.backend_registry.get("wespeaker-ecapa1024-lm-onnx")
    backend.load()

    speaker_a = tmp_path / "speaker_a.wav"
    speaker_b = tmp_path / "speaker_b.wav"
    query = tmp_path / "query.wav"
    _write_tone(speaker_a, 220.0)
    _write_tone(speaker_b, 440.0)
    _write_tone(query, 220.0)

    service = IdentificationService(backend)
    profiles = service.build_profiles(
        [
            EnrollmentRecord(speaker_id="alice", audio_paths=[speaker_a]),
            EnrollmentRecord(speaker_id="bob", audio_paths=[speaker_b]),
        ]
    )
    result = service.identify(
        query,
        profiles,
        threshold_value=0.5,
        scoring_strategy=ScoringStrategy.CENTER,
    )

    assert result.decision == DecisionLabel.ACCEPT
    assert result.final_label == "alice"


def test_identification_service_rejects_with_strict_threshold(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    container = FrameworkContainer()
    register_default_backends(project_root=project_root, registry=container.backend_registry)
    backend = container.backend_registry.get("wespeaker-ecapa1024-lm-onnx")
    backend.load()

    speaker_a = tmp_path / "speaker_a.wav"
    query = tmp_path / "query.wav"
    _write_tone(speaker_a, 220.0)
    _write_tone(query, 220.0)

    service = IdentificationService(backend)
    profiles = service.build_profiles(
        [EnrollmentRecord(speaker_id="alice", audio_paths=[speaker_a])]
    )
    result = service.identify(
        query,
        profiles,
        threshold_value=1.01,
        scoring_strategy=ScoringStrategy.CENTER,
    )

    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"


def test_identification_service_builds_quality_weighted_profiles_by_default(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    container = FrameworkContainer()
    register_default_backends(project_root=project_root, registry=container.backend_registry)
    backend = container.backend_registry.get("wespeaker-ecapa1024-lm-onnx")
    backend.load()

    speaker_a_1 = tmp_path / "speaker_a_1.wav"
    speaker_a_2 = tmp_path / "speaker_a_2.wav"
    _write_tone(speaker_a_1, 220.0, duration_sec=1.0)
    _write_tone(speaker_a_2, 220.0, duration_sec=3.0)

    service = IdentificationService(backend)
    profiles = service.build_profiles(
        [EnrollmentRecord(speaker_id="alice", audio_paths=[speaker_a_1, speaker_a_2])]
    )

    profile = profiles[0]
    assert profile.aggregation_strategy == "quality_weighted_center"
    assert profile.metadata["sample_count"] == 2
    assert profile.metadata["avg_quality_score"] > 0
    assert all(member.weight_value > 0 for member in profile.members)


def test_identification_service_applies_heldout_calibration_trials(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    container = FrameworkContainer()
    register_default_backends(project_root=project_root, registry=container.backend_registry)
    backend = container.backend_registry.get("wespeaker-ecapa1024-lm-onnx")
    backend.load()

    speaker_a_1 = tmp_path / "speaker_a_1.wav"
    speaker_a_2 = tmp_path / "speaker_a_2.wav"
    _write_tone(speaker_a_1, 220.0, duration_sec=1.0)
    _write_tone(speaker_a_2, 220.0, duration_sec=3.0)

    service = IdentificationService(backend)
    profiles = service.build_profiles(
        [EnrollmentRecord(speaker_id="alice", audio_paths=[speaker_a_1, speaker_a_2])],
        heldout_calibration_trials=[
            HeldoutCalibrationTrial(speaker_id="alice", raw_score=0.91, is_target=True),
            HeldoutCalibrationTrial(speaker_id="alice", raw_score=0.88, is_target=True),
            HeldoutCalibrationTrial(speaker_id="alice", raw_score=0.41, is_target=False),
            HeldoutCalibrationTrial(speaker_id="alice", raw_score=0.38, is_target=False),
        ],
    )

    profile = profiles[0]
    assert profile.metadata["calibration_status"] == "heldout_calibrated"
    assert profile.metadata["calibration_type"] == "linear_heldout"
    assert profile.metadata["heldout_trial_count"] == 4


def test_identification_service_builds_heldout_trials_from_clips(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    container = FrameworkContainer()
    register_default_backends(project_root=project_root, registry=container.backend_registry)
    backend = container.backend_registry.get("wespeaker-ecapa1024-lm-onnx")
    backend.load()

    speaker_a = tmp_path / "speaker_a.wav"
    speaker_b = tmp_path / "speaker_b.wav"
    query_a = tmp_path / "query_a.wav"
    query_unknown = tmp_path / "query_unknown.wav"
    _write_tone(speaker_a, 220.0)
    _write_tone(speaker_b, 440.0)
    _write_tone(query_a, 220.0)
    _write_tone(query_unknown, 660.0)

    service = IdentificationService(backend)
    trials = service.build_heldout_calibration_trials(
        [
            EnrollmentRecord(speaker_id="alice", audio_paths=[speaker_a]),
            EnrollmentRecord(speaker_id="bob", audio_paths=[speaker_b]),
        ],
        [
            BenchmarkClip(
                clip_id="known",
                audio_path=query_a,
                truth_label="alice",
                expected_label="alice",
                evaluation_group="external_known",
            ),
            BenchmarkClip(
                clip_id="unknown",
                audio_path=query_unknown,
                truth_label="UNKNOWN",
                expected_label="UNKNOWN",
                evaluation_group="external_unknown",
            ),
        ],
        scoring_strategy=ScoringStrategy.CENTER,
    )

    alice_trials = [trial for trial in trials if trial.speaker_id == "alice"]
    bob_trials = [trial for trial in trials if trial.speaker_id == "bob"]
    assert len(alice_trials) == 4
    assert len(bob_trials) == 4
    assert sum(1 for trial in alice_trials if trial.is_target) >= 2
    assert sum(1 for trial in bob_trials if not trial.is_target) >= 3
