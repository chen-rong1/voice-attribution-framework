import numpy as np

from app.embedding_backends.models import EmbeddingResult
from app.profiles.builder import build_speaker_profile, finalize_speaker_profiles
from app.profiles.calibration import HeldoutCalibrationTrial
from app.profiles.models import ProfileRiskLevel


def _embedding_result(
    sample_id: str,
    vector: list[float],
    *,
    duration_sec: float = 2.0,
    quality_score: float = 0.9,
) -> EmbeddingResult:
    embedding = np.asarray(vector, dtype=np.float32)
    return EmbeddingResult(
        sample_id=sample_id,
        backend_name="dummy",
        backend_version="0.1.0",
        feature_version="fbank80_v1",
        embedding=embedding,
        embedding_dim=embedding.shape[0],
        duration_sec=duration_sec,
        quality_score=quality_score,
    )


def test_build_speaker_profile_populates_profile_statistics() -> None:
    profile = build_speaker_profile(
        "alice",
        [
            _embedding_result("alice-1", [1.0, 0.0]),
            _embedding_result("alice-2", [0.96, 0.28]),
            _embedding_result("alice-3", [0.94, 0.34]),
        ],
    )

    assert profile.center_vector.shape == (2,)
    assert len(profile.member_vectors) == 3
    assert len(profile.sub_centers) >= 1
    assert profile.intra_score_mean > 0.9
    assert profile.open_set_floor > 0.15
    assert profile.metadata["sample_count"] == 3


def test_finalize_speaker_profiles_adds_impostor_statistics() -> None:
    profiles = finalize_speaker_profiles(
        [
            build_speaker_profile(
                "alice",
                [
                    _embedding_result("alice-1", [1.0, 0.0]),
                    _embedding_result("alice-2", [0.97, 0.24]),
                ],
            ),
            build_speaker_profile(
                "bob",
                [
                    _embedding_result("bob-1", [0.0, 1.0]),
                    _embedding_result("bob-2", [0.18, 0.98]),
                ],
            ),
            build_speaker_profile(
                "carol",
                [
                    _embedding_result("carol-1", [-1.0, 0.0]),
                    _embedding_result("carol-2", [-0.98, 0.12]),
                ],
            ),
        ]
    )

    alice = next(profile for profile in profiles if profile.speaker_id == "alice")
    assert alice.impostor_score_std > 0.0
    assert alice.calibrated_threshold == alice.open_set_floor
    assert alice.metadata["risk_level"] in {
        ProfileRiskLevel.LOW.value,
        ProfileRiskLevel.MEDIUM.value,
        ProfileRiskLevel.HIGH.value,
    }


def test_finalize_speaker_profiles_applies_heldout_calibration() -> None:
    profiles = finalize_speaker_profiles(
        [
            build_speaker_profile(
                "alice",
                [
                    _embedding_result("alice-1", [1.0, 0.0]),
                    _embedding_result("alice-2", [0.97, 0.24]),
                ],
            ),
            build_speaker_profile(
                "bob",
                [
                    _embedding_result("bob-1", [0.0, 1.0]),
                    _embedding_result("bob-2", [0.18, 0.98]),
                ],
            ),
            build_speaker_profile(
                "carol",
                [
                    _embedding_result("carol-1", [-1.0, 0.0]),
                    _embedding_result("carol-2", [-0.98, 0.12]),
                ],
            ),
        ],
        heldout_trials=[
            HeldoutCalibrationTrial(speaker_id="alice", raw_score=0.92, is_target=True),
            HeldoutCalibrationTrial(speaker_id="alice", raw_score=0.88, is_target=True),
            HeldoutCalibrationTrial(speaker_id="alice", raw_score=0.36, is_target=False),
            HeldoutCalibrationTrial(speaker_id="alice", raw_score=0.31, is_target=False),
        ],
    )

    alice = next(profile for profile in profiles if profile.speaker_id == "alice")
    assert alice.metadata["calibration_status"] == "heldout_calibrated"
    assert alice.metadata["calibration_type"] == "linear_heldout"
    assert 0.5 <= alice.metadata["calibration_scale"] <= 1.0
    assert alice.metadata["calibration_bias"] <= 0.0
    assert alice.metadata["heldout_trial_count"] == 4
    assert alice.calibrated_threshold > 0.0
