"""Minimal speaker identification service for the first end-to-end workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.audio.io import load_audio_chunk
from app.embedding_backends.base import EmbeddingBackend
from app.embedding_backends.models import EmbeddingRequest
from app.profiles.calibration import HeldoutCalibrationTrial
from app.profiles.builder import build_speaker_profile, finalize_speaker_profiles
from app.profiles.models import SpeakerProfile
from app.scoring.models import DecisionResult, ScoringStrategy
from app.scoring.normalization import compute_cohort_scores, normalize_profile_score
from app.scoring.strategies import (
    build_decision,
    score_profile_center,
    score_profile_max,
    score_profile_quality_weighted_center,
    score_profile_top_k_mean,
)

if TYPE_CHECKING:
    from app.benchmark.models import BenchmarkClip


@dataclass(slots=True)
class EnrollmentRecord:
    """One speaker and the enrollment audio files used to build the profile."""

    speaker_id: str
    audio_paths: list[Path]


class IdentificationService:
    """Orchestrates backend extraction, profile building, and final scoring."""

    def __init__(self, backend: EmbeddingBackend) -> None:
        self.backend = backend

    def build_profiles(
        self,
        enrollments: list[EnrollmentRecord],
        *,
        profile_aggregation_strategy: str = "quality_weighted_center",
        heldout_calibration_trials: list[HeldoutCalibrationTrial] | None = None,
    ) -> list[SpeakerProfile]:
        """Build speaker profiles from enrollment audio files."""

        profiles: list[SpeakerProfile] = []
        for enrollment in enrollments:
            results = []
            for index, audio_path in enumerate(enrollment.audio_paths, start=1):
                audio_chunk = load_audio_chunk(audio_path)
                result = self.backend.extract_embedding(
                    EmbeddingRequest(
                        sample_id=f"{enrollment.speaker_id}-enroll-{index}",
                        audio=audio_chunk,
                    )
                )
                results.append(result)
            profiles.append(
                build_speaker_profile(
                    enrollment.speaker_id,
                    results,
                    aggregation_strategy=profile_aggregation_strategy,
                )
            )
        return finalize_speaker_profiles(
            profiles,
            heldout_trials=heldout_calibration_trials,
        )

    def identify(
        self,
        query_audio_path: Path,
        profiles: list[SpeakerProfile],
        *,
        threshold_value: float,
        scoring_strategy: ScoringStrategy = ScoringStrategy.CENTER,
    ) -> DecisionResult:
        """Identify one query clip against a prepared set of speaker profiles."""

        query_audio = load_audio_chunk(query_audio_path)
        query_result = self.backend.extract_embedding(
            EmbeddingRequest(sample_id="query", audio=query_audio)
        )
        return build_decision(
            query_result.embedding,
            profiles,
            threshold_value=threshold_value,
            scoring_strategy=scoring_strategy,
            query_duration_sec=query_result.duration_sec,
            query_quality_score=query_result.quality_score,
        )

    def build_heldout_calibration_trials(
        self,
        enrollments: list[EnrollmentRecord],
        calibration_clips: list["BenchmarkClip"],
        *,
        profile_aggregation_strategy: str = "quality_weighted_center",
        scoring_strategy: ScoringStrategy = ScoringStrategy.CENTER,
        include_enrollment_self_trials: bool = True,
    ) -> list[HeldoutCalibrationTrial]:
        """Generate heldout calibration trials against statistical profiles."""

        profiles = self.build_profiles(
            enrollments,
            profile_aggregation_strategy=profile_aggregation_strategy,
        )
        scoring_fn = _resolve_scoring_fn(scoring_strategy)
        trials: list[HeldoutCalibrationTrial] = []

        if include_enrollment_self_trials:
            for profile in profiles:
                for member in profile.members:
                    trials.append(
                        HeldoutCalibrationTrial(
                            speaker_id=profile.speaker_id,
                            raw_score=_compute_precalibrated_score(
                                member.embedding_result.embedding,
                                profile,
                                profiles,
                                scoring_fn,
                            ),
                            is_target=True,
                        )
                    )
                for other_profile in profiles:
                    if other_profile.speaker_id == profile.speaker_id:
                        continue
                    for other_member in other_profile.members:
                        trials.append(
                            HeldoutCalibrationTrial(
                                speaker_id=profile.speaker_id,
                                raw_score=_compute_precalibrated_score(
                                    other_member.embedding_result.embedding,
                                    profile,
                                    profiles,
                                    scoring_fn,
                                ),
                                is_target=False,
                            )
                        )

        for clip in calibration_clips:
            query_audio = load_audio_chunk(clip.audio_path)
            query_result = self.backend.extract_embedding(
                EmbeddingRequest(
                    sample_id=f"heldout-{clip.clip_id}",
                    audio=query_audio,
                )
            )
            for profile in profiles:
                trials.append(
                    HeldoutCalibrationTrial(
                        speaker_id=profile.speaker_id,
                        raw_score=_compute_precalibrated_score(
                            query_result.embedding,
                            profile,
                            profiles,
                            scoring_fn,
                        ),
                        is_target=clip.expected_label == profile.speaker_id,
                    )
                )
        return trials


def _resolve_scoring_fn(scoring_strategy: ScoringStrategy):
    return {
        ScoringStrategy.CENTER: score_profile_center,
        ScoringStrategy.MAX: score_profile_max,
        ScoringStrategy.TOP_K_MEAN: score_profile_top_k_mean,
        ScoringStrategy.QUALITY_WEIGHTED_CENTER: score_profile_quality_weighted_center,
    }[scoring_strategy]


def _compute_precalibrated_score(
    query_embedding,
    profile: SpeakerProfile,
    profiles: list[SpeakerProfile],
    scoring_fn,
) -> float:
    raw_score = float(scoring_fn(query_embedding, profile))
    cohort_scores = compute_cohort_scores(
        query_embedding,
        profiles,
        excluded_speaker_id=profile.speaker_id,
        scoring_fn=scoring_fn,
    )
    _, adaptive_s_norm_score, _, _ = normalize_profile_score(
        raw_score=raw_score,
        profile=profile,
        cohort_scores=cohort_scores,
    )
    return float(adaptive_s_norm_score)
