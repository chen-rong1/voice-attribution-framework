"""Minimal speaker identification service for the first end-to-end workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.audio.io import load_audio_chunk
from app.embedding_backends.base import EmbeddingBackend
from app.embedding_backends.models import EmbeddingRequest
from app.profiles.builder import build_speaker_profile
from app.profiles.models import SpeakerProfile
from app.scoring.models import DecisionResult, ScoringStrategy
from app.scoring.strategies import build_decision


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
        return profiles

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
        )
