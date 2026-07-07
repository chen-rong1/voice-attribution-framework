"""Minimal benchmark runner for the first end-to-end framework iteration."""

from __future__ import annotations

from app.benchmark.models import (
    BenchmarkClip,
    BenchmarkItemResult,
    BenchmarkRunConfig,
    BenchmarkRunResult,
)
from app.services.identification import EnrollmentRecord, IdentificationService


class BenchmarkRunner:
    """Runs one benchmark configuration against a prepared backend service."""

    def __init__(self, identification_service: IdentificationService) -> None:
        self.identification_service = identification_service

    def run(
        self,
        *,
        config: BenchmarkRunConfig,
        enrollments: list[EnrollmentRecord],
        clips: list[BenchmarkClip],
    ) -> BenchmarkRunResult:
        """Build profiles once, then score every benchmark clip."""

        profiles = self.identification_service.build_profiles(
            enrollments,
            profile_aggregation_strategy=config.profile_aggregation_strategy,
        )
        items: list[BenchmarkItemResult] = []
        for clip in clips:
            decision = self.identification_service.identify(
                clip.audio_path,
                profiles,
                threshold_value=config.threshold_value,
                scoring_strategy=config.scoring_strategy,
            )
            items.append(
                BenchmarkItemResult(
                    clip_id=clip.clip_id,
                    audio_path=clip.audio_path,
                    truth_label=clip.truth_label,
                    expected_label=clip.expected_label,
                    final_label=decision.final_label,
                    decision=decision.decision.value,
                    best_score=decision.best_score,
                    threshold_value=decision.threshold_value,
                    is_correct=decision.final_label == clip.expected_label,
                    score_breakdown=decision.score_breakdown,
                )
            )
        return BenchmarkRunResult(config=config, items=items)
