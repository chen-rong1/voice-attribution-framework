"""Minimal benchmark runner for the first end-to-end framework iteration."""

from __future__ import annotations

from time import perf_counter

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
        heldout_calibration_clips: list[BenchmarkClip] | None = None,
    ) -> BenchmarkRunResult:
        """Build profiles once, then score every benchmark clip."""

        heldout_trials = None
        if heldout_calibration_clips:
            heldout_trials = self.identification_service.build_heldout_calibration_trials(
                enrollments,
                heldout_calibration_clips,
                profile_aggregation_strategy=config.profile_aggregation_strategy,
                scoring_strategy=config.scoring_strategy,
            )
        profiles = self.identification_service.build_profiles(
            enrollments,
            profile_aggregation_strategy=config.profile_aggregation_strategy,
            heldout_calibration_trials=heldout_trials,
        )
        items: list[BenchmarkItemResult] = []
        for clip in clips:
            started_at = perf_counter()
            decision = self.identification_service.identify(
                clip.audio_path,
                profiles,
                threshold_value=config.threshold_value,
                scoring_strategy=config.scoring_strategy,
            )
            latency_ms = (perf_counter() - started_at) * 1000.0
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
                    latency_ms=latency_ms,
                    is_correct=decision.final_label == clip.expected_label,
                    evaluation_group=clip.evaluation_group,
                    score_breakdown=decision.score_breakdown,
                    metadata={
                        **clip.metadata,
                        **decision.metadata,
                    },
                )
            )
        return BenchmarkRunResult(config=config, items=items)
