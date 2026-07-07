"""Score normalization helpers for open-set identification."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from app.profiles.models import SpeakerProfile

EPSILON = 1e-6
COHORT_STD_FLOOR = 0.05
MAX_NORMALIZED_ABS_SCORE = 8.0


def normalize_profile_score(
    *,
    raw_score: float,
    profile: SpeakerProfile,
    cohort_scores: list[float],
    cohort_top_k: int = 5,
) -> tuple[float, float, float, float]:
    """Return z-norm, adaptive s-norm, calibrated, and cohort-relative scores."""

    impostor_mu = float(profile.impostor_score_mean)
    impostor_sigma = max(float(profile.impostor_score_std), COHORT_STD_FLOOR)
    z_norm_score = _clip_normalized_score((raw_score - impostor_mu) / impostor_sigma)

    if not cohort_scores:
        adaptive_s_norm_score = z_norm_score
        calibrated_score = _apply_profile_calibration(adaptive_s_norm_score, profile)
        return (
            z_norm_score,
            adaptive_s_norm_score,
            calibrated_score,
            calibrated_score,
        )

    sorted_scores = sorted(cohort_scores, reverse=True)
    top_scores = sorted_scores[: max(1, min(cohort_top_k, len(sorted_scores)))]
    if len(top_scores) < 2:
        adaptive_s_norm_score = z_norm_score
        calibrated_score = _apply_profile_calibration(adaptive_s_norm_score, profile)
        return (
            z_norm_score,
            adaptive_s_norm_score,
            calibrated_score,
            calibrated_score,
        )
    cohort_mean = float(np.mean(top_scores, dtype=np.float32))
    cohort_std = float(np.std(top_scores, dtype=np.float32))
    effective_cohort_std = max(cohort_std, impostor_sigma, COHORT_STD_FLOOR)
    t_norm_score = _clip_normalized_score((raw_score - cohort_mean) / effective_cohort_std)
    adaptive_s_norm_score = _clip_normalized_score((z_norm_score + t_norm_score) / 2.0)
    calibrated_score = _apply_profile_calibration(adaptive_s_norm_score, profile)
    cohort_relative_score = _apply_profile_calibration(t_norm_score, profile)
    if not math.isfinite(cohort_relative_score):
        cohort_relative_score = calibrated_score
    return (
        z_norm_score,
        adaptive_s_norm_score,
        calibrated_score,
        cohort_relative_score,
    )


def compute_cohort_scores(
    query_embedding: np.ndarray,
    profiles: list[SpeakerProfile],
    *,
    excluded_speaker_id: str,
    scoring_fn: Callable[[np.ndarray, SpeakerProfile], float],
) -> list[float]:
    """Score the query against other profiles to build an adaptive cohort."""

    return [
        scoring_fn(query_embedding, profile)
        for profile in profiles
        if profile.speaker_id != excluded_speaker_id
    ]


def _apply_profile_calibration(score: float, profile: SpeakerProfile) -> float:
    scale = float(profile.metadata.get("calibration_scale", 1.0))
    bias = float(profile.metadata.get("calibration_bias", 0.0))
    calibrated = float(score * scale + bias)
    if not math.isfinite(calibrated):
        return float(score)
    return calibrated


def _clip_normalized_score(score: float) -> float:
    clipped = float(np.clip(score, -MAX_NORMALIZED_ABS_SCORE, MAX_NORMALIZED_ABS_SCORE))
    if not math.isfinite(clipped):
        return 0.0
    return clipped
