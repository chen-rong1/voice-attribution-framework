"""Profile-level calibration helpers for open-set identification."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.profiles.cohort import collect_impostor_vectors
from app.profiles.models import ProfileRiskLevel, SpeakerProfile
from app.scoring.similarity import cosine_similarity

DEFAULT_IMPOSTOR_STD = 0.05
MIN_OPEN_SET_FLOOR = 0.15
MIN_HELDOUT_TRIALS = 4
MIN_HELDOUT_POSITIVES = 2
MIN_HELDOUT_NEGATIVES = 2
MIN_HELDOUT_SCALE = 0.5
MAX_HELDOUT_SCALE = 1.0


@dataclass(frozen=True, slots=True)
class HeldoutCalibrationTrial:
    """One heldout trial used to learn profile-wise score calibration."""

    speaker_id: str
    raw_score: float
    is_target: bool


@dataclass(frozen=True, slots=True)
class HeldoutCalibrationResult:
    """Learned calibration parameters for one profile."""

    speaker_id: str
    calibration_scale: float
    calibration_bias: float
    calibrated_threshold: float
    positive_mean: float
    negative_mean: float
    trial_count: int


def apply_heldout_calibration(
    profiles: list[SpeakerProfile],
    trials: list[HeldoutCalibrationTrial],
) -> list[SpeakerProfile]:
    """Learn and attach profile-wise linear calibration from heldout trials."""

    if not profiles or not trials:
        return profiles

    calibration_by_speaker = _fit_heldout_calibration_by_speaker(trials)
    if not calibration_by_speaker:
        return profiles

    for profile in profiles:
        calibration = calibration_by_speaker.get(profile.speaker_id)
        if calibration is None:
            continue
        profile.calibrated_threshold = calibration.calibrated_threshold
        profile.metadata["calibration_status"] = "heldout_calibrated"
        profile.metadata["calibration_type"] = "linear_heldout"
        profile.metadata["calibration_scale"] = calibration.calibration_scale
        profile.metadata["calibration_bias"] = calibration.calibration_bias
        profile.metadata["calibrated_threshold"] = calibration.calibrated_threshold
        profile.metadata["heldout_positive_mean"] = calibration.positive_mean
        profile.metadata["heldout_negative_mean"] = calibration.negative_mean
        profile.metadata["heldout_trial_count"] = calibration.trial_count
    return profiles


def _fit_heldout_calibration_by_speaker(
    trials: list[HeldoutCalibrationTrial],
) -> dict[str, HeldoutCalibrationResult]:
    grouped: dict[str, list[HeldoutCalibrationTrial]] = {}
    for trial in trials:
        grouped.setdefault(trial.speaker_id, []).append(trial)

    results: dict[str, HeldoutCalibrationResult] = {}
    for speaker_id, speaker_trials in grouped.items():
        calibration = _fit_heldout_calibration(speaker_id, speaker_trials)
        if calibration is not None:
            results[speaker_id] = calibration
    return results


def _fit_heldout_calibration(
    speaker_id: str,
    trials: list[HeldoutCalibrationTrial],
) -> HeldoutCalibrationResult | None:
    if len(trials) < MIN_HELDOUT_TRIALS:
        return None
    positive_scores = np.asarray(
        [trial.raw_score for trial in trials if trial.is_target],
        dtype=np.float32,
    )
    negative_scores = np.asarray(
        [trial.raw_score for trial in trials if not trial.is_target],
        dtype=np.float32,
    )
    if len(positive_scores) < MIN_HELDOUT_POSITIVES or len(negative_scores) < MIN_HELDOUT_NEGATIVES:
        return None

    all_scores = np.concatenate([positive_scores, negative_scores])
    score_std = max(float(np.std(all_scores, dtype=np.float32)), DEFAULT_IMPOSTOR_STD)
    scale = float(np.clip(1.0 / max(score_std, 1.0), MIN_HELDOUT_SCALE, MAX_HELDOUT_SCALE))
    negative_mean_raw = float(np.mean(negative_scores, dtype=np.float32))
    bias = float(-negative_mean_raw * scale)
    calibrated_positive_scores = positive_scores * scale + bias
    calibrated_negative_scores = negative_scores * scale + bias
    calibrated_threshold = _optimize_heldout_threshold(
        calibrated_positive_scores,
        calibrated_negative_scores,
    )
    positive_mean = float(np.mean(calibrated_positive_scores, dtype=np.float32))
    negative_mean = float(np.mean(calibrated_negative_scores, dtype=np.float32))

    return HeldoutCalibrationResult(
        speaker_id=speaker_id,
        calibration_scale=scale,
        calibration_bias=bias,
        calibrated_threshold=calibrated_threshold,
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        trial_count=len(trials),
    )


def _optimize_heldout_threshold(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> float:
    if positive_scores.size == 0 or negative_scores.size == 0:
        return 0.0
    unique_scores = sorted(
        {
            float(score)
            for score in np.concatenate([positive_scores, negative_scores]).tolist()
        }
    )
    if not unique_scores:
        return 0.0
    candidates = [unique_scores[0] - DEFAULT_IMPOSTOR_STD]
    candidates.extend(
        (left + right) / 2.0 for left, right in zip(unique_scores, unique_scores[1:])
    )
    candidates.append(unique_scores[-1] + DEFAULT_IMPOSTOR_STD)

    best_threshold = float(candidates[0])
    best_objective = (
        int(np.sum(negative_scores >= best_threshold)),
        int(np.sum(positive_scores < best_threshold)),
        -best_threshold,
    )
    for threshold in candidates[1:]:
        objective = (
            int(np.sum(negative_scores >= threshold)),
            int(np.sum(positive_scores < threshold)),
            -float(threshold),
        )
        if objective < best_objective:
            best_objective = objective
            best_threshold = float(threshold)
    return best_threshold


def calibrate_speaker_profiles(profiles: list[SpeakerProfile]) -> list[SpeakerProfile]:
    """Populate profile-wise impostor statistics and open-set thresholds."""

    if not profiles:
        return profiles

    for profile in profiles:
        impostor_vectors = collect_impostor_vectors(
            profiles,
            excluded_speaker_id=profile.speaker_id,
        )
        if impostor_vectors:
            impostor_scores = np.asarray(
                [
                    cosine_similarity(impostor_vector, profile.center_vector)
                    for impostor_vector in impostor_vectors
                ],
                dtype=np.float32,
            )
            impostor_mean = float(np.mean(impostor_scores, dtype=np.float32))
            impostor_std = float(np.std(impostor_scores, dtype=np.float32))
        else:
            impostor_mean = 0.0
            impostor_std = DEFAULT_IMPOSTOR_STD

        profile.impostor_score_mean = impostor_mean
        profile.impostor_score_std = max(impostor_std, DEFAULT_IMPOSTOR_STD)
        profile.open_set_floor = _resolve_open_set_floor(profile)
        profile.calibrated_threshold = profile.open_set_floor
        profile.risk_level = _resolve_risk_level(profile)
        profile.metadata["calibration_status"] = "statistical"
        profile.metadata["calibration_type"] = "profile_impostor_stats"
        profile.metadata["calibration_scale"] = 1.0
        profile.metadata["calibration_bias"] = 0.0
        profile.metadata["impostor_score_mean"] = profile.impostor_score_mean
        profile.metadata["impostor_score_std"] = profile.impostor_score_std
        profile.metadata["open_set_floor"] = profile.open_set_floor
        profile.metadata["calibrated_threshold"] = profile.calibrated_threshold
        profile.metadata["risk_level"] = profile.risk_level.value
    return profiles


def _resolve_open_set_floor(profile: SpeakerProfile) -> float:
    sample_count = int(profile.metadata.get("sample_count", len(profile.member_vectors) or 1))
    impostor_floor = profile.impostor_score_mean + 2.5 * profile.impostor_score_std
    intra_floor = profile.intra_score_mean - 2.0 * profile.intra_score_std
    if sample_count <= 1:
        return float(max(MIN_OPEN_SET_FLOOR, impostor_floor))
    if not math.isfinite(intra_floor):
        return float(max(MIN_OPEN_SET_FLOOR, impostor_floor))
    return float(max(MIN_OPEN_SET_FLOOR, min(intra_floor, max(impostor_floor, 0.0))))


def _resolve_risk_level(profile: SpeakerProfile) -> ProfileRiskLevel:
    sample_count = int(profile.metadata.get("sample_count", len(profile.member_vectors) or 1))
    if sample_count <= 1:
        return ProfileRiskLevel.HIGH
    if profile.impostor_score_mean >= 0.35 or profile.intra_score_std >= 0.12:
        return ProfileRiskLevel.HIGH
    if profile.impostor_score_mean >= 0.2 or profile.intra_score_std >= 0.06:
        return ProfileRiskLevel.MEDIUM
    return ProfileRiskLevel.LOW
