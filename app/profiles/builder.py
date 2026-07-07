"""Helpers for turning enrollment embeddings into speaker profiles."""

from __future__ import annotations

import numpy as np

from app.common.constants import DEFAULT_PROFILE_NAME
from app.embedding_backends.models import EmbeddingResult
from app.profiles.calibration import HeldoutCalibrationTrial, apply_heldout_calibration, calibrate_speaker_profiles
from app.profiles.models import SpeakerEmbeddingSample, SpeakerProfile
from app.scoring.similarity import cosine_similarity


def build_speaker_profile(
    speaker_id: str,
    embedding_results: list[EmbeddingResult],
    *,
    profile_name: str = DEFAULT_PROFILE_NAME,
    aggregation_strategy: str = "center",
) -> SpeakerProfile:
    """Build one speaker profile from one or more enrollment embeddings."""

    if not embedding_results:
        raise ValueError("At least one embedding result is required to build a profile.")

    samples = [
        SpeakerEmbeddingSample(
            speaker_id=speaker_id,
            embedding_result=result,
            weight_value=_resolve_sample_weight(result),
        )
        for result in embedding_results
    ]
    member_vectors = [
        np.asarray(sample.embedding_result.embedding, dtype=np.float32) for sample in samples
    ]
    if aggregation_strategy == "quality_weighted_center":
        weights = np.asarray([sample.weight_value for sample in samples], dtype=np.float32)
        weights = weights / np.sum(weights)
        vector = np.average(
            np.stack(member_vectors, axis=0),
            axis=0,
            weights=weights,
        ).astype(np.float32)
    else:
        vector = np.mean(
            np.stack(member_vectors, axis=0),
            axis=0,
        ).astype(np.float32)
    vector = _normalize_vector(vector)
    first = embedding_results[0]
    intra_score_mean, intra_score_std = _compute_intra_stats(member_vectors)
    open_set_floor = _resolve_provisional_open_set_floor(
        intra_score_mean=intra_score_mean,
        intra_score_std=intra_score_std,
        sample_count=len(samples),
    )
    return SpeakerProfile(
        speaker_id=speaker_id,
        profile_name=profile_name,
        backend_name=first.backend_name,
        backend_version=first.backend_version,
        feature_version=first.feature_version,
        aggregation_strategy=aggregation_strategy,
        vector=vector,
        center_vector=vector,
        members=samples,
        sub_centers=_build_sub_centers(member_vectors),
        member_vectors=member_vectors,
        intra_score_mean=intra_score_mean,
        intra_score_std=intra_score_std,
        open_set_floor=open_set_floor,
        calibrated_threshold=open_set_floor,
        metadata={
            "sample_count": len(samples),
            "avg_quality_score": float(
                np.mean([sample.weight_value for sample in samples], dtype=np.float32)
            ),
            "default_top_k": min(3, len(samples)),
        },
    )


def finalize_speaker_profiles(
    profiles: list[SpeakerProfile],
    *,
    heldout_trials: list[HeldoutCalibrationTrial] | None = None,
) -> list[SpeakerProfile]:
    """Apply cohort-aware calibration after individual profiles are built."""

    calibrated_profiles = calibrate_speaker_profiles(profiles)
    if heldout_trials:
        calibrated_profiles = apply_heldout_calibration(calibrated_profiles, heldout_trials)
    return calibrated_profiles


def _resolve_sample_weight(result: EmbeddingResult) -> float:
    quality_score = result.quality_score if result.quality_score is not None else 1.0
    duration_bonus = min((result.duration_sec or 0.0) / 6.0, 1.0)
    return float(max(0.1, 0.7 * quality_score + 0.3 * duration_bonus))


def _compute_intra_stats(member_vectors: list[np.ndarray]) -> tuple[float, float]:
    if len(member_vectors) <= 1:
        return 1.0, 0.0
    pairwise_scores = [
        cosine_similarity(member_vectors[left_index], member_vectors[right_index])
        for left_index in range(len(member_vectors))
        for right_index in range(left_index + 1, len(member_vectors))
    ]
    return (
        float(np.mean(pairwise_scores, dtype=np.float32)),
        float(np.std(pairwise_scores, dtype=np.float32)),
    )


def _resolve_provisional_open_set_floor(
    *,
    intra_score_mean: float,
    intra_score_std: float,
    sample_count: int,
) -> float:
    if sample_count <= 1:
        return 0.15
    return float(max(0.15, intra_score_mean - 2.0 * intra_score_std))


def _build_sub_centers(
    member_vectors: list[np.ndarray],
    *,
    max_sub_centers: int = 3,
    min_diversity_gap: float = 0.15,
) -> list[np.ndarray]:
    if not member_vectors:
        return []
    if len(member_vectors) == 1:
        return [member_vectors[0].copy()]

    normalized_members = [_normalize_vector(member_vector) for member_vector in member_vectors]
    sub_centers = [normalized_members[0]]
    for candidate in normalized_members[1:]:
        best_similarity = max(
            cosine_similarity(candidate, existing_sub_center) for existing_sub_center in sub_centers
        )
        if best_similarity <= 1.0 - min_diversity_gap and len(sub_centers) < max_sub_centers:
            sub_centers.append(candidate)
    if not sub_centers:
        return [normalized_members[0]]
    return sub_centers


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm_value = float(np.linalg.norm(vector))
    if norm_value == 0:
        return vector.astype(np.float32)
    return (vector / norm_value).astype(np.float32)
