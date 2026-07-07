"""Reference scoring helpers for the first framework iteration."""

from __future__ import annotations

import numpy as np

from app.common.constants import DEFAULT_REJECT_LABEL
from app.profiles.models import SpeakerProfile
from app.scoring.models import DecisionLabel, DecisionResult, ScoringStrategy


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = np.asarray(left, dtype=np.float32).reshape(-1)
    right_vector = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)


def score_profile_center(query_embedding: np.ndarray, profile: SpeakerProfile) -> float:
    return cosine_similarity(query_embedding, profile.vector)


def score_profile_quality_weighted_center(
    query_embedding: np.ndarray,
    profile: SpeakerProfile,
) -> float:
    if not profile.members:
        return score_profile_center(query_embedding, profile)
    weighted_center = _build_weighted_center(profile)
    return cosine_similarity(query_embedding, weighted_center)


def score_profile_max(query_embedding: np.ndarray, profile: SpeakerProfile) -> float:
    if not profile.members:
        return score_profile_center(query_embedding, profile)
    return max(
        cosine_similarity(query_embedding, sample.embedding_result.embedding)
        for sample in profile.members
    )


def score_profile_top_k_mean(
    query_embedding: np.ndarray,
    profile: SpeakerProfile,
) -> float:
    if not profile.members:
        return score_profile_center(query_embedding, profile)
    scores = sorted(
        (
            cosine_similarity(query_embedding, sample.embedding_result.embedding)
            for sample in profile.members
        ),
        reverse=True,
    )
    top_k = int(profile.metadata.get("default_top_k", min(3, len(scores))))
    top_k = max(1, min(top_k, len(scores)))
    return float(np.mean(scores[:top_k], dtype=np.float32))


def build_decision(
    query_embedding: np.ndarray,
    profiles: list[SpeakerProfile],
    *,
    threshold_value: float,
    scoring_strategy: ScoringStrategy,
) -> DecisionResult:
    if not profiles:
        raise ValueError("At least one speaker profile is required for scoring.")

    scoring_fn = {
        ScoringStrategy.CENTER: score_profile_center,
        ScoringStrategy.MAX: score_profile_max,
        ScoringStrategy.TOP_K_MEAN: score_profile_top_k_mean,
        ScoringStrategy.QUALITY_WEIGHTED_CENTER: score_profile_quality_weighted_center,
    }[scoring_strategy]
    score_breakdown = {
        profile.speaker_id: scoring_fn(query_embedding, profile) for profile in profiles
    }
    best_speaker_id, best_score = max(score_breakdown.items(), key=lambda item: item[1])
    if best_score >= threshold_value:
        return DecisionResult(
            decision=DecisionLabel.ACCEPT,
            final_label=best_speaker_id,
            best_speaker_id=best_speaker_id,
            best_score=best_score,
            threshold_value=threshold_value,
            scoring_strategy=scoring_strategy,
            score_breakdown=score_breakdown,
        )
    return DecisionResult(
        decision=DecisionLabel.REJECT,
        final_label=DEFAULT_REJECT_LABEL,
        best_speaker_id=None,
        best_score=best_score,
        threshold_value=threshold_value,
        scoring_strategy=scoring_strategy,
        score_breakdown=score_breakdown,
    )


def _build_weighted_center(profile: SpeakerProfile) -> np.ndarray:
    weights = np.asarray([sample.weight_value for sample in profile.members], dtype=np.float32)
    weights = weights / np.sum(weights)
    vectors = np.stack(
        [sample.embedding_result.embedding for sample in profile.members],
        axis=0,
    )
    return np.average(vectors, axis=0, weights=weights).astype(np.float32)
