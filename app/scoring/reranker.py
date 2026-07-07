"""Candidate reranking utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.profiles.models import SpeakerProfile
from app.scoring.models import CandidateScore
from app.scoring.normalization import compute_cohort_scores, normalize_profile_score
from app.scoring.similarity import cosine_similarity


def build_candidate_scores(
    query_embedding: np.ndarray,
    profiles: list[SpeakerProfile],
    *,
    scoring_fn: Callable[[np.ndarray, SpeakerProfile], float],
    top_k: int = 3,
) -> list[CandidateScore]:
    """Build candidate features, then rerank the top-k list."""

    raw_score_breakdown = {
        profile.speaker_id: scoring_fn(query_embedding, profile) for profile in profiles
    }
    raw_ranked_profiles = sorted(
        profiles,
        key=lambda profile: raw_score_breakdown[profile.speaker_id],
        reverse=True,
    )[: max(1, min(top_k, len(profiles)))]

    candidates: list[CandidateScore] = []
    for profile in raw_ranked_profiles:
        raw_score = float(raw_score_breakdown[profile.speaker_id])
        cohort_scores = compute_cohort_scores(
            query_embedding,
            profiles,
            excluded_speaker_id=profile.speaker_id,
            scoring_fn=scoring_fn,
        )
        (
            z_norm_score,
            adaptive_s_norm_score,
            calibrated_score,
            cohort_relative_score,
        ) = normalize_profile_score(
            raw_score=raw_score,
            profile=profile,
            cohort_scores=cohort_scores,
        )
        member_consistency_score = _compute_member_consistency(query_embedding, profile)
        sub_center_score = _compute_sub_center_score(query_embedding, profile)
        bounded_calibrated_score = _bound_rerank_signal(calibrated_score)
        bounded_cohort_relative_score = _bound_rerank_signal(cohort_relative_score)
        reranked_score = float(
            0.35 * raw_score
            + 0.2 * member_consistency_score
            + 0.15 * sub_center_score
            + 0.2 * bounded_calibrated_score
            + 0.1 * bounded_cohort_relative_score
        )
        candidates.append(
            CandidateScore(
                speaker_id=profile.speaker_id,
                profile=profile,
                raw_score=raw_score,
                z_norm_score=z_norm_score,
                adaptive_s_norm_score=adaptive_s_norm_score,
                calibrated_score=calibrated_score,
                cohort_relative_score=cohort_relative_score,
                member_consistency_score=member_consistency_score,
                sub_center_score=sub_center_score,
                reranked_score=reranked_score,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.reranked_score, reverse=True)


def _compute_member_consistency(query_embedding: np.ndarray, profile: SpeakerProfile) -> float:
    member_vectors = profile.member_vectors or [profile.center_vector]
    member_scores = sorted(
        (cosine_similarity(query_embedding, member_vector) for member_vector in member_vectors),
        reverse=True,
    )
    default_top_k = int(profile.metadata.get("default_top_k", min(3, len(member_scores))))
    top_k = max(1, min(default_top_k, len(member_scores)))
    return float(np.mean(member_scores[:top_k], dtype=np.float32))


def _compute_sub_center_score(query_embedding: np.ndarray, profile: SpeakerProfile) -> float:
    sub_centers = profile.sub_centers or [profile.center_vector]
    return float(
        max(cosine_similarity(query_embedding, sub_center) for sub_center in sub_centers)
    )


def _bound_rerank_signal(score: float) -> float:
    return float(np.clip(score, -3.0, 3.0))
