"""Reference scoring helpers for the first framework iteration."""

from __future__ import annotations

import os

import numpy as np

from app.profiles.models import ProfileRiskLevel, SpeakerProfile
from app.scoring.models import AcceptReason, DecisionResult, RejectReason, ScoringStrategy
from app.scoring.pipeline import run_scoring_pipeline
from app.scoring.reranker import build_candidate_scores
from app.scoring.similarity import cosine_similarity as _cosine_similarity

SINGLE_SAMPLE_SINGLE_SPEAKER_THRESHOLD = 0.311
MULTI_SPEAKER_MIN_MARGIN = 0.02
MULTI_SPEAKER_MARGIN_GUARD_BAND = 0.08
DEFAULT_MULTI_SPEAKER_DURATION_THRESHOLDS = (
    (0.8, None),
    (1.5, 0.33),
    (float("inf"), 0.31),
)
DEFAULT_MULTI_SPEAKER_MARGIN_THRESHOLDS = (
    (1.5, MULTI_SPEAKER_MIN_MARGIN),
    (float("inf"), 0.015),
)
MULTI_SPEAKER_CLUSTER_GUARD_MIN_DURATION = 2.0
MULTI_SPEAKER_CLUSTER_GUARD_MIN_QUALITY = 0.75
MULTI_SPEAKER_CLUSTER_GUARD_MIN_TOP2 = 0.43
MULTI_SPEAKER_CLUSTER_GUARD_MIN_TOP3 = 0.43
MULTI_SPEAKER_CLUSTER_GUARD_MAX_TOP1_TOP3_GAP = 0.13
MULTI_SPEAKER_RUNOFF_MIN_DURATION = 0.7
MULTI_SPEAKER_RUNOFF_MIN_QUALITY = 0.35
MULTI_SPEAKER_RUNOFF_MIN_TOP1 = 0.29
MULTI_SPEAKER_RUNOFF_MIN_TOP2 = 0.29
MULTI_SPEAKER_RUNOFF_MAX_TOP1_TOP2_GAP = 0.005
MULTI_SPEAKER_RUNOFF_MIN_TOP2_TOP3_GAP = 0.1
MULTI_SPEAKER_STRONG_LEADER_MIN_DURATION = 1.0
MULTI_SPEAKER_STRONG_LEADER_MAX_DURATION = 1.3
MULTI_SPEAKER_STRONG_LEADER_MIN_QUALITY = 0.34
MULTI_SPEAKER_STRONG_LEADER_MIN_TOP1 = 0.29
MULTI_SPEAKER_STRONG_LEADER_MAX_SCORE_DEFICIT = 0.04
MULTI_SPEAKER_STRONG_LEADER_MIN_TOP1_TOP2_GAP = 0.1
SHORT_CALIBRATED_LEADER_MIN_DURATION = 0.7
SHORT_CALIBRATED_LEADER_MAX_DURATION = 1.0
SHORT_CALIBRATED_LEADER_MIN_QUALITY = 0.28
SHORT_CALIBRATED_LEADER_MIN_TOP1 = 0.24
SHORT_CALIBRATED_LEADER_MAX_SCORE_DEFICIT = 0.11
SHORT_CALIBRATED_LEADER_MIN_TOP1_TOP2_GAP = 0.1
GATE_REVIEW_OVERRIDE_MIN_DURATION = 2.3
GATE_REVIEW_OVERRIDE_MIN_QUALITY = 0.7
GATE_REVIEW_OVERRIDE_MIN_RAW_SURPLUS = 0.025
GATE_REVIEW_OVERRIDE_MIN_CALIBRATED_SCORE = 1.8
GATE_REVIEW_OVERRIDE_MAX_RERANKED_MARGIN = 0.015
ADAPTIVE_THRESHOLD_SHRINK = 0.5
CALIBRATED_OVERRIDE_MAX_RAW_DEFICIT = 0.06
HIGH_RISK_LOW_FLOOR_MAX = 0.33
HIGH_RISK_LOW_FLOOR_LONG_DURATION = 3.0
HIGH_RISK_LOW_FLOOR_MAX_RAW = 0.355
HIGH_RISK_HIGH_FLOOR_MIN = 0.38
HIGH_RISK_HIGH_FLOOR_LONG_DURATION = 3.0
HIGH_RISK_HIGH_FLOOR_MIN_QUALITY = 0.75
HIGH_RISK_HIGH_FLOOR_MAX_RAW = 0.37
HIGH_RISK_HIGH_FLOOR_MAX_MARGIN = 0.1
HIGH_RISK_HIGH_FLOOR_MIN_RAW_GAP = 0.1
HIGH_RISK_HIGH_FLOOR_MID_DURATION = 2.3
HIGH_RISK_HIGH_FLOOR_MID_QUALITY = 0.5
HIGH_RISK_HIGH_FLOOR_MID_MAX_RAW = 0.29
HIGH_RISK_HIGH_FLOOR_MID_MAX_MARGIN = 0.06
HIGH_RISK_SHORT_LOW_QUALITY_MAX_DURATION = 1.0
HIGH_RISK_SHORT_LOW_QUALITY_MAX_QUALITY = 0.35
HIGH_RISK_SHORT_LOW_QUALITY_MAX_RAW = 0.31


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return _cosine_similarity(left, right)


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
    query_duration_sec: float | None = None,
    query_quality_score: float | None = None,
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
    candidates = build_candidate_scores(
        query_embedding,
        profiles,
        scoring_fn=scoring_fn,
    )
    best_candidate = candidates[0]
    second_candidate = candidates[1] if len(candidates) > 1 else None
    third_candidate = candidates[2] if len(candidates) > 2 else None
    best_speaker_id, best_score = best_candidate.speaker_id, best_candidate.raw_score
    top2_speaker_id, top2_score = (
        (second_candidate.speaker_id, second_candidate.raw_score)
        if second_candidate is not None
        else (None, None)
    )
    top3_speaker_id, top3_score = (
        (third_candidate.speaker_id, third_candidate.raw_score)
        if third_candidate is not None
        else (None, None)
    )
    margin_value = (
        float(best_score - top2_score)
        if top2_score is not None
        else float(best_score)
    )
    calibrated_margin_value = (
        float(best_candidate.calibrated_score - second_candidate.calibrated_score)
        if second_candidate is not None
        else float(best_candidate.calibrated_score)
    )
    reranked_margin_value = (
        float(best_candidate.reranked_score - second_candidate.reranked_score)
        if second_candidate is not None
        else float(best_candidate.reranked_score)
    )
    effective_threshold = _resolve_effective_threshold(
        profiles,
        threshold_value,
        query_duration_sec=query_duration_sec,
    )
    effective_calibrated_threshold = _resolve_effective_calibrated_threshold(
        best_candidate.profile,
        effective_threshold,
    )
    metadata: dict[str, str | float | int] = {
        "requested_threshold_value": float(threshold_value),
        "effective_threshold_value": float(effective_threshold),
        "effective_calibrated_threshold_value": (
            float(effective_calibrated_threshold)
            if effective_calibrated_threshold is not None
            else ""
        ),
        "profile_count": len(profiles),
        "top1_speaker_id": best_speaker_id,
        "top1_score": float(best_score),
        "margin": margin_value,
        "calibrated_margin": calibrated_margin_value,
        "effective_margin_threshold": _resolve_margin_threshold(
            profile_count=len(profiles),
            query_duration_sec=query_duration_sec,
        ),
    }
    if top2_speaker_id is not None and top2_score is not None:
        metadata["top2_speaker_id"] = top2_speaker_id
        metadata["top2_score"] = float(top2_score)
    if top3_speaker_id is not None and top3_score is not None:
        metadata["top3_speaker_id"] = top3_speaker_id
        metadata["top3_score"] = float(top3_score)
    if query_duration_sec is not None:
        metadata["query_duration_sec"] = float(query_duration_sec)
    if query_quality_score is not None:
        metadata["query_quality_score"] = float(query_quality_score)

    accepts_in_raw_space = best_score >= effective_threshold
    accepts_in_calibrated_space = (
        effective_calibrated_threshold is not None
        and best_candidate.calibrated_score >= effective_calibrated_threshold
        and (effective_threshold - best_score) <= CALIBRATED_OVERRIDE_MAX_RAW_DEFICIT
    )
    raw_reject_for_margin = _should_reject_for_margin(
        best_score=best_score,
        top2_score=top2_score,
        effective_threshold=effective_threshold,
        profile_count=len(profiles),
        query_duration_sec=query_duration_sec,
    )
    raw_reject_for_cluster = _should_reject_for_cluster(
        best_score=best_score,
        top2_score=top2_score,
        top3_score=top3_score,
        profile_count=len(profiles),
        query_duration_sec=query_duration_sec,
        query_quality_score=query_quality_score,
    )
    raw_reject_for_profile_guard = _should_reject_for_high_risk_profile_guard(
        profile=best_candidate.profile,
        best_score=best_score,
        margin=margin_value,
        query_duration_sec=query_duration_sec,
        query_quality_score=query_quality_score,
    )
    accepts_normally = (
        accepts_in_raw_space
        and not raw_reject_for_margin
        and not raw_reject_for_cluster
        and not raw_reject_for_profile_guard
    ) or (
        not accepts_in_raw_space
        and accepts_in_calibrated_space
        and not raw_reject_for_profile_guard
    )
    accepts_for_runoff = _should_accept_for_two_candidate_runoff(
        best_score=best_score,
        top2_score=top2_score,
        top3_score=top3_score,
        effective_threshold=effective_threshold,
        profile_count=len(profiles),
        query_duration_sec=query_duration_sec,
        query_quality_score=query_quality_score,
    )
    accepts_for_strong_leader = _should_accept_for_strong_leader_below_threshold(
        best_score=best_score,
        top2_score=top2_score,
        effective_threshold=effective_threshold,
        profile_count=len(profiles),
        query_duration_sec=query_duration_sec,
        query_quality_score=query_quality_score,
    )
    accepts_for_short_calibrated_leader = _should_accept_for_short_calibrated_leader(
        best_score=best_score,
        top2_score=top2_score,
        calibrated_score=best_candidate.calibrated_score,
        effective_threshold=effective_threshold,
        effective_calibrated_threshold=effective_calibrated_threshold,
        profile_count=len(profiles),
        query_duration_sec=query_duration_sec,
        query_quality_score=query_quality_score,
    )
    accepts_for_gate_review_override = _should_accept_for_gate_review_override(
        best_score=best_score,
        top2_score=top2_score,
        calibrated_score=best_candidate.calibrated_score,
        effective_threshold=effective_threshold,
        effective_calibrated_threshold=effective_calibrated_threshold,
        query_duration_sec=query_duration_sec,
        query_quality_score=query_quality_score,
        reranked_margin=reranked_margin_value,
    )
    if (
        accepts_normally
        or accepts_for_runoff
        or accepts_for_strong_leader
        or accepts_for_short_calibrated_leader
        or accepts_for_gate_review_override
    ):
        if accepts_for_runoff:
            metadata["accept_reason"] = AcceptReason.TWO_CANDIDATE_RUNOFF.value
            metadata["accept_score_space"] = "raw"
        elif accepts_for_strong_leader:
            metadata["accept_reason"] = AcceptReason.STRONG_LEADER_BELOW_THRESHOLD.value
            metadata["accept_score_space"] = "raw"
        elif accepts_for_short_calibrated_leader:
            metadata["accept_reason"] = AcceptReason.SHORT_CALIBRATED_LEADER.value
            metadata["accept_score_space"] = "calibrated"
        elif accepts_for_gate_review_override:
            metadata["accept_reason"] = AcceptReason.GATE_REVIEW_OVERRIDE.value
            metadata["accept_score_space"] = "calibrated"
        else:
            metadata["accept_reason"] = AcceptReason.NORMAL_ACCEPT.value
            metadata["accept_score_space"] = (
                "raw" if accepts_in_raw_space else "calibrated"
            )
        metadata["decision_reason"] = str(metadata["accept_reason"])
        metadata["reject_reason"] = ""
    else:
        metadata["reject_reason"] = _resolve_reject_reason(
            best_score=best_score,
            calibrated_score=best_candidate.calibrated_score,
            top2_score=top2_score,
            effective_threshold=effective_threshold,
            effective_calibrated_threshold=effective_calibrated_threshold,
            profile_count=len(profiles),
            query_duration_sec=query_duration_sec,
            top3_score=top3_score,
            query_quality_score=query_quality_score,
            rejected_for_profile_guard=raw_reject_for_profile_guard,
        )
        metadata["decision_reason"] = str(metadata["reject_reason"])
        metadata["accept_reason"] = ""
    return run_scoring_pipeline(
        query_embedding,
        profiles,
        effective_threshold=effective_threshold,
        scoring_strategy=scoring_strategy,
        scoring_fn=scoring_fn,
        query_duration_sec=query_duration_sec,
        query_quality_score=query_quality_score,
        raw_score_breakdown=score_breakdown,
        accepts_normally=accepts_normally,
        accepts_for_runoff=accepts_for_runoff,
        accepts_for_strong_leader=accepts_for_strong_leader,
        accepts_for_short_calibrated_leader=accepts_for_short_calibrated_leader,
        accepts_for_gate_review_override=accepts_for_gate_review_override,
        metadata=metadata,
        candidates=candidates,
    )


def _build_weighted_center(profile: SpeakerProfile) -> np.ndarray:
    weights = np.asarray([sample.weight_value for sample in profile.members], dtype=np.float32)
    weights = weights / np.sum(weights)
    vectors = np.stack(
        [sample.embedding_result.embedding for sample in profile.members],
        axis=0,
    )
    return np.average(vectors, axis=0, weights=weights).astype(np.float32)


def _resolve_effective_threshold(
    profiles: list[SpeakerProfile],
    requested_threshold: float,
    *,
    query_duration_sec: float | None = None,
) -> float:
    if len(profiles) == 1:
        sample_count = int(profiles[0].metadata.get("sample_count", len(profiles[0].members)))
        if sample_count == 1:
            if requested_threshold > 1.0:
                return float(requested_threshold)
            return float(min(requested_threshold, SINGLE_SAMPLE_SINGLE_SPEAKER_THRESHOLD))
        return float(requested_threshold)
    if query_duration_sec is None:
        return float(requested_threshold)
    duration_threshold = _resolve_multi_speaker_duration_threshold(
        requested_threshold=requested_threshold,
        query_duration_sec=query_duration_sec,
    )
    return float(min(requested_threshold, duration_threshold))


def _should_reject_for_margin(
    *,
    best_score: float,
    top2_score: float | None,
    effective_threshold: float,
    profile_count: int,
    query_duration_sec: float | None,
) -> bool:
    if top2_score is None:
        return False
    if best_score >= effective_threshold + MULTI_SPEAKER_MARGIN_GUARD_BAND:
        return False
    return (best_score - top2_score) < _resolve_margin_threshold(
        profile_count=profile_count,
        query_duration_sec=query_duration_sec,
    )


def _resolve_reject_reason(
    *,
    best_score: float,
    calibrated_score: float,
    top2_score: float | None,
    top3_score: float | None,
    effective_threshold: float,
    effective_calibrated_threshold: float | None,
    profile_count: int,
    query_duration_sec: float | None,
    query_quality_score: float | None,
    rejected_for_profile_guard: bool,
) -> str:
    if best_score < effective_threshold and (
        effective_calibrated_threshold is None
        or calibrated_score < effective_calibrated_threshold
    ):
        return RejectReason.BELOW_THRESHOLD.value
    if rejected_for_profile_guard:
        return RejectReason.HIGH_RISK_PROFILE_GUARD.value
    if (
        effective_calibrated_threshold is not None
        and calibrated_score >= effective_calibrated_threshold
        and (effective_threshold - best_score) > CALIBRATED_OVERRIDE_MAX_RAW_DEFICIT
    ):
        return RejectReason.CALIBRATED_OVERRIDE_RAW_DEFICIT.value
    if _should_reject_for_margin(
        best_score=best_score,
        top2_score=top2_score,
        effective_threshold=effective_threshold,
        profile_count=profile_count,
        query_duration_sec=query_duration_sec,
    ):
        return RejectReason.LOW_MARGIN.value
    if _should_reject_for_cluster(
        best_score=best_score,
        top2_score=top2_score,
        top3_score=top3_score,
        profile_count=profile_count,
        query_duration_sec=query_duration_sec,
        query_quality_score=query_quality_score,
    ):
        return RejectReason.CROWDED_HIGH_SCORE_CLUSTER.value
    return RejectReason.REJECTED.value


def _should_reject_for_high_risk_profile_guard(
    *,
    profile: SpeakerProfile,
    best_score: float,
    margin: float,
    query_duration_sec: float | None,
    query_quality_score: float | None,
) -> bool:
    if profile.risk_level != ProfileRiskLevel.HIGH:
        return False
    duration = query_duration_sec or 0.0
    quality = query_quality_score or 0.0
    floor = float(profile.open_set_floor)
    if (
        floor <= HIGH_RISK_LOW_FLOOR_MAX
        and duration < HIGH_RISK_SHORT_LOW_QUALITY_MAX_DURATION
        and quality < HIGH_RISK_SHORT_LOW_QUALITY_MAX_QUALITY
        and best_score < HIGH_RISK_SHORT_LOW_QUALITY_MAX_RAW
        and best_score < floor
    ):
        return True
    if (
        floor <= HIGH_RISK_LOW_FLOOR_MAX
        and duration >= HIGH_RISK_LOW_FLOOR_LONG_DURATION
        and best_score < HIGH_RISK_LOW_FLOOR_MAX_RAW
    ):
        return True
    if (
        floor >= HIGH_RISK_HIGH_FLOOR_MIN
        and duration >= HIGH_RISK_HIGH_FLOOR_MID_DURATION
        and quality >= HIGH_RISK_HIGH_FLOOR_MID_QUALITY
        and best_score < HIGH_RISK_HIGH_FLOOR_MID_MAX_RAW
        and margin < HIGH_RISK_HIGH_FLOOR_MID_MAX_MARGIN
    ):
        return True
    return (
        floor >= HIGH_RISK_HIGH_FLOOR_MIN
        and duration >= HIGH_RISK_HIGH_FLOOR_LONG_DURATION
        and quality >= HIGH_RISK_HIGH_FLOOR_MIN_QUALITY
        and (
            (best_score < HIGH_RISK_HIGH_FLOOR_MAX_RAW and margin < HIGH_RISK_HIGH_FLOOR_MAX_MARGIN)
            or (floor - best_score) > HIGH_RISK_HIGH_FLOOR_MIN_RAW_GAP
        )
    )


def _resolve_effective_calibrated_threshold(
    profile: SpeakerProfile,
    effective_threshold: float,
) -> float | None:
    calibration_status = str(profile.metadata.get("calibration_status", ""))
    if calibration_status == "heldout_calibrated":
        return float(profile.calibrated_threshold)
    if "impostor_score_mean" not in profile.metadata:
        return None
    impostor_mu = float(profile.impostor_score_mean)
    impostor_sigma = max(float(profile.impostor_score_std), 0.05)
    adaptive_threshold = ((effective_threshold - impostor_mu) / impostor_sigma) * (
        ADAPTIVE_THRESHOLD_SHRINK
    )
    scale = float(profile.metadata.get("calibration_scale", 1.0))
    bias = float(profile.metadata.get("calibration_bias", 0.0))
    return float(adaptive_threshold * scale + bias)


def _should_accept_for_two_candidate_runoff(
    *,
    best_score: float,
    top2_score: float | None,
    top3_score: float | None,
    effective_threshold: float,
    profile_count: int,
    query_duration_sec: float | None,
    query_quality_score: float | None,
) -> bool:
    if profile_count < 3:
        return False
    if top2_score is None or top3_score is None:
        return False
    if query_duration_sec is None or query_duration_sec < MULTI_SPEAKER_RUNOFF_MIN_DURATION:
        return False
    if query_quality_score is None or query_quality_score < MULTI_SPEAKER_RUNOFF_MIN_QUALITY:
        return False
    if best_score < MULTI_SPEAKER_RUNOFF_MIN_TOP1 or top2_score < MULTI_SPEAKER_RUNOFF_MIN_TOP2:
        return False
    if best_score >= effective_threshold:
        return False
    if (best_score - top2_score) > MULTI_SPEAKER_RUNOFF_MAX_TOP1_TOP2_GAP:
        return False
    if (top2_score - top3_score) < MULTI_SPEAKER_RUNOFF_MIN_TOP2_TOP3_GAP:
        return False
    return True


def _should_accept_for_strong_leader_below_threshold(
    *,
    best_score: float,
    top2_score: float | None,
    effective_threshold: float,
    profile_count: int,
    query_duration_sec: float | None,
    query_quality_score: float | None,
) -> bool:
    if profile_count < 3:
        return False
    if top2_score is None:
        return False
    if query_duration_sec is None:
        return False
    if not (
        MULTI_SPEAKER_STRONG_LEADER_MIN_DURATION
        <= query_duration_sec
        <= MULTI_SPEAKER_STRONG_LEADER_MAX_DURATION
    ):
        return False
    if query_quality_score is None or query_quality_score < MULTI_SPEAKER_STRONG_LEADER_MIN_QUALITY:
        return False
    if best_score >= effective_threshold:
        return False
    if (effective_threshold - best_score) > MULTI_SPEAKER_STRONG_LEADER_MAX_SCORE_DEFICIT:
        return False
    if best_score < MULTI_SPEAKER_STRONG_LEADER_MIN_TOP1:
        return False
    return (best_score - top2_score) >= MULTI_SPEAKER_STRONG_LEADER_MIN_TOP1_TOP2_GAP


def _should_accept_for_short_calibrated_leader(
    *,
    best_score: float,
    top2_score: float | None,
    calibrated_score: float,
    effective_threshold: float,
    effective_calibrated_threshold: float | None,
    profile_count: int,
    query_duration_sec: float | None,
    query_quality_score: float | None,
) -> bool:
    if profile_count < 3:
        return False
    if top2_score is None or effective_calibrated_threshold is None:
        return False
    if query_duration_sec is None:
        return False
    if not (
        SHORT_CALIBRATED_LEADER_MIN_DURATION
        <= query_duration_sec
        <= SHORT_CALIBRATED_LEADER_MAX_DURATION
    ):
        return False
    if query_quality_score is None or query_quality_score < SHORT_CALIBRATED_LEADER_MIN_QUALITY:
        return False
    if best_score >= effective_threshold:
        return False
    if calibrated_score < effective_calibrated_threshold:
        return False
    if (effective_threshold - best_score) > SHORT_CALIBRATED_LEADER_MAX_SCORE_DEFICIT:
        return False
    if best_score < SHORT_CALIBRATED_LEADER_MIN_TOP1:
        return False
    return (best_score - top2_score) >= SHORT_CALIBRATED_LEADER_MIN_TOP1_TOP2_GAP


def _should_accept_for_gate_review_override(
    *,
    best_score: float,
    top2_score: float | None,
    calibrated_score: float,
    effective_threshold: float,
    effective_calibrated_threshold: float | None,
    query_duration_sec: float | None,
    query_quality_score: float | None,
    reranked_margin: float,
) -> bool:
    if effective_calibrated_threshold is None:
        return False
    if top2_score is None or top2_score <= best_score:
        return False
    if query_duration_sec is None or query_duration_sec < GATE_REVIEW_OVERRIDE_MIN_DURATION:
        return False
    if query_quality_score is None or query_quality_score < GATE_REVIEW_OVERRIDE_MIN_QUALITY:
        return False
    if best_score < (effective_threshold + GATE_REVIEW_OVERRIDE_MIN_RAW_SURPLUS):
        return False
    if calibrated_score < max(
        effective_calibrated_threshold,
        GATE_REVIEW_OVERRIDE_MIN_CALIBRATED_SCORE,
    ):
        return False
    return 0.0 < reranked_margin <= GATE_REVIEW_OVERRIDE_MAX_RERANKED_MARGIN


def _should_reject_for_cluster(
    *,
    best_score: float,
    top2_score: float | None,
    top3_score: float | None,
    profile_count: int,
    query_duration_sec: float | None,
    query_quality_score: float | None,
) -> bool:
    if profile_count < 3:
        return False
    if top2_score is None or top3_score is None:
        return False
    if query_duration_sec is None or query_duration_sec < MULTI_SPEAKER_CLUSTER_GUARD_MIN_DURATION:
        return False
    if query_quality_score is None or query_quality_score < MULTI_SPEAKER_CLUSTER_GUARD_MIN_QUALITY:
        return False
    if top2_score < MULTI_SPEAKER_CLUSTER_GUARD_MIN_TOP2:
        return False
    if top3_score < MULTI_SPEAKER_CLUSTER_GUARD_MIN_TOP3:
        return False
    return (best_score - top3_score) < MULTI_SPEAKER_CLUSTER_GUARD_MAX_TOP1_TOP3_GAP


def _resolve_multi_speaker_duration_threshold(
    *,
    requested_threshold: float,
    query_duration_sec: float,
) -> float:
    for upper_bound, threshold in _load_multi_speaker_duration_thresholds():
        if query_duration_sec < upper_bound:
            if threshold is None:
                return float(requested_threshold)
            return float(threshold)
    return float(requested_threshold)


def _load_multi_speaker_duration_thresholds() -> tuple[tuple[float, float | None], ...]:
    raw = os.getenv("VOICE_FRAMEWORK_MULTI_SPEAKER_DURATION_THRESHOLDS", "").strip()
    if not raw:
        return DEFAULT_MULTI_SPEAKER_DURATION_THRESHOLDS
    rules: list[tuple[float, float | None]] = []
    for chunk in raw.split(","):
        upper_text, threshold_text = chunk.strip().split(":", maxsplit=1)
        normalized = threshold_text.strip().lower()
        threshold = None if normalized == "keep" else float(normalized)
        rules.append((float(upper_text), threshold))
    if not rules:
        return DEFAULT_MULTI_SPEAKER_DURATION_THRESHOLDS
    return tuple(rules)


def _resolve_margin_threshold(
    *,
    profile_count: int,
    query_duration_sec: float | None,
) -> float:
    if profile_count <= 1 or query_duration_sec is None:
        return float(MULTI_SPEAKER_MIN_MARGIN)
    for upper_bound, threshold in DEFAULT_MULTI_SPEAKER_MARGIN_THRESHOLDS:
        if query_duration_sec < upper_bound:
            return float(threshold)
    return float(MULTI_SPEAKER_MIN_MARGIN)
