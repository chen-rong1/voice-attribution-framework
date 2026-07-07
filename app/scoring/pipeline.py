"""Open-set scoring pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from app.common.constants import DEFAULT_REJECT_LABEL
from app.scoring.models import (
    AcceptReason,
    DecisionLabel,
    DecisionResult,
    RejectReason,
    ScoringStrategy,
)
from app.scoring.open_set_gate import OpenSetDecision, OpenSetGateEvidence, evaluate_open_set_gate
from app.scoring.reranker import build_candidate_scores


def run_scoring_pipeline(
    query_embedding: np.ndarray,
    profiles: list,
    *,
    effective_threshold: float,
    scoring_strategy: ScoringStrategy,
    scoring_fn: Callable,
    query_duration_sec: float | None,
    query_quality_score: float | None,
    raw_score_breakdown: dict[str, float],
    accepts_normally: bool,
    accepts_for_runoff: bool,
    accepts_for_strong_leader: bool,
    accepts_for_short_calibrated_leader: bool,
    accepts_for_gate_review_override: bool,
    metadata: dict[str, Any],
    candidates: list | None = None,
) -> DecisionResult:
    """Run candidate reranking and open-set gating on top of raw scoring."""

    if candidates is None:
        candidates = build_candidate_scores(
            query_embedding,
            profiles,
            scoring_fn=scoring_fn,
        )
    best_candidate = candidates[0]
    second_candidate = candidates[1] if len(candidates) > 1 else None
    open_set_margin = float(
        best_candidate.calibrated_score - second_candidate.calibrated_score
        if second_candidate is not None
        else best_candidate.calibrated_score
    )
    top_k_candidates = candidates[: max(1, min(3, len(candidates)))]
    peer_candidates = top_k_candidates[1:]
    top1_topk_mean_gap = float(
        best_candidate.calibrated_score
        - (
            float(np.mean([candidate.calibrated_score for candidate in peer_candidates]))
            if peer_candidates
            else best_candidate.calibrated_score
        )
    )
    reranked_margin = float(
        best_candidate.reranked_score - second_candidate.reranked_score
        if second_candidate is not None
        else best_candidate.reranked_score
    )

    metadata["z_norm_score"] = float(best_candidate.z_norm_score)
    metadata["adaptive_s_norm_score"] = float(best_candidate.adaptive_s_norm_score)
    metadata["calibrated_score"] = float(best_candidate.calibrated_score)
    metadata["cohort_relative_score"] = float(best_candidate.cohort_relative_score)
    metadata["member_consistency_score"] = float(best_candidate.member_consistency_score)
    metadata["sub_center_score"] = float(best_candidate.sub_center_score)
    metadata["reranked_score"] = float(best_candidate.reranked_score)
    metadata["open_set_margin"] = open_set_margin
    metadata["top1_topk_mean_gap"] = top1_topk_mean_gap
    metadata["reranked_margin"] = reranked_margin
    metadata["profile_open_set_floor"] = float(best_candidate.profile.open_set_floor)
    metadata["profile_calibrated_threshold"] = float(best_candidate.profile.calibrated_threshold)
    metadata["profile_risk_level"] = best_candidate.profile.risk_level.value
    metadata["calibration_status"] = str(best_candidate.profile.metadata.get("calibration_status", ""))
    metadata["calibration_type"] = str(best_candidate.profile.metadata.get("calibration_type", ""))
    metadata["calibration_scale"] = float(best_candidate.profile.metadata.get("calibration_scale", 1.0))
    metadata["calibration_bias"] = float(best_candidate.profile.metadata.get("calibration_bias", 0.0))
    metadata["top1_raw_score"] = float(best_candidate.raw_score)
    if second_candidate is not None:
        metadata["top2_calibrated_score"] = float(second_candidate.calibrated_score)
        metadata["top2_reranked_score"] = float(second_candidate.reranked_score)

    gate_enabled = len(profiles) >= 3 and all(
        "impostor_score_mean" in profile.metadata for profile in profiles
    )
    metadata["open_set_gate_enabled"] = int(gate_enabled)

    if gate_enabled:
        gate_decision = evaluate_open_set_gate(
            OpenSetGateEvidence(
                raw_score=best_candidate.raw_score,
                z_norm_score=best_candidate.z_norm_score,
                adaptive_s_norm_score=best_candidate.adaptive_s_norm_score,
                calibrated_score=best_candidate.calibrated_score,
                cohort_relative_score=best_candidate.cohort_relative_score,
                open_set_margin=open_set_margin,
                top1_topk_mean_gap=top1_topk_mean_gap,
                reranked_margin=reranked_margin,
                member_consistency_score=best_candidate.member_consistency_score,
                effective_threshold=effective_threshold,
                open_set_floor=best_candidate.profile.open_set_floor,
                calibrated_threshold=_resolve_gate_calibrated_threshold(best_candidate.profile),
                query_duration_sec=query_duration_sec,
                query_quality_score=query_quality_score,
                profile_risk_level=best_candidate.profile.risk_level,
            )
        )
    else:
        gate_decision = OpenSetDecision.ACCEPT
    metadata["open_set_gate_decision"] = gate_decision.value

    if (
        accepts_normally
        or accepts_for_runoff
        or accepts_for_strong_leader
        or accepts_for_short_calibrated_leader
        or accepts_for_gate_review_override
    ):
        if gate_decision == OpenSetDecision.REJECT_UNKNOWN:
            metadata["accept_reason"] = ""
            metadata["reject_reason"] = RejectReason.OPEN_SET_GATE.value
            metadata["decision_reason"] = RejectReason.OPEN_SET_GATE.value
            metadata["decision_evidence"] = _build_decision_evidence(metadata)
            return DecisionResult(
                decision=DecisionLabel.REJECT,
                final_label=DEFAULT_REJECT_LABEL,
                best_speaker_id=None,
                best_score=best_candidate.raw_score,
                threshold_value=effective_threshold,
                scoring_strategy=scoring_strategy,
                score_breakdown=raw_score_breakdown,
                metadata=metadata,
            )
        if gate_decision == OpenSetDecision.REVIEW and not accepts_for_gate_review_override:
            metadata["accept_reason"] = ""
            metadata["reject_reason"] = RejectReason.REVIEW.value
            metadata["decision_reason"] = RejectReason.REVIEW.value
            metadata["decision_evidence"] = _build_decision_evidence(metadata)
            return DecisionResult(
                decision=DecisionLabel.REVIEW,
                final_label=DEFAULT_REJECT_LABEL,
                best_speaker_id=best_candidate.speaker_id,
                best_score=best_candidate.raw_score,
                threshold_value=effective_threshold,
                scoring_strategy=scoring_strategy,
                score_breakdown=raw_score_breakdown,
                metadata=metadata,
            )
        metadata["accept_reason"] = str(
            metadata.get("accept_reason", AcceptReason.NORMAL_ACCEPT.value)
        )
        metadata["decision_reason"] = str(metadata["accept_reason"])
        metadata["decision_evidence"] = _build_decision_evidence(metadata)
        return DecisionResult(
            decision=DecisionLabel.ACCEPT,
            final_label=best_candidate.speaker_id,
            best_speaker_id=best_candidate.speaker_id,
            best_score=best_candidate.raw_score,
            threshold_value=effective_threshold,
            scoring_strategy=scoring_strategy,
            score_breakdown=raw_score_breakdown,
            metadata=metadata,
        )
    metadata["reject_reason"] = str(
        metadata.get("reject_reason", RejectReason.REJECTED.value)
    )
    metadata["decision_reason"] = str(metadata["reject_reason"])
    metadata["decision_evidence"] = _build_decision_evidence(metadata)

    return DecisionResult(
        decision=DecisionLabel.REJECT,
        final_label=DEFAULT_REJECT_LABEL,
        best_speaker_id=None,
        best_score=best_candidate.raw_score,
        threshold_value=effective_threshold,
        scoring_strategy=scoring_strategy,
        score_breakdown=raw_score_breakdown,
        metadata=metadata,
    )


def _build_decision_evidence(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "v2",
        "summary": {
            "decision_reason": str(metadata.get("decision_reason", "")),
            "accept_reason": str(metadata.get("accept_reason", "")),
            "reject_reason": str(metadata.get("reject_reason", "")),
        },
        "score_evidence": {
            "requested_threshold_value": float(metadata.get("requested_threshold_value", 0.0)),
            "effective_threshold_value": float(metadata.get("effective_threshold_value", 0.0)),
            "effective_calibrated_threshold_value": (
                float(metadata.get("effective_calibrated_threshold_value", 0.0))
                if metadata.get("effective_calibrated_threshold_value", "") != ""
                else None
            ),
            "top1_score": float(metadata.get("top1_score", 0.0)),
            "top1_raw_score": float(metadata.get("top1_raw_score", 0.0)),
            "z_norm_score": float(metadata.get("z_norm_score", 0.0)),
            "adaptive_s_norm_score": float(metadata.get("adaptive_s_norm_score", 0.0)),
            "calibrated_score": float(metadata.get("calibrated_score", 0.0)),
            "cohort_relative_score": float(metadata.get("cohort_relative_score", 0.0)),
            "member_consistency_score": float(metadata.get("member_consistency_score", 0.0)),
            "sub_center_score": float(metadata.get("sub_center_score", 0.0)),
            "reranked_score": float(metadata.get("reranked_score", 0.0)),
            "open_set_margin": float(metadata.get("open_set_margin", 0.0)),
            "top1_topk_mean_gap": float(metadata.get("top1_topk_mean_gap", 0.0)),
            "reranked_margin": float(metadata.get("reranked_margin", 0.0)),
            "margin": float(metadata.get("margin", 0.0)),
            "calibrated_margin": float(metadata.get("calibrated_margin", 0.0)),
            "effective_margin_threshold": float(metadata.get("effective_margin_threshold", 0.0)),
        },
        "gate_evidence": {
            "open_set_gate_enabled": int(metadata.get("open_set_gate_enabled", 0)),
            "open_set_gate_decision": str(metadata.get("open_set_gate_decision", "")),
        },
        "candidate_evidence": {
            "top1_speaker_id": str(metadata.get("top1_speaker_id", "")),
            "top2_speaker_id": str(metadata.get("top2_speaker_id", ""))
            if "top2_speaker_id" in metadata
            else None,
            "top3_speaker_id": str(metadata.get("top3_speaker_id", ""))
            if "top3_speaker_id" in metadata
            else None,
            "top2_score": float(metadata.get("top2_score", 0.0))
            if "top2_score" in metadata
            else None,
            "top3_score": float(metadata.get("top3_score", 0.0))
            if "top3_score" in metadata
            else None,
            "top2_calibrated_score": float(metadata.get("top2_calibrated_score", 0.0))
            if "top2_calibrated_score" in metadata
            else None,
            "top2_reranked_score": float(metadata.get("top2_reranked_score", 0.0))
            if "top2_reranked_score" in metadata
            else None,
        },
        "profile_evidence": {
            "profile_open_set_floor": float(metadata.get("profile_open_set_floor", 0.0)),
            "profile_calibrated_threshold": float(
                metadata.get("profile_calibrated_threshold", 0.0)
            ),
            "profile_risk_level": str(metadata.get("profile_risk_level", "")),
            "calibration_status": str(metadata.get("calibration_status", "")),
            "calibration_type": str(metadata.get("calibration_type", "")),
            "calibration_scale": float(metadata.get("calibration_scale", 1.0)),
            "calibration_bias": float(metadata.get("calibration_bias", 0.0)),
        },
        "query_evidence": {
            "query_duration_sec": float(metadata.get("query_duration_sec", 0.0))
            if "query_duration_sec" in metadata
            else None,
            "query_quality_score": float(metadata.get("query_quality_score", 0.0))
            if "query_quality_score" in metadata
            else None,
        },
    }


def _resolve_gate_calibrated_threshold(profile) -> float | None:
    if str(profile.metadata.get("calibration_status", "")) != "heldout_calibrated":
        return None
    return float(profile.calibrated_threshold)
