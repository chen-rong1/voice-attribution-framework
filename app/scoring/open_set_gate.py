"""Open-set gating primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.profiles.models import ProfileRiskLevel


class OpenSetDecision(StrEnum):
    ACCEPT = "accept"
    REJECT_UNKNOWN = "reject_unknown"
    REVIEW = "review"


@dataclass(slots=True)
class OpenSetGateEvidence:
    """Evidence used to decide whether a candidate survives open-set gating."""

    raw_score: float
    z_norm_score: float
    adaptive_s_norm_score: float
    calibrated_score: float
    cohort_relative_score: float
    open_set_margin: float
    top1_topk_mean_gap: float
    reranked_margin: float
    member_consistency_score: float
    effective_threshold: float
    open_set_floor: float
    calibrated_threshold: float | None
    query_duration_sec: float | None
    query_quality_score: float | None
    profile_risk_level: ProfileRiskLevel


def evaluate_open_set_gate(evidence: OpenSetGateEvidence) -> OpenSetDecision:
    """Make a conservative open-set decision using calibrated evidence."""

    if _requires_stronger_raw_evidence(evidence):
        return OpenSetDecision.REJECT_UNKNOWN
    minimum_calibrated_score = _minimum_calibrated_score(
        evidence.profile_risk_level,
        evidence.query_duration_sec,
    )
    minimum_cohort_relative_score = _minimum_cohort_relative_score(
        evidence.profile_risk_level,
        evidence.query_duration_sec,
    )
    minimum_margin = _minimum_margin(
        evidence.profile_risk_level,
        evidence.query_duration_sec,
    )

    if _should_reject_for_floor_violation(
        evidence,
        minimum_calibrated_score=minimum_calibrated_score,
        minimum_cohort_relative_score=minimum_cohort_relative_score,
        minimum_margin=minimum_margin,
    ):
        return OpenSetDecision.REJECT_UNKNOWN
    if (
        evidence.calibrated_threshold is not None
        and evidence.calibrated_score < evidence.calibrated_threshold
    ):
        return OpenSetDecision.REJECT_UNKNOWN
    if evidence.calibrated_score < minimum_calibrated_score:
        return OpenSetDecision.REJECT_UNKNOWN
    if evidence.cohort_relative_score < minimum_cohort_relative_score:
        return OpenSetDecision.REJECT_UNKNOWN
    if evidence.open_set_margin < minimum_margin:
        return OpenSetDecision.REJECT_UNKNOWN
    if evidence.top1_topk_mean_gap < _minimum_topk_mean_gap(
        evidence.profile_risk_level,
        evidence.query_duration_sec,
    ):
        return OpenSetDecision.REJECT_UNKNOWN
    if evidence.reranked_margin < _minimum_reranked_margin(
        evidence.profile_risk_level,
        evidence.query_duration_sec,
    ):
        return OpenSetDecision.REVIEW
    if (
        evidence.query_duration_sec is not None
        and evidence.query_duration_sec >= 2.0
        and evidence.query_quality_score is not None
        and evidence.query_quality_score >= 0.5
        and evidence.open_set_margin < 0.03
        and evidence.calibrated_score < 1.0
    ):
        return OpenSetDecision.REVIEW
    return OpenSetDecision.ACCEPT


def _should_reject_for_floor_violation(
    evidence: OpenSetGateEvidence,
    *,
    minimum_calibrated_score: float,
    minimum_cohort_relative_score: float,
    minimum_margin: float,
) -> bool:
    strongest_raw_signal = max(evidence.raw_score, evidence.member_consistency_score)
    floor_gap = evidence.open_set_floor - strongest_raw_signal
    if floor_gap <= _floor_tolerance(evidence):
        return False
    if evidence.calibrated_score >= minimum_calibrated_score + 0.35:
        if evidence.cohort_relative_score >= minimum_cohort_relative_score:
            return False
    if evidence.open_set_margin >= minimum_margin + 0.01:
        return False
    return True


def _floor_tolerance(evidence: OpenSetGateEvidence) -> float:
    duration = evidence.query_duration_sec or 0.0
    if evidence.profile_risk_level == ProfileRiskLevel.HIGH:
        return 0.05 if duration >= 1.5 else 0.03
    if evidence.profile_risk_level == ProfileRiskLevel.MEDIUM:
        return 0.03
    return 0.02


def _requires_stronger_raw_evidence(evidence: OpenSetGateEvidence) -> bool:
    duration = evidence.query_duration_sec or 0.0
    if evidence.profile_risk_level != ProfileRiskLevel.HIGH:
        return False
    if duration < 8.0:
        return False
    if evidence.open_set_floor > 0.24:
        return False
    minimum_raw_score = max(
        evidence.effective_threshold + 0.05,
        evidence.open_set_floor + 0.14,
    )
    return max(evidence.raw_score, evidence.member_consistency_score) < minimum_raw_score


def _minimum_calibrated_score(
    risk_level: ProfileRiskLevel,
    query_duration_sec: float | None,
) -> float:
    if risk_level == ProfileRiskLevel.HIGH:
        return 0.75 if (query_duration_sec or 0.0) >= 1.5 else 0.4
    if risk_level == ProfileRiskLevel.MEDIUM:
        return 0.3
    return 0.0


def _minimum_margin(
    risk_level: ProfileRiskLevel,
    query_duration_sec: float | None,
) -> float:
    if query_duration_sec is not None and query_duration_sec < 1.0:
        return 0.005
    if risk_level == ProfileRiskLevel.HIGH:
        return 0.02
    if risk_level == ProfileRiskLevel.MEDIUM:
        return 0.012
    return 0.008


def _minimum_cohort_relative_score(
    risk_level: ProfileRiskLevel,
    query_duration_sec: float | None,
) -> float:
    if risk_level == ProfileRiskLevel.HIGH:
        return 0.4 if (query_duration_sec or 0.0) >= 1.5 else 0.2
    if risk_level == ProfileRiskLevel.MEDIUM:
        return 0.15
    return -0.05


def _minimum_topk_mean_gap(
    risk_level: ProfileRiskLevel,
    query_duration_sec: float | None,
) -> float:
    if query_duration_sec is not None and query_duration_sec < 1.0:
        return 0.005
    if risk_level == ProfileRiskLevel.HIGH:
        return 0.02
    if risk_level == ProfileRiskLevel.MEDIUM:
        return 0.012
    return 0.008


def _minimum_reranked_margin(
    risk_level: ProfileRiskLevel,
    query_duration_sec: float | None,
) -> float:
    if query_duration_sec is not None and query_duration_sec < 1.0:
        return 0.003
    if risk_level == ProfileRiskLevel.HIGH:
        return 0.015
    if risk_level == ProfileRiskLevel.MEDIUM:
        return 0.01
    return 0.006
