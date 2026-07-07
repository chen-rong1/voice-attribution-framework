"""Scoring-layer data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.profiles.models import SpeakerProfile


class ScoringStrategy(StrEnum):
    CENTER = "center"
    MAX = "max"
    TOP_K_MEAN = "top_k_mean"
    QUALITY_WEIGHTED_CENTER = "quality_weighted_center"


class DecisionLabel(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REVIEW = "REVIEW"


class AcceptReason(StrEnum):
    NORMAL_ACCEPT = "normal_accept"
    TWO_CANDIDATE_RUNOFF = "two_candidate_runoff"
    STRONG_LEADER_BELOW_THRESHOLD = "strong_leader_below_threshold"
    SHORT_CALIBRATED_LEADER = "short_calibrated_leader"
    GATE_REVIEW_OVERRIDE = "gate_review_override"


class RejectReason(StrEnum):
    BELOW_THRESHOLD = "below_threshold"
    CALIBRATED_OVERRIDE_RAW_DEFICIT = "calibrated_override_raw_deficit"
    LOW_MARGIN = "low_margin"
    CROWDED_HIGH_SCORE_CLUSTER = "crowded_high_score_cluster"
    OPEN_SET_GATE = "open_set_gate"
    HIGH_RISK_PROFILE_GUARD = "high_risk_profile_guard"
    REVIEW = "review"
    REJECTED = "rejected"


@dataclass(slots=True)
class ScoreEntry:
    """A scored speaker candidate."""

    speaker_id: str
    score: float
    profile: SpeakerProfile


@dataclass(slots=True)
class CandidateScore:
    """Expanded candidate metrics used by the new scoring pipeline."""

    speaker_id: str
    profile: SpeakerProfile
    raw_score: float
    z_norm_score: float
    adaptive_s_norm_score: float
    calibrated_score: float
    cohort_relative_score: float
    member_consistency_score: float
    sub_center_score: float
    reranked_score: float


@dataclass(slots=True)
class DecisionResult:
    """Final decision returned by the scoring layer."""

    decision: DecisionLabel
    final_label: str
    best_speaker_id: str | None
    best_score: float
    threshold_value: float
    scoring_strategy: ScoringStrategy
    score_breakdown: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
