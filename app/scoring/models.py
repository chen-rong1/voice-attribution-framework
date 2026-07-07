"""Scoring-layer data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.profiles.models import SpeakerProfile


class ScoringStrategy(StrEnum):
    CENTER = "center"
    MAX = "max"
    TOP_K_MEAN = "top_k_mean"
    QUALITY_WEIGHTED_CENTER = "quality_weighted_center"


class DecisionLabel(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


@dataclass(slots=True)
class ScoreEntry:
    """A scored speaker candidate."""

    speaker_id: str
    score: float
    profile: SpeakerProfile


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
    metadata: dict[str, str | float | int] = field(default_factory=dict)
