"""Benchmark configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.scoring.models import ScoringStrategy


@dataclass(slots=True)
class BenchmarkRunConfig:
    """Minimum configuration needed to describe one benchmark run."""

    run_name: str
    dataset_name: str
    dataset_version: str
    backend_name: str
    scoring_strategy: ScoringStrategy
    threshold_value: float
    exclude_mixed: bool = True
    profile_aggregation_strategy: str = "quality_weighted_center"


@dataclass(slots=True)
class BenchmarkClip:
    """One labeled test clip used in a benchmark run."""

    clip_id: str
    audio_path: Path
    expected_label: str
    truth_label: str
    metadata: dict[str, str | float | int] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkItemResult:
    """Per-clip benchmark result ready for reporting."""

    clip_id: str
    audio_path: Path
    truth_label: str
    expected_label: str
    final_label: str
    decision: str
    best_score: float
    threshold_value: float
    is_correct: bool
    score_breakdown: dict[str, float]


@dataclass(slots=True)
class BenchmarkRunResult:
    """One completed benchmark run with detail rows and summary metrics."""

    config: BenchmarkRunConfig
    items: list[BenchmarkItemResult]

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def correct(self) -> int:
        return sum(1 for item in self.items if item.is_correct)

    @property
    def accuracy(self) -> float:
        if not self.items:
            return 0.0
        return self.correct / self.total

    @property
    def positive_total(self) -> int:
        return sum(1 for item in self.items if item.expected_label != "UNKNOWN")

    @property
    def positive_correct(self) -> int:
        return sum(
            1
            for item in self.items
            if item.expected_label != "UNKNOWN" and item.final_label == item.expected_label
        )

    @property
    def positive_recall(self) -> float:
        if self.positive_total == 0:
            return 0.0
        return self.positive_correct / self.positive_total

    @property
    def unknown_total(self) -> int:
        return sum(1 for item in self.items if item.expected_label == "UNKNOWN")

    @property
    def unknown_correct(self) -> int:
        return sum(
            1
            for item in self.items
            if item.expected_label == "UNKNOWN" and item.final_label == "UNKNOWN"
        )

    @property
    def unknown_reject_rate(self) -> float:
        if self.unknown_total == 0:
            return 0.0
        return self.unknown_correct / self.unknown_total

    @property
    def summary(self) -> dict[str, int | float | str]:
        return {
            "run_name": self.config.run_name,
            "dataset_name": self.config.dataset_name,
            "backend_name": self.config.backend_name,
            "scoring_strategy": self.config.scoring_strategy.value,
            "threshold_value": self.config.threshold_value,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "positive_total": self.positive_total,
            "positive_correct": self.positive_correct,
            "positive_recall": self.positive_recall,
            "unknown_total": self.unknown_total,
            "unknown_correct": self.unknown_correct,
            "unknown_reject_rate": self.unknown_reject_rate,
        }
