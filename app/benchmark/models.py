"""Benchmark configuration contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.common.constants import DEFAULT_REJECT_LABEL
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
    dataset_role: str = "general"


@dataclass(slots=True)
class BenchmarkClip:
    """One labeled test clip used in a benchmark run."""

    clip_id: str
    audio_path: Path
    expected_label: str
    truth_label: str
    evaluation_group: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


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
    latency_ms: float
    is_correct: bool
    evaluation_group: str
    score_breakdown: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


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
    def average_latency_ms(self) -> float:
        if not self.items:
            return 0.0
        return sum(item.latency_ms for item in self.items) / self.total

    @property
    def max_latency_ms(self) -> float:
        if not self.items:
            return 0.0
        return max(item.latency_ms for item in self.items)

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
    def accept_count(self) -> int:
        return sum(1 for item in self.items if item.decision == "ACCEPT")

    @property
    def reject_count(self) -> int:
        return sum(1 for item in self.items if item.decision == "REJECT")

    @property
    def review_count(self) -> int:
        return sum(1 for item in self.items if item.decision == "REVIEW")

    @property
    def false_accept_count(self) -> int:
        return sum(
            1
            for item in self.items
            if item.expected_label == DEFAULT_REJECT_LABEL and item.final_label != DEFAULT_REJECT_LABEL
        )

    @property
    def false_reject_count(self) -> int:
        return sum(
            1
            for item in self.items
            if item.expected_label != DEFAULT_REJECT_LABEL and item.final_label == DEFAULT_REJECT_LABEL
        )

    @property
    def external_known_total(self) -> int:
        return sum(1 for item in self.items if item.evaluation_group == "external_known")

    @property
    def external_known_correct(self) -> int:
        return sum(
            1
            for item in self.items
            if item.evaluation_group == "external_known" and item.final_label == item.expected_label
        )

    @property
    def external_known_top1_accuracy(self) -> float:
        if self.external_known_total == 0:
            return 0.0
        return self.external_known_correct / self.external_known_total

    @property
    def external_unknown_total(self) -> int:
        return sum(1 for item in self.items if item.evaluation_group == "external_unknown")

    @property
    def external_unknown_correct(self) -> int:
        return sum(
            1
            for item in self.items
            if item.evaluation_group == "external_unknown"
            and item.final_label == DEFAULT_REJECT_LABEL
        )

    @property
    def external_unknown_reject_rate(self) -> float:
        if self.external_unknown_total == 0:
            return 0.0
        return self.external_unknown_correct / self.external_unknown_total

    @property
    def high_risk_false_accept_count(self) -> int:
        return sum(
            1
            for item in self.items
            if item.expected_label == DEFAULT_REJECT_LABEL
            and item.final_label != DEFAULT_REJECT_LABEL
            and str(item.metadata.get("profile_risk_level", "")) == "high"
        )

    @property
    def accept_reason_counts(self) -> dict[str, int]:
        return self._count_reasons_for_decision("ACCEPT")

    @property
    def reject_reason_counts(self) -> dict[str, int]:
        return self._count_reasons_for_decision("REJECT")

    @property
    def review_reason_counts(self) -> dict[str, int]:
        return self._count_reasons_for_decision("REVIEW")

    @property
    def calibration_status_counts(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for item in self.items:
            counter[str(item.metadata.get("calibration_status", "unknown"))] += 1
        return dict(sorted(counter.items()))

    @property
    def heldout_calibrated_count(self) -> int:
        return sum(
            1
            for item in self.items
            if str(item.metadata.get("calibration_status", "")) == "heldout_calibrated"
        )

    @property
    def decision_reason_stats(self) -> dict[str, dict[str, int | str]]:
        stats: dict[str, dict[str, int | str]] = {}
        for item in self.items:
            reason = self._resolve_item_reason(item)
            bucket = stats.setdefault(
                reason,
                {
                    "decision": item.decision,
                    "count": 0,
                    "correct": 0,
                    "incorrect": 0,
                },
            )
            bucket["count"] = int(bucket["count"]) + 1
            if item.is_correct:
                bucket["correct"] = int(bucket["correct"]) + 1
            else:
                bucket["incorrect"] = int(bucket["incorrect"]) + 1
        return dict(sorted(stats.items()))

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "run_name": self.config.run_name,
            "dataset_name": self.config.dataset_name,
            "dataset_role": self.config.dataset_role,
            "backend_name": self.config.backend_name,
            "scoring_strategy": self.config.scoring_strategy.value,
            "threshold_value": self.config.threshold_value,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "average_latency_ms": self.average_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "positive_total": self.positive_total,
            "positive_correct": self.positive_correct,
            "positive_recall": self.positive_recall,
            "unknown_total": self.unknown_total,
            "unknown_correct": self.unknown_correct,
            "unknown_reject_rate": self.unknown_reject_rate,
            "accept_count": self.accept_count,
            "reject_count": self.reject_count,
            "review_count": self.review_count,
            "false_accept_count": self.false_accept_count,
            "false_reject_count": self.false_reject_count,
            "external_known_total": self.external_known_total,
            "external_known_correct": self.external_known_correct,
            "external_known_top1_accuracy": self.external_known_top1_accuracy,
            "external_unknown_total": self.external_unknown_total,
            "external_unknown_correct": self.external_unknown_correct,
            "external_unknown_reject_rate": self.external_unknown_reject_rate,
            "high_risk_false_accept_count": self.high_risk_false_accept_count,
            "accept_reason_counts": self.accept_reason_counts,
            "reject_reason_counts": self.reject_reason_counts,
            "review_reason_counts": self.review_reason_counts,
            "calibration_status_counts": self.calibration_status_counts,
            "heldout_calibrated_count": self.heldout_calibrated_count,
            "decision_reason_stats": self.decision_reason_stats,
        }

    def _count_reasons_for_decision(self, decision: str) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for item in self.items:
            if item.decision != decision:
                continue
            counter[self._resolve_item_reason(item)] += 1
        return dict(sorted(counter.items()))

    def _resolve_item_reason(self, item: BenchmarkItemResult) -> str:
        decision_reason = str(item.metadata.get("decision_reason", "")).strip()
        accept_reason = str(item.metadata.get("accept_reason", "")).strip()
        reject_reason = str(item.metadata.get("reject_reason", "")).strip()
        if decision_reason:
            return decision_reason
        if item.decision == "ACCEPT":
            return accept_reason or "normal_accept"
        if item.decision == "REVIEW":
            return reject_reason or "review"
        return reject_reason or "rejected"
