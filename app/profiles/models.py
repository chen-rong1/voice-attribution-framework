"""Profile-layer data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from app.embedding_backends.models import EmbeddingResult


@dataclass(slots=True)
class SpeakerEmbeddingSample:
    """One registered embedding sample attached to a speaker."""

    speaker_id: str
    embedding_result: EmbeddingResult
    weight_value: float = 1.0


class ProfileRiskLevel(StrEnum):
    """Risk level attached to one speaker profile."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True)
class SpeakerProfile:
    """A speaker profile built from one or more registered embedding samples."""

    speaker_id: str
    profile_name: str
    backend_name: str
    backend_version: str
    feature_version: str
    aggregation_strategy: str
    vector: np.ndarray
    center_vector: np.ndarray | None = None
    members: list[SpeakerEmbeddingSample] = field(default_factory=list)
    sub_centers: list[np.ndarray] = field(default_factory=list)
    member_vectors: list[np.ndarray] = field(default_factory=list)
    intra_score_mean: float = 1.0
    intra_score_std: float = 0.0
    impostor_score_mean: float = 0.0
    impostor_score_std: float = 1.0
    open_set_floor: float = 0.0
    calibrated_threshold: float = 0.0
    risk_level: ProfileRiskLevel = ProfileRiskLevel.MEDIUM
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=np.float32)
        if self.center_vector is None:
            self.center_vector = self.vector.copy()
        else:
            self.center_vector = np.asarray(self.center_vector, dtype=np.float32)
            self.vector = self.center_vector.copy()

        if not self.member_vectors and self.members:
            self.member_vectors = [
                np.asarray(sample.embedding_result.embedding, dtype=np.float32)
                for sample in self.members
            ]
        else:
            self.member_vectors = [
                np.asarray(member_vector, dtype=np.float32)
                for member_vector in self.member_vectors
            ]

        if not self.sub_centers:
            self.sub_centers = [self.center_vector.copy()]
        else:
            self.sub_centers = [
                np.asarray(sub_center, dtype=np.float32) for sub_center in self.sub_centers
            ]

        self.metadata.setdefault("sample_count", len(self.members) or len(self.member_vectors))
        if self.members and "avg_quality_score" not in self.metadata:
            self.metadata["avg_quality_score"] = float(
                np.mean([sample.weight_value for sample in self.members], dtype=np.float32)
            )
        self.metadata.setdefault("default_top_k", min(3, max(1, len(self.member_vectors) or 1)))
