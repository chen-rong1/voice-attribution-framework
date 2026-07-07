"""Contracts shared by all embedding backends."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.audio.models import AudioChunk
from app.features.models import FeatureMatrix


@dataclass(slots=True)
class EmbeddingRequest:
    """A backend request carrying either normalized audio or a feature matrix."""

    sample_id: str
    audio: AudioChunk | None = None
    features: FeatureMatrix | None = None
    metadata: dict[str, str | float | int] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddingResult:
    """A standard embedding response returned by all backends."""

    sample_id: str
    backend_name: str
    backend_version: str
    feature_version: str
    embedding: np.ndarray
    embedding_dim: int
    duration_sec: float | None = None
    quality_score: float | None = None
    metadata: dict[str, str | float | int] = field(default_factory=dict)
