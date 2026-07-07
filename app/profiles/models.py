"""Profile-layer data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.embedding_backends.models import EmbeddingResult


@dataclass(slots=True)
class SpeakerEmbeddingSample:
    """One registered embedding sample attached to a speaker."""

    speaker_id: str
    embedding_result: EmbeddingResult
    weight_value: float = 1.0


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
    members: list[SpeakerEmbeddingSample] = field(default_factory=list)
    metadata: dict[str, str | float | int] = field(default_factory=dict)
