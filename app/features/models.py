"""Feature-layer data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class FeatureMatrix:
    """A feature tensor produced from a normalized audio chunk."""

    values: np.ndarray
    feature_version: str
    num_frames: int
    metadata: dict[str, str | float | int] = field(default_factory=dict)
