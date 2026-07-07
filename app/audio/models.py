"""Typed audio data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class AudioChunk:
    """A normalized mono waveform ready for downstream processing."""

    waveform: np.ndarray
    sample_rate: int
    source_path: Path | None = None
    duration_sec: float | None = None
    metadata: dict[str, str | float | int] = field(default_factory=dict)
