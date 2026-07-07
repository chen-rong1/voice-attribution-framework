"""Reporting data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ReportArtifact:
    """A generated report artifact path and its semantic role."""

    artifact_type: str
    path: Path
