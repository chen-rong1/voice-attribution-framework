"""Framework-specific exception types."""

from __future__ import annotations

import json
from dataclasses import dataclass


class VoiceAttributionError(Exception):
    """Base exception for the framework."""


class BackendAlreadyRegisteredError(VoiceAttributionError):
    """Raised when the same backend name is registered twice."""


class BackendNotFoundError(VoiceAttributionError):
    """Raised when a backend cannot be resolved from the registry."""


class BackendNotLoadedError(VoiceAttributionError):
    """Raised when extraction is requested before a backend is loaded."""


class InvalidAudioInputError(VoiceAttributionError):
    """Raised when an audio chunk does not meet the normalization contract."""


class EmbeddingExtractionError(VoiceAttributionError):
    """Raised when a backend fails to return a valid embedding."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One structured validation issue."""

    code: str
    message: str
    row_number: int | None = None
    column_name: str | None = None

    def to_cli_line(self) -> str:
        parts = [f"code={self.code}"]
        if self.row_number is not None:
            parts.append(f"row={self.row_number}")
        if self.column_name:
            parts.append(f"column={self.column_name}")
        parts.append(f"message={self.message}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "code": self.code,
            "message": self.message,
            "row_number": self.row_number,
            "column_name": self.column_name,
        }


class InvalidManifestError(VoiceAttributionError):
    """Raised when a benchmark manifest does not meet the expected schema."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        if not issues:
            raise ValueError("InvalidManifestError requires at least one issue.")
        self.issues = issues
        super().__init__(self.__str__())

    def __str__(self) -> str:
        return "; ".join(issue.to_cli_line() for issue in self.issues)

    def to_cli_text(self) -> str:
        lines = ["manifest 校验失败:"]
        lines.extend(f"- {issue.to_cli_line()}" for issue in self.issues)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "error_type": "InvalidManifestError",
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
