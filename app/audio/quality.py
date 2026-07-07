"""Simple audio quality heuristics used by the first framework iteration."""

from __future__ import annotations


def estimate_audio_quality_score(
    *,
    duration_sec: float | None,
    rms: float | None,
) -> float:
    """Estimate a lightweight quality score from duration and waveform energy."""

    duration_value = max(duration_sec or 0.0, 0.0)
    rms_value = max(rms or 0.0, 0.0)
    duration_component = min(duration_value / 3.0, 1.0)
    energy_component = min(rms_value / 0.12, 1.0)
    score = 0.75 * duration_component + 0.25 * energy_component
    return max(0.05, min(score, 1.0))
