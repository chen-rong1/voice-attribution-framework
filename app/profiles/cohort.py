"""Helpers for constructing cohort-based impostor statistics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from app.profiles.models import SpeakerProfile


def collect_impostor_vectors(
    profiles: Iterable[SpeakerProfile],
    *,
    excluded_speaker_id: str,
) -> list[np.ndarray]:
    """Collect member vectors from all speakers except the current profile."""

    impostor_vectors: list[np.ndarray] = []
    for profile in profiles:
        if profile.speaker_id == excluded_speaker_id:
            continue
        if profile.member_vectors:
            impostor_vectors.extend(profile.member_vectors)
            continue
        impostor_vectors.append(profile.center_vector.copy())
    return impostor_vectors
