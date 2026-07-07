"""Helpers for turning enrollment embeddings into speaker profiles."""

from __future__ import annotations

import numpy as np

from app.common.constants import DEFAULT_PROFILE_NAME
from app.embedding_backends.models import EmbeddingResult
from app.profiles.models import SpeakerEmbeddingSample, SpeakerProfile


def build_speaker_profile(
    speaker_id: str,
    embedding_results: list[EmbeddingResult],
    *,
    profile_name: str = DEFAULT_PROFILE_NAME,
    aggregation_strategy: str = "center",
) -> SpeakerProfile:
    """Build one speaker profile from one or more enrollment embeddings."""

    if not embedding_results:
        raise ValueError("At least one embedding result is required to build a profile.")

    samples = [
        SpeakerEmbeddingSample(
            speaker_id=speaker_id,
            embedding_result=result,
            weight_value=_resolve_sample_weight(result),
        )
        for result in embedding_results
    ]
    if aggregation_strategy == "quality_weighted_center":
        weights = np.asarray([sample.weight_value for sample in samples], dtype=np.float32)
        weights = weights / np.sum(weights)
        vector = np.average(
            np.stack([sample.embedding_result.embedding for sample in samples], axis=0),
            axis=0,
            weights=weights,
        ).astype(np.float32)
    else:
        vector = np.mean(
            np.stack([result.embedding for result in embedding_results], axis=0),
            axis=0,
        ).astype(np.float32)
    first = embedding_results[0]
    return SpeakerProfile(
        speaker_id=speaker_id,
        profile_name=profile_name,
        backend_name=first.backend_name,
        backend_version=first.backend_version,
        feature_version=first.feature_version,
        aggregation_strategy=aggregation_strategy,
        vector=vector,
        members=samples,
        metadata={
            "sample_count": len(samples),
            "avg_quality_score": float(
                np.mean([sample.weight_value for sample in samples], dtype=np.float32)
            ),
            "default_top_k": min(3, len(samples)),
        },
    )


def _resolve_sample_weight(result: EmbeddingResult) -> float:
    quality_score = result.quality_score if result.quality_score is not None else 1.0
    duration_bonus = min((result.duration_sec or 0.0) / 6.0, 1.0)
    return float(max(0.1, 0.7 * quality_score + 0.3 * duration_bonus))
