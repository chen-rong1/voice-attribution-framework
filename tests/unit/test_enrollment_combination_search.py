from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.embedding_backends.models import EmbeddingResult
from app.profiles.builder import build_speaker_profile
from app.scoring.models import ScoringStrategy


def _load_search_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "search_best_enrollment_combination.py"
    )
    spec = importlib.util.spec_from_file_location("search_best_enrollment_combination", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _Clip:
    clip_id: str
    expected_label: str


def _embedding(sample_id: str, vector: list[float]) -> EmbeddingResult:
    array = np.asarray(vector, dtype=np.float32)
    return EmbeddingResult(
        sample_id=sample_id,
        backend_name="dummy",
        backend_version="0.1.0",
        feature_version="fbank80_v1",
        embedding=array,
        embedding_dim=array.shape[0],
        quality_score=1.0,
        duration_sec=2.0,
    )


def test_combination_count_matches_expected_value() -> None:
    module = _load_search_module()
    assert module.combination_count(15, 4) == 1365
    assert module.combination_count(3, 4) == 0


def test_evaluate_profile_with_clipset_reports_correct_metrics() -> None:
    module = _load_search_module()
    enrollments = [
        _embedding("enroll-1", [1.0, 0.0]),
        _embedding("enroll-2", [0.9, 0.1]),
    ]
    profile = build_speaker_profile("xiaoli", enrollments, aggregation_strategy="center")
    clip_embeddings = {
        "p1": _embedding("p1", [1.0, 0.0]),
        "p2": _embedding("p2", [0.9, 0.1]),
        "u1": _embedding("u1", [0.0, 1.0]),
        "u2": _embedding("u2", [0.7, 0.3]),
    }
    clip_catalog = {
        "p1": _Clip("p1", "xiaoli"),
        "p2": _Clip("p2", "xiaoli"),
        "u1": _Clip("u1", "UNKNOWN"),
        "u2": _Clip("u2", "UNKNOWN"),
    }

    result = module.evaluate_profile_with_clipset(
        speaker_id="xiaoli",
        profile=profile,
        clips_to_score=list(clip_embeddings.keys()),
        clip_catalog=clip_catalog,
        clip_embeddings=clip_embeddings,
        strategy=ScoringStrategy.MAX,
        threshold=0.8,
    )

    assert result["correct"] == 3
    assert result["total"] == 4
    assert result["positive_correct"] == 2
    assert result["unknown_correct"] == 1
    assert result["false_accepts"] == ["u2"]
    assert result["false_rejects"] == []
