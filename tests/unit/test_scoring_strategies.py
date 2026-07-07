import numpy as np

from app.embedding_backends.models import EmbeddingResult
from app.profiles.builder import build_speaker_profile
from app.profiles.models import SpeakerEmbeddingSample, SpeakerProfile
from app.scoring.models import DecisionLabel, ScoringStrategy
from app.scoring.strategies import build_decision, cosine_similarity


def _profile(speaker_id: str, vector: list[float]) -> SpeakerProfile:
    embedding = np.asarray(vector, dtype=np.float32)
    result = EmbeddingResult(
        sample_id=f"{speaker_id}-sample",
        backend_name="dummy",
        backend_version="0.1.0",
        feature_version="fbank80_v1",
        embedding=embedding,
        embedding_dim=embedding.shape[0],
    )
    return SpeakerProfile(
        speaker_id=speaker_id,
        profile_name="default",
        backend_name="dummy",
        backend_version="0.1.0",
        feature_version="fbank80_v1",
        aggregation_strategy="center",
        vector=embedding,
        members=[SpeakerEmbeddingSample(speaker_id=speaker_id, embedding_result=result)],
    )


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    value = cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    assert value == 1.0


def test_build_decision_accepts_when_best_score_exceeds_threshold() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profiles = [_profile("alice", [1.0, 0.0]), _profile("bob", [0.0, 1.0])]
    result = build_decision(
        query,
        profiles,
        threshold_value=0.5,
        scoring_strategy=ScoringStrategy.CENTER,
    )
    assert result.decision == DecisionLabel.ACCEPT
    assert result.final_label == "alice"


def test_build_decision_rejects_when_threshold_is_not_met() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profiles = [_profile("alice", [0.1, 1.0]), _profile("bob", [0.0, 1.0])]
    result = build_decision(
        query,
        profiles,
        threshold_value=0.95,
        scoring_strategy=ScoringStrategy.CENTER,
    )
    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"


def test_top_k_mean_uses_average_of_top_member_scores() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    samples = [
        EmbeddingResult(
            sample_id="alice-1",
            backend_name="dummy",
            backend_version="0.1.0",
            feature_version="fbank80_v1",
            embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            embedding_dim=2,
        ),
        EmbeddingResult(
            sample_id="alice-2",
            backend_name="dummy",
            backend_version="0.1.0",
            feature_version="fbank80_v1",
            embedding=np.asarray([0.8, 0.2], dtype=np.float32),
            embedding_dim=2,
        ),
        EmbeddingResult(
            sample_id="alice-3",
            backend_name="dummy",
            backend_version="0.1.0",
            feature_version="fbank80_v1",
            embedding=np.asarray([-1.0, 0.0], dtype=np.float32),
            embedding_dim=2,
        ),
    ]
    profile = build_speaker_profile("alice", samples)
    profile.metadata["default_top_k"] = 2
    result = build_decision(
        query,
        [profile],
        threshold_value=0.1,
        scoring_strategy=ScoringStrategy.TOP_K_MEAN,
    )
    assert result.decision == DecisionLabel.ACCEPT
    assert result.best_score > 0.8


def test_quality_weighted_center_prefers_higher_quality_member() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profile = SpeakerProfile(
        speaker_id="alice",
        profile_name="default",
        backend_name="dummy",
        backend_version="0.1.0",
        feature_version="fbank80_v1",
        aggregation_strategy="quality_weighted_center",
        vector=np.asarray([0.0, 1.0], dtype=np.float32),
        members=[
            SpeakerEmbeddingSample(
                speaker_id="alice",
                embedding_result=EmbeddingResult(
                    sample_id="a1",
                    backend_name="dummy",
                    backend_version="0.1.0",
                    feature_version="fbank80_v1",
                    embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                    embedding_dim=2,
                    quality_score=1.0,
                ),
                weight_value=1.0,
            ),
            SpeakerEmbeddingSample(
                speaker_id="alice",
                embedding_result=EmbeddingResult(
                    sample_id="a2",
                    backend_name="dummy",
                    backend_version="0.1.0",
                    feature_version="fbank80_v1",
                    embedding=np.asarray([0.0, 1.0], dtype=np.float32),
                    embedding_dim=2,
                    quality_score=0.1,
                ),
                weight_value=0.1,
            ),
        ],
    )
    result = build_decision(
        query,
        [profile],
        threshold_value=0.1,
        scoring_strategy=ScoringStrategy.QUALITY_WEIGHTED_CENTER,
    )
    assert result.decision == DecisionLabel.ACCEPT
    assert result.best_score > 0.9
