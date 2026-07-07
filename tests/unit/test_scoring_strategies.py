import numpy as np

from app.embedding_backends.models import EmbeddingResult
from app.profiles.builder import build_speaker_profile
from app.profiles.models import ProfileRiskLevel, SpeakerEmbeddingSample, SpeakerProfile
from app.scoring.open_set_gate import OpenSetDecision, OpenSetGateEvidence, evaluate_open_set_gate
from app.scoring.models import DecisionLabel, ScoringStrategy
from app.scoring.normalization import normalize_profile_score
from app.scoring.strategies import (
    _should_accept_for_gate_review_override,
    build_decision,
    cosine_similarity,
)


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
    assert result.metadata["decision_reason"] == "normal_accept"
    assert result.metadata["accept_reason"] == "normal_accept"
    assert result.metadata["decision_evidence"]["schema_version"] == "v2"
    assert result.metadata["decision_evidence"]["summary"]["decision_reason"] == "normal_accept"
    assert result.metadata["decision_evidence"]["candidate_evidence"]["top1_speaker_id"] == "alice"
    assert "adaptive_s_norm_score" in result.metadata["decision_evidence"]["score_evidence"]
    assert "top1_topk_mean_gap" in result.metadata["decision_evidence"]["score_evidence"]
    assert "reranked_margin" in result.metadata["decision_evidence"]["score_evidence"]


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
    assert result.metadata["decision_reason"] == "below_threshold"
    assert result.metadata["reject_reason"] == "below_threshold"
    assert result.metadata["decision_evidence"]["summary"]["decision_reason"] == "below_threshold"
    assert result.metadata["decision_evidence"]["score_evidence"]["effective_threshold_value"] == 0.95


def test_build_decision_relaxes_threshold_for_single_sample_single_speaker() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profile = _profile("alice", [0.32, np.sqrt(1.0 - 0.32**2)])
    profile.metadata["sample_count"] = 1
    result = build_decision(
        query,
        [profile],
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
    )
    assert result.decision == DecisionLabel.ACCEPT
    assert result.final_label == "alice"
    assert result.threshold_value == 0.311
    assert result.metadata["requested_threshold_value"] == 0.35


def test_build_decision_rejects_borderline_multi_speaker_when_margin_is_too_small() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profiles = [
        _profile("alice", [1.0, 0.0]),
        _profile("bob", [0.995, np.sqrt(1.0 - 0.995**2)]),
    ]
    result = build_decision(
        query,
        profiles,
        threshold_value=0.98,
        scoring_strategy=ScoringStrategy.CENTER,
    )
    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"
    assert result.metadata["decision_reason"] == "low_margin"
    assert result.metadata["reject_reason"] == "low_margin"


def test_build_decision_relaxes_multi_speaker_threshold_for_longer_queries() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profiles = [
        _profile("alice", [0.32, np.sqrt(1.0 - 0.32**2)]),
        _profile("bob", [0.0, 1.0]),
    ]
    result = build_decision(
        query,
        profiles,
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=2.0,
    )
    assert result.decision == DecisionLabel.ACCEPT
    assert result.final_label == "alice"
    assert result.threshold_value == 0.31


def test_build_decision_accepts_strong_calibrated_candidate_below_raw_threshold() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    alice = _profile("alice", [0.2555, np.sqrt(1.0 - 0.2555**2)])
    bob = _profile("bob", [0.12, np.sqrt(1.0 - 0.12**2)])
    carol = _profile("carol", [-0.15, np.sqrt(1.0 - 0.15**2)])
    for profile, mean, std in (
        (alice, 0.05, 0.12),
        (bob, 0.02, 0.1),
        (carol, 0.01, 0.1),
    ):
        profile.impostor_score_mean = mean
        profile.impostor_score_std = std
        profile.open_set_floor = 0.18
        profile.metadata["impostor_score_mean"] = mean
        profile.metadata["impostor_score_std"] = std
        profile.metadata["open_set_floor"] = 0.18
        profile.metadata["calibration_status"] = "statistical"
        profile.metadata["calibration_scale"] = 1.0
        profile.metadata["calibration_bias"] = 0.0

    result = build_decision(
        query,
        [alice, bob, carol],
        threshold_value=0.31,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=2.0,
        query_quality_score=0.8,
    )

    assert result.decision == DecisionLabel.ACCEPT
    assert result.final_label == "alice"
    assert result.metadata["accept_score_space"] == "calibrated"


def test_build_decision_accepts_short_calibrated_leader_with_large_raw_deficit() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    huangbaichao = _profile("huangbaichao", [0.2865, np.sqrt(1.0 - 0.2865**2)])
    passerby_a = _profile("passerby_a", [0.1812, np.sqrt(1.0 - 0.1812**2)])
    chenrong = _profile("chenrong", [0.0700, np.sqrt(1.0 - 0.0700**2)])
    for profile, mean, std, floor in (
        (huangbaichao, 0.01, 0.05, 0.18),
        (passerby_a, 0.03, 0.11, 0.18),
        (chenrong, 0.02, 0.1, 0.16),
    ):
        profile.impostor_score_mean = mean
        profile.impostor_score_std = std
        profile.open_set_floor = floor
        profile.metadata["impostor_score_mean"] = mean
        profile.metadata["impostor_score_std"] = std
        profile.metadata["open_set_floor"] = floor
        profile.metadata["calibration_status"] = "statistical"
        profile.metadata["calibration_scale"] = 1.0
        profile.metadata["calibration_bias"] = 0.0

    result = build_decision(
        query,
        [huangbaichao, passerby_a, chenrong],
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=0.743,
        query_quality_score=0.291,
    )

    assert result.decision == DecisionLabel.ACCEPT
    assert result.final_label == "huangbaichao"
    assert result.metadata["accept_reason"] == "short_calibrated_leader"
    assert result.metadata["accept_score_space"] == "calibrated"


def test_build_decision_accepts_calibrated_candidate_below_profile_floor_when_guard_allows() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    hongjiaying = _profile("hongjiaying", [0.2850, np.sqrt(1.0 - 0.2850**2)])
    passerby_a = _profile("passerby_a", [0.1780, np.sqrt(1.0 - 0.1780**2)])
    wangyingrou = _profile("wangyingrou", [0.2029, np.sqrt(1.0 - 0.2029**2)])
    for profile, mean, std, floor in (
        (hongjiaying, 0.01, 0.05, 0.2945),
        (passerby_a, 0.03, 0.11, 0.18),
        (wangyingrou, 0.04, 0.11, 0.2),
    ):
        profile.impostor_score_mean = mean
        profile.impostor_score_std = std
        profile.open_set_floor = floor
        profile.risk_level = ProfileRiskLevel.HIGH if profile is hongjiaying else ProfileRiskLevel.MEDIUM
        profile.metadata["impostor_score_mean"] = mean
        profile.metadata["impostor_score_std"] = std
        profile.metadata["open_set_floor"] = floor
        profile.metadata["calibration_status"] = "statistical"
        profile.metadata["calibration_scale"] = 1.0
        profile.metadata["calibration_bias"] = 0.0

    result = build_decision(
        query,
        [hongjiaying, passerby_a, wangyingrou],
        threshold_value=0.41,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=1.063,
        query_quality_score=0.311,
    )

    assert result.decision == DecisionLabel.ACCEPT
    assert result.final_label == "hongjiaying"
    assert result.metadata["accept_score_space"] == "calibrated"


def test_build_decision_keeps_requested_threshold_for_short_multi_speaker_queries() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profiles = [
        _profile("alice", [0.32, np.sqrt(1.0 - 0.32**2)]),
        _profile("bob", [0.0, 1.0]),
    ]
    result = build_decision(
        query,
        profiles,
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=0.6,
    )
    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"
    assert result.threshold_value == 0.35


def test_build_decision_relaxes_margin_for_long_multi_speaker_queries() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profiles = [
        _profile("alice", [0.3148, np.sqrt(1.0 - 0.3148**2)]),
        _profile("bob", [0.2977, np.sqrt(1.0 - 0.2977**2)]),
    ]
    result = build_decision(
        query,
        profiles,
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=1.9,
    )
    assert result.decision == DecisionLabel.ACCEPT
    assert result.final_label == "alice"
    assert result.metadata["effective_margin_threshold"] == 0.015


def test_build_decision_rejects_crowded_high_score_cluster() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profiles = [
        _profile("guoweicheng", [0.5653, np.sqrt(1.0 - 0.5653**2)]),
        _profile("laobai", [0.4489, np.sqrt(1.0 - 0.4489**2)]),
        _profile("zhouguorong", [0.4408, np.sqrt(1.0 - 0.4408**2)]),
    ]
    result = build_decision(
        query,
        profiles,
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=3.5,
        query_quality_score=0.8,
    )
    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"
    assert result.metadata["decision_reason"] == "crowded_high_score_cluster"
    assert result.metadata["reject_reason"] == "crowded_high_score_cluster"
    assert result.metadata["top3_speaker_id"] == "zhouguorong"


def test_build_decision_accepts_two_candidate_runoff() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profiles = [
        _profile("xiaoli", [0.2938, np.sqrt(1.0 - 0.2938**2)]),
        _profile("renzong", [0.2935, np.sqrt(1.0 - 0.2935**2)]),
        _profile("xiaoyu", [0.1512, np.sqrt(1.0 - 0.1512**2)]),
    ]
    result = build_decision(
        query,
        profiles,
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=2.345,
        query_quality_score=0.6689,
    )
    assert result.decision == DecisionLabel.ACCEPT
    assert result.final_label == "xiaoli"
    assert result.metadata["decision_reason"] == "two_candidate_runoff"
    assert result.metadata["accept_reason"] == "two_candidate_runoff"
    assert result.metadata["reject_reason"] == ""
    assert result.metadata["decision_evidence"]["candidate_evidence"]["top2_speaker_id"] == "renzong"
    assert result.metadata["decision_evidence"]["candidate_evidence"]["top3_speaker_id"] == "xiaoyu"


def test_build_decision_keeps_rejecting_non_runoff_pattern() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profiles = [
        _profile("xiaoli", [0.2923, np.sqrt(1.0 - 0.2923**2)]),
        _profile("renzong", [0.2164, np.sqrt(1.0 - 0.2164**2)]),
        _profile("huangbaichao", [0.1769, np.sqrt(1.0 - 0.1769**2)]),
    ]
    result = build_decision(
        query,
        profiles,
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=3.088,
        query_quality_score=0.8320,
    )
    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"
    assert result.metadata["reject_reason"] == "below_threshold"


def test_build_decision_accepts_strong_leader_just_below_threshold() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profiles = [
        _profile("chenrong", [0.2928, np.sqrt(1.0 - 0.2928**2)]),
        _profile("xiaoli", [0.1799, np.sqrt(1.0 - 0.1799**2)]),
        _profile("renzong", [0.1700, np.sqrt(1.0 - 0.1700**2)]),
    ]
    result = build_decision(
        query,
        profiles,
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=1.164,
        query_quality_score=0.3490,
    )
    assert result.decision == DecisionLabel.ACCEPT
    assert result.final_label == "chenrong"
    assert result.metadata["decision_reason"] == "strong_leader_below_threshold"
    assert result.metadata["accept_reason"] == "strong_leader_below_threshold"


def test_build_decision_keeps_rejecting_long_unknown_like_strong_leader() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profiles = [
        _profile("xiaoli", [0.2720, np.sqrt(1.0 - 0.2720**2)]),
        _profile("xiaoming", [0.1019, np.sqrt(1.0 - 0.1019**2)]),
        _profile("renzong", [0.0958, np.sqrt(1.0 - 0.0958**2)]),
    ]
    result = build_decision(
        query,
        profiles,
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=3.206,
        query_quality_score=0.7683,
    )
    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"
    assert result.metadata["reject_reason"] == "below_threshold"


def test_build_decision_marks_large_raw_deficit_when_calibrated_override_is_too_risky() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    alice = _profile("alice", [0.1950, np.sqrt(1.0 - 0.1950**2)])
    bob = _profile("bob", [0.1260, np.sqrt(1.0 - 0.1260**2)])
    carol = _profile("carol", [0.1180, np.sqrt(1.0 - 0.1180**2)])
    for profile, mean, std, floor in (
        (alice, 0.02, 0.05, 0.18),
        (bob, 0.01, 0.11, 0.15),
        (carol, 0.01, 0.11, 0.15),
    ):
        profile.impostor_score_mean = mean
        profile.impostor_score_std = std
        profile.open_set_floor = floor
        profile.metadata["impostor_score_mean"] = mean
        profile.metadata["impostor_score_std"] = std
        profile.metadata["open_set_floor"] = floor
        profile.metadata["calibration_status"] = "statistical"
        profile.metadata["calibration_scale"] = 1.0
        profile.metadata["calibration_bias"] = 0.0
    alice.calibrated_threshold = 0.4
    alice.metadata["calibration_status"] = "heldout_calibrated"
    alice.metadata["calibration_type"] = "linear_heldout"

    result = build_decision(
        query,
        [alice, bob, carol],
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=2.4,
        query_quality_score=0.86,
    )

    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"
    assert result.metadata["reject_reason"] == "calibrated_override_raw_deficit"


def test_build_decision_respects_heldout_calibrated_threshold() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    profiles = [
        _profile("alice", [1.0, 0.0]),
        _profile("bob", [0.2, np.sqrt(1.0 - 0.2**2)]),
        _profile("carol", [-0.2, np.sqrt(1.0 - 0.2**2)]),
    ]
    for profile in profiles:
        profile.impostor_score_mean = 0.1
        profile.impostor_score_std = 0.1
        profile.open_set_floor = 0.15
        profile.metadata["impostor_score_mean"] = 0.1
        profile.metadata["impostor_score_std"] = 0.1
        profile.metadata["open_set_floor"] = 0.15
        profile.metadata["calibration_status"] = "statistical"
        profile.metadata["calibration_type"] = "profile_impostor_stats"
        profile.metadata["calibration_scale"] = 1.0
        profile.metadata["calibration_bias"] = 0.0
    alice = profiles[0]
    alice.calibrated_threshold = 0.45
    alice.metadata["calibration_status"] = "heldout_calibrated"
    alice.metadata["calibration_type"] = "linear_heldout"
    alice.metadata["calibration_scale"] = 0.05
    alice.metadata["calibration_bias"] = 0.0

    result = build_decision(
        query,
        profiles,
        threshold_value=0.5,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=2.0,
        query_quality_score=0.9,
    )

    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"
    assert result.metadata["decision_reason"] == "open_set_gate"
    assert result.metadata["calibration_status"] == "heldout_calibrated"
    assert result.metadata["decision_evidence"]["profile_evidence"]["calibration_status"] == "heldout_calibrated"
    assert result.metadata["decision_evidence"]["score_evidence"]["calibrated_score"] < 0.45


def test_build_decision_rejects_long_low_floor_high_risk_candidate_with_weak_raw_support() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    xiaoli = _profile("xiaoli", [0.3486, np.sqrt(1.0 - 0.3486**2)])
    renzong = _profile("renzong", [0.0736, np.sqrt(1.0 - 0.0736**2)])
    xiaoyu = _profile("xiaoyu", [0.0610, np.sqrt(1.0 - 0.0610**2)])
    for profile in (xiaoli, renzong, xiaoyu):
        profile.impostor_score_mean = 0.05
        profile.impostor_score_std = 0.12
        profile.open_set_floor = 0.3243 if profile is xiaoli else 0.18
        profile.risk_level = ProfileRiskLevel.HIGH if profile is xiaoli else ProfileRiskLevel.MEDIUM
        profile.metadata["impostor_score_mean"] = profile.impostor_score_mean
        profile.metadata["impostor_score_std"] = profile.impostor_score_std
        profile.metadata["open_set_floor"] = profile.open_set_floor
        profile.metadata["calibration_status"] = "statistical"
        profile.metadata["calibration_scale"] = 1.0
        profile.metadata["calibration_bias"] = 0.0

    result = build_decision(
        query,
        [xiaoli, renzong, xiaoyu],
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=3.206,
        query_quality_score=0.768,
    )

    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"
    assert result.metadata["reject_reason"] == "high_risk_profile_guard"


def test_build_decision_rejects_high_floor_high_risk_candidate_with_thin_margin() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    wangyingrou = _profile("wangyingrou", [0.3597, np.sqrt(1.0 - 0.3597**2)])
    renzong = _profile("renzong", [0.2626, np.sqrt(1.0 - 0.2626**2)])
    zhouguorong = _profile("zhouguorong", [0.2007, np.sqrt(1.0 - 0.2007**2)])
    for profile in (wangyingrou, renzong, zhouguorong):
        profile.impostor_score_mean = 0.09
        profile.impostor_score_std = 0.14
        profile.open_set_floor = 0.3999 if profile is wangyingrou else 0.2
        profile.risk_level = (
            ProfileRiskLevel.HIGH if profile is wangyingrou else ProfileRiskLevel.MEDIUM
        )
        profile.metadata["impostor_score_mean"] = profile.impostor_score_mean
        profile.metadata["impostor_score_std"] = profile.impostor_score_std
        profile.metadata["open_set_floor"] = profile.open_set_floor
        profile.metadata["calibration_status"] = "statistical"
        profile.metadata["calibration_scale"] = 1.0
        profile.metadata["calibration_bias"] = 0.0

    result = build_decision(
        query,
        [wangyingrou, renzong, zhouguorong],
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=3.493,
        query_quality_score=0.792,
    )

    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"
    assert result.metadata["reject_reason"] == "high_risk_profile_guard"


def test_build_decision_rejects_high_floor_high_risk_candidate_with_large_floor_gap() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    wangyingrou = _profile("wangyingrou", [0.2862, np.sqrt(1.0 - 0.2862**2)])
    renzong = _profile("renzong", [0.1290, np.sqrt(1.0 - 0.1290**2)])
    hongjiaying = _profile("hongjiaying", [0.0537, np.sqrt(1.0 - 0.0537**2)])
    for profile in (wangyingrou, renzong, hongjiaying):
        profile.impostor_score_mean = 0.09
        profile.impostor_score_std = 0.14
        profile.open_set_floor = 0.3999 if profile is wangyingrou else 0.2
        profile.risk_level = (
            ProfileRiskLevel.HIGH if profile is wangyingrou else ProfileRiskLevel.MEDIUM
        )
        profile.metadata["impostor_score_mean"] = profile.impostor_score_mean
        profile.metadata["impostor_score_std"] = profile.impostor_score_std
        profile.metadata["open_set_floor"] = profile.open_set_floor
        profile.metadata["calibration_status"] = "statistical"
        profile.metadata["calibration_scale"] = 1.0
        profile.metadata["calibration_bias"] = 0.0

    result = build_decision(
        query,
        [wangyingrou, renzong, hongjiaying],
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=22.57,
        query_quality_score=0.99,
    )

    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"
    assert result.metadata["reject_reason"] == "high_risk_profile_guard"


def test_build_decision_rejects_short_low_quality_high_risk_candidate_with_weak_raw_support() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    xiaoli = _profile("xiaoli", [0.3058, np.sqrt(1.0 - 0.3058**2)])
    huangbaichao = _profile("huangbaichao", [0.2476, np.sqrt(1.0 - 0.2476**2)])
    renzong = _profile("renzong", [0.2481, np.sqrt(1.0 - 0.2481**2)])
    for profile in (xiaoli, huangbaichao, renzong):
        profile.impostor_score_mean = 0.05
        profile.impostor_score_std = 0.12
        profile.open_set_floor = 0.3243 if profile is xiaoli else 0.18
        profile.risk_level = ProfileRiskLevel.HIGH if profile is xiaoli else ProfileRiskLevel.MEDIUM
        profile.metadata["impostor_score_mean"] = profile.impostor_score_mean
        profile.metadata["impostor_score_std"] = profile.impostor_score_std
        profile.metadata["open_set_floor"] = profile.open_set_floor
        profile.metadata["calibration_status"] = "statistical"
        profile.metadata["calibration_scale"] = 1.0
        profile.metadata["calibration_bias"] = 0.0

    result = build_decision(
        query,
        [xiaoli, huangbaichao, renzong],
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=0.945,
        query_quality_score=0.278,
    )

    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"
    assert result.metadata["reject_reason"] == "high_risk_profile_guard"


def test_gate_review_override_accepts_only_raw_conflict_review_like_case() -> None:
    assert _should_accept_for_gate_review_override(
        best_score=0.33889320492744446,
        top2_score=0.3711903691291809,
        calibrated_score=1.8911685397981468,
        effective_threshold=0.31,
        effective_calibrated_threshold=1.1635975508166576,
        query_duration_sec=2.481,
        query_quality_score=0.7488,
        reranked_margin=0.00794816149029598,
    )


def test_gate_review_override_rejects_normal_accept_like_case() -> None:
    assert not _should_accept_for_gate_review_override(
        best_score=0.3486,
        top2_score=0.0736,
        calibrated_score=2.12,
        effective_threshold=0.31,
        effective_calibrated_threshold=1.15,
        query_duration_sec=3.206,
        query_quality_score=0.768,
        reranked_margin=0.3496,
    )


def test_build_decision_rejects_mid_duration_high_floor_high_risk_candidate_with_thin_margin() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    wangyingrou = _profile("wangyingrou", [0.2869, np.sqrt(1.0 - 0.2869**2)])
    renzong = _profile("renzong", [0.2276, np.sqrt(1.0 - 0.2276**2)])
    xiaoli = _profile("xiaoli", [0.1638, np.sqrt(1.0 - 0.1638**2)])
    for profile, mean, std, floor, risk_level in (
        (wangyingrou, 0.09, 0.14, 0.3999, ProfileRiskLevel.HIGH),
        (renzong, 0.05, 0.12, 0.18, ProfileRiskLevel.MEDIUM),
        (xiaoli, 0.05, 0.12, 0.3243, ProfileRiskLevel.HIGH),
    ):
        profile.impostor_score_mean = mean
        profile.impostor_score_std = std
        profile.open_set_floor = floor
        profile.risk_level = risk_level
        profile.metadata["impostor_score_mean"] = mean
        profile.metadata["impostor_score_std"] = std
        profile.metadata["open_set_floor"] = floor
        profile.metadata["calibration_status"] = "statistical"
        profile.metadata["calibration_scale"] = 1.0
        profile.metadata["calibration_bias"] = 0.0

    result = build_decision(
        query,
        [wangyingrou, renzong, xiaoli],
        threshold_value=0.35,
        scoring_strategy=ScoringStrategy.CENTER,
        query_duration_sec=2.565,
        query_quality_score=0.6679,
    )

    assert result.decision == DecisionLabel.REJECT
    assert result.final_label == "UNKNOWN"
    assert result.metadata["reject_reason"] == "high_risk_profile_guard"


def test_open_set_gate_returns_review_for_unstable_reranked_lead() -> None:
    decision = evaluate_open_set_gate(
        OpenSetGateEvidence(
            raw_score=0.68,
            z_norm_score=1.1,
            adaptive_s_norm_score=1.0,
            calibrated_score=1.0,
            cohort_relative_score=0.55,
            open_set_margin=0.04,
            top1_topk_mean_gap=0.03,
            reranked_margin=0.004,
            member_consistency_score=0.66,
            effective_threshold=0.35,
            open_set_floor=0.2,
            calibrated_threshold=None,
            query_duration_sec=2.2,
            query_quality_score=0.9,
            profile_risk_level=ProfileRiskLevel.MEDIUM,
        )
    )

    assert decision == OpenSetDecision.REVIEW


def test_open_set_gate_rejects_when_cohort_relative_evidence_is_too_weak() -> None:
    decision = evaluate_open_set_gate(
        OpenSetGateEvidence(
            raw_score=0.7,
            z_norm_score=0.9,
            adaptive_s_norm_score=0.8,
            calibrated_score=0.8,
            cohort_relative_score=0.05,
            open_set_margin=0.03,
            top1_topk_mean_gap=0.02,
            reranked_margin=0.03,
            member_consistency_score=0.67,
            effective_threshold=0.35,
            open_set_floor=0.2,
            calibrated_threshold=None,
            query_duration_sec=2.0,
            query_quality_score=0.8,
            profile_risk_level=ProfileRiskLevel.MEDIUM,
        )
    )

    assert decision == OpenSetDecision.REJECT_UNKNOWN


def test_open_set_gate_allows_strong_calibrated_evidence_to_override_small_floor_gap() -> None:
    decision = evaluate_open_set_gate(
        OpenSetGateEvidence(
            raw_score=0.3264,
            z_norm_score=1.8,
            adaptive_s_norm_score=1.7,
            calibrated_score=1.7,
            cohort_relative_score=0.8,
            open_set_margin=0.05,
            top1_topk_mean_gap=0.03,
            reranked_margin=0.02,
            member_consistency_score=0.325,
            effective_threshold=0.31,
            open_set_floor=0.3499,
            calibrated_threshold=None,
            query_duration_sec=2.1,
            query_quality_score=0.8,
            profile_risk_level=ProfileRiskLevel.HIGH,
        )
    )

    assert decision == OpenSetDecision.ACCEPT


def test_open_set_gate_rejects_long_high_risk_low_floor_when_raw_evidence_is_too_weak() -> None:
    decision = evaluate_open_set_gate(
        OpenSetGateEvidence(
            raw_score=0.3549,
            z_norm_score=4.02,
            adaptive_s_norm_score=3.52,
            calibrated_score=3.52,
            cohort_relative_score=3.02,
            open_set_margin=2.06,
            top1_topk_mean_gap=2.77,
            reranked_margin=0.56,
            member_consistency_score=0.3549,
            effective_threshold=0.31,
            open_set_floor=0.2239,
            calibrated_threshold=0.2239,
            query_duration_sec=16.7,
            query_quality_score=0.88,
            profile_risk_level=ProfileRiskLevel.HIGH,
        )
    )

    assert decision == OpenSetDecision.REJECT_UNKNOWN


def test_normalize_profile_score_avoids_exploding_when_only_one_cohort_score_exists() -> None:
    profile = _profile("alice", [1.0, 0.0])
    profile.impostor_score_mean = 0.15
    profile.impostor_score_std = 0.12

    z_norm_score, adaptive_s_norm_score, calibrated_score, cohort_relative_score = (
        normalize_profile_score(
            raw_score=0.48,
            profile=profile,
            cohort_scores=[0.479999],
        )
    )

    assert abs(z_norm_score) < 8.1
    assert adaptive_s_norm_score == z_norm_score
    assert calibrated_score == adaptive_s_norm_score
    assert cohort_relative_score == calibrated_score


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
