from pathlib import Path

from app.benchmark.models import BenchmarkItemResult, BenchmarkRunConfig, BenchmarkRunResult
from app.common.errors import InvalidManifestError, ValidationIssue
from app.reporting.json_summary import write_benchmark_json_summary
from app.reporting.markdown import write_benchmark_markdown
from app.reporting.tsv import write_benchmark_tsv
from app.scoring.models import ScoringStrategy


def _build_result() -> BenchmarkRunResult:
    config = BenchmarkRunConfig(
        run_name="report-smoke",
        dataset_name="holdout-demo",
        dataset_version="v1",
        backend_name="dummy",
        scoring_strategy=ScoringStrategy.CENTER,
        threshold_value=0.35,
        dataset_role="external_holdout",
    )
    items = [
        BenchmarkItemResult(
            clip_id="known-1",
            audio_path=Path("/tmp/known-1.wav"),
            truth_label="alice",
            expected_label="alice",
            final_label="alice",
            decision="ACCEPT",
            best_score=0.81,
            threshold_value=0.35,
            latency_ms=12.5,
            is_correct=True,
            evaluation_group="external_known",
            score_breakdown={"alice": 0.81},
            metadata={
                "profile_risk_level": "low",
                "accept_reason": "normal_accept",
                "decision_reason": "normal_accept",
                "calibration_status": "heldout_calibrated",
                "calibration_type": "linear_heldout",
                "decision_evidence": {
                    "schema_version": "v2",
                    "summary": {"decision_reason": "normal_accept"},
                    "candidate_evidence": {"top1_speaker_id": "alice"},
                    "score_evidence": {"top1_score": 0.81},
                    "profile_evidence": {"calibration_status": "heldout_calibrated"},
                },
            },
        ),
        BenchmarkItemResult(
            clip_id="unknown-1",
            audio_path=Path("/tmp/unknown-1.wav"),
            truth_label="eve",
            expected_label="UNKNOWN",
            final_label="UNKNOWN",
            decision="REJECT",
            best_score=0.22,
            threshold_value=0.35,
            latency_ms=10.0,
            is_correct=True,
            evaluation_group="external_unknown",
            score_breakdown={"alice": 0.22},
            metadata={
                "profile_risk_level": "medium",
                "reject_reason": "below_threshold",
                "decision_reason": "below_threshold",
                "calibration_status": "statistical",
                "calibration_type": "profile_impostor_stats",
                "decision_evidence": {
                    "schema_version": "v2",
                    "summary": {"decision_reason": "below_threshold"},
                    "score_evidence": {"effective_threshold_value": 0.35},
                    "profile_evidence": {"calibration_status": "statistical"},
                },
            },
        ),
        BenchmarkItemResult(
            clip_id="unknown-2",
            audio_path=Path("/tmp/unknown-2.wav"),
            truth_label="mallory",
            expected_label="UNKNOWN",
            final_label="alice",
            decision="ACCEPT",
            best_score=0.57,
            threshold_value=0.35,
            latency_ms=13.2,
            is_correct=False,
            evaluation_group="external_unknown",
            score_breakdown={"alice": 0.57},
            metadata={
                "profile_risk_level": "high",
                "accept_reason": "two_candidate_runoff",
                "decision_reason": "two_candidate_runoff",
                "calibration_status": "statistical",
                "calibration_type": "profile_impostor_stats",
                "decision_evidence": {
                    "schema_version": "v2",
                    "summary": {"decision_reason": "two_candidate_runoff"},
                    "candidate_evidence": {"top2_speaker_id": "bob"},
                    "profile_evidence": {"calibration_status": "statistical"},
                },
            },
        ),
        BenchmarkItemResult(
            clip_id="internal-1",
            audio_path=Path("/tmp/internal-1.wav"),
            truth_label="bob",
            expected_label="bob",
            final_label="UNKNOWN",
            decision="REVIEW",
            best_score=0.41,
            threshold_value=0.35,
            latency_ms=11.8,
            is_correct=False,
            evaluation_group="internal_known",
            score_breakdown={"bob": 0.41},
            metadata={
                "profile_risk_level": "medium",
                "reject_reason": "review",
                "decision_reason": "review",
                "calibration_status": "statistical",
                "calibration_type": "profile_impostor_stats",
                "decision_evidence": {
                    "schema_version": "v2",
                    "summary": {"decision_reason": "review"},
                    "gate_evidence": {"open_set_gate_decision": "review"},
                    "profile_evidence": {"calibration_status": "statistical"},
                },
            },
        ),
    ]
    return BenchmarkRunResult(config=config, items=items)


def test_benchmark_run_result_exposes_open_set_summary_metrics() -> None:
    result = _build_result()

    assert result.total == 4
    assert result.correct == 2
    assert result.review_count == 1
    assert result.average_latency_ms > 0.0
    assert result.max_latency_ms == 13.2
    assert result.false_accept_count == 1
    assert result.false_reject_count == 1
    assert result.external_known_total == 1
    assert result.external_known_top1_accuracy == 1.0
    assert result.external_unknown_total == 2
    assert result.external_unknown_reject_rate == 0.5
    assert result.high_risk_false_accept_count == 1
    assert result.accept_reason_counts == {"normal_accept": 1, "two_candidate_runoff": 1}
    assert result.reject_reason_counts == {"below_threshold": 1}
    assert result.review_reason_counts == {"review": 1}
    assert result.calibration_status_counts == {"heldout_calibrated": 1, "statistical": 3}
    assert result.heldout_calibrated_count == 1
    assert result.decision_reason_stats["normal_accept"]["correct"] == 1
    assert result.decision_reason_stats["two_candidate_runoff"]["incorrect"] == 1


def test_reporting_outputs_include_new_open_set_fields(tmp_path: Path) -> None:
    result = _build_result()

    markdown = write_benchmark_markdown(result, tmp_path / "report.md").path.read_text(
        encoding="utf-8"
    )
    summary_json = write_benchmark_json_summary(
        result,
        tmp_path / "summary.json",
    ).path.read_text(encoding="utf-8")
    tsv = write_benchmark_tsv(result, tmp_path / "summary.tsv").path.read_text(
        encoding="utf-8"
    )

    assert "复核数" in markdown
    assert "数据集角色" in markdown
    assert "平均时延" in markdown
    assert "外部已知 Top1 准确率" in markdown
    assert "接受原因分布" in markdown
    assert "\"dataset_role\": \"external_holdout\"" in summary_json
    assert "\"latency_ms\": 12.5" in summary_json
    assert "\"evaluation_group\": \"external_known\"" in summary_json
    assert "\"review_count\": 1" in summary_json
    assert "\"decision_reason\": \"normal_accept\"" in summary_json
    assert "\"decision_evidence\"" in summary_json
    assert "\"schema_version\": \"v2\"" in summary_json
    assert "\"candidate_evidence\": {" in summary_json
    assert "\"calibration_status\": \"heldout_calibrated\"" in summary_json
    assert "\"accept_reason_counts\"" in summary_json
    assert "\"calibration_status_counts\"" in summary_json
    assert "\"decision_reason_stats\"" in summary_json
    assert "\"two_candidate_runoff\": 1" in summary_json
    assert "判决原因" in markdown
    assert "校准状态分布" in markdown
    assert "判决证据" in markdown
    assert "判决理由统计" in markdown
    assert "评测分组" in tsv
    assert "判决原因" in tsv
    assert "校准状态" in tsv
    assert "判决证据" in tsv
    assert "时延(ms)" in tsv
    assert "判决元数据" in tsv


def test_invalid_manifest_error_supports_machine_readable_serialization() -> None:
    error = InvalidManifestError(
        [
            ValidationIssue(
                code="invalid_trial_role",
                row_number=3,
                column_name="trial_role",
                message="manifest.csv: 第 3 行的 `trial_role` 非法：`bad_role`。",
            )
        ]
    )

    payload = error.to_dict()
    encoded = error.to_json()

    assert payload["error_type"] == "InvalidManifestError"
    assert payload["issues"][0]["code"] == "invalid_trial_role"
    assert payload["issues"][0]["row_number"] == 3
    assert payload["issues"][0]["column_name"] == "trial_role"
    assert "\"error_type\": \"InvalidManifestError\"" in encoded
    assert "\"code\": \"invalid_trial_role\"" in encoded
