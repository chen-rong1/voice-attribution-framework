from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _load_rows(tsv_path: Path) -> list[dict[str, str]]:
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def _parse_metadata(row: dict[str, str]) -> dict:
    raw = row.get("判决元数据", "") or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _parse_evidence(row: dict[str, str]) -> dict:
    raw = row.get("判决证据", "") or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _extract_case(row: dict[str, str], dataset_name: str) -> dict:
    metadata = _parse_metadata(row)
    evidence = _parse_evidence(row)
    candidate = evidence.get("candidate_evidence", {})
    query = evidence.get("query_evidence", {})
    score = evidence.get("score_evidence", {})

    return {
        "dataset": dataset_name,
        "clip_id": row.get("片段编号", ""),
        "expected": row.get("预期标签", ""),
        "final": row.get("最终标签", ""),
        "decision": row.get("决策", ""),
        "decision_reason": row.get("判决原因", ""),
        "correct": row.get("是否正确", ""),
        "accept_reason": metadata.get("accept_reason", ""),
        "accept_score_space": metadata.get("accept_score_space", ""),
        "top1_speaker_id": metadata.get("top1_speaker_id", candidate.get("top1_speaker_id", "")),
        "top2_speaker_id": metadata.get("top2_speaker_id", candidate.get("top2_speaker_id", "")),
        "top3_speaker_id": metadata.get("top3_speaker_id", candidate.get("top3_speaker_id", "")),
        "top1_raw_score": metadata.get("top1_raw_score", score.get("top1_raw_score")),
        "top2_score": metadata.get("top2_score", candidate.get("top2_score")),
        "top3_score": metadata.get("top3_score", candidate.get("top3_score")),
        "calibrated_score": metadata.get("calibrated_score", score.get("calibrated_score")),
        "top2_calibrated_score": metadata.get(
            "top2_calibrated_score", candidate.get("top2_calibrated_score")
        ),
        "margin": metadata.get("margin", score.get("margin")),
        "reranked_margin": metadata.get("reranked_margin", score.get("reranked_margin")),
        "query_duration_sec": metadata.get(
            "query_duration_sec", query.get("query_duration_sec")
        ),
        "query_quality_score": metadata.get(
            "query_quality_score", query.get("query_quality_score")
        ),
        "profile_open_set_floor": metadata.get(
            "profile_open_set_floor", evidence.get("profile_evidence", {}).get("profile_open_set_floor")
        ),
        "effective_threshold_value": metadata.get(
            "effective_threshold_value", score.get("effective_threshold_value")
        ),
        "effective_calibrated_threshold_value": metadata.get(
            "effective_calibrated_threshold_value",
            score.get("effective_calibrated_threshold_value"),
        ),
    }


def _matches_pair(row: dict[str, str], speaker_a: str, speaker_b: str) -> bool:
    expected = row.get("预期标签", "")
    final_label = row.get("最终标签", "")
    metadata = _parse_metadata(row)
    top1 = metadata.get("top1_speaker_id", "")
    top2 = metadata.get("top2_speaker_id", "")
    top3 = metadata.get("top3_speaker_id", "")
    pair = {speaker_a, speaker_b}

    labels = {expected, final_label, top1, top2, top3}
    return pair.issubset(labels) or (
        expected in pair and (final_label in pair or top1 in pair or top2 in pair or top3 in pair)
    )


def _rank_bucket(case: dict, speaker_a: str, speaker_b: str) -> str:
    expected = case["expected"]
    for key in ("top1_speaker_id", "top2_speaker_id", "top3_speaker_id"):
        if case.get(key) == expected:
            return key.replace("_speaker_id", "")
    if expected in {speaker_a, speaker_b}:
        return "outside_top3"
    return "not_target_pair"


def _format_case(case: dict) -> dict:
    keys = [
        "dataset",
        "clip_id",
        "expected",
        "final",
        "decision",
        "decision_reason",
        "accept_reason",
        "accept_score_space",
        "top1_speaker_id",
        "top2_speaker_id",
        "top3_speaker_id",
        "top1_raw_score",
        "top2_score",
        "top3_score",
        "calibrated_score",
        "top2_calibrated_score",
        "margin",
        "reranked_margin",
        "query_duration_sec",
        "query_quality_score",
        "profile_open_set_floor",
        "effective_threshold_value",
        "effective_calibrated_threshold_value",
        "correct",
    ]
    return {key: case.get(key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit confusion pairs from benchmark TSV reports.")
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=Path(
            "/Users/工作/声纹识别/voice-benchmark/api_service/benchmark_outputs/once/"
            "framework_regression_suite_max035_branchfix_external_known_v5b"
        ),
        help="Regression suite output directory.",
    )
    parser.add_argument("--speaker-a", required=True, help="First speaker in the confusion pair.")
    parser.add_argument("--speaker-b", required=True, help="Second speaker in the confusion pair.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["full_147", "liaoning0222", "stress_40"],
        help="Datasets to inspect inside the suite directory.",
    )
    args = parser.parse_args()

    matched_cases: list[dict] = []
    for dataset in args.datasets:
        tsv_path = args.suite_dir / dataset / f"{dataset}_测试总表.tsv"
        if not tsv_path.exists():
            continue
        for row in _load_rows(tsv_path):
            if _matches_pair(row, args.speaker_a, args.speaker_b):
                matched_cases.append(_extract_case(row, dataset))

    rank_counter = Counter()
    decision_counter = Counter()
    reason_counter = Counter()
    wrong_accept_cases = []
    wrong_reject_cases = []

    for case in matched_cases:
        rank_counter[_rank_bucket(case, args.speaker_a, args.speaker_b)] += 1
        decision_counter[case["decision"]] += 1
        reason_counter[case["decision_reason"]] += 1

        if case["correct"] == "否" and case["decision"] == "ACCEPT":
            wrong_accept_cases.append(_format_case(case))
        if case["correct"] == "否" and case["decision"] == "REJECT":
            wrong_reject_cases.append(_format_case(case))

    payload = {
        "suite_dir": str(args.suite_dir),
        "speaker_pair": [args.speaker_a, args.speaker_b],
        "datasets": args.datasets,
        "matched_total": len(matched_cases),
        "rank_distribution": dict(rank_counter),
        "decision_distribution": dict(decision_counter),
        "decision_reason_distribution": dict(reason_counter),
        "wrong_accept_cases": wrong_accept_cases,
        "wrong_reject_cases": wrong_reject_cases,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
