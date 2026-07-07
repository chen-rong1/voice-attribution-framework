"""Search the best enrollment clip combination for a business dataset."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

from app.audio.io import load_audio_chunk
from app.benchmark.business import load_business_benchmark_clips
from app.benchmark.filesystem import (
    load_benchmark_clips_from_directory,
    load_enrollments_from_directory,
)
from app.benchmark.models import BenchmarkRunConfig
from app.benchmark.runner import BenchmarkRunner
from app.common.config import load_simple_yaml_map
from app.embedding_backends.models import EmbeddingRequest, EmbeddingResult
from app.profiles.builder import build_speaker_profile
from app.scoring.models import ScoringStrategy
from app.scoring.strategies import build_decision
from app.services.bootstrap import register_default_backends
from app.services.container import FrameworkContainer
from app.services.identification import IdentificationService


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="搜索最优注册样本组合。")
    parser.add_argument("--business-dataset-dir", required=True, type=Path, help="平铺业务集目录")
    parser.add_argument("--speaker-id", required=True, help="目标说话人 ID")
    parser.add_argument("--combination-size", type=int, default=4, help="组合大小")
    parser.add_argument(
        "--scoring-config",
        action="append",
        required=True,
        type=Path,
        help="打分配置文件，可重复传入",
    )
    parser.add_argument("--top-n", type=int, default=10, help="输出前 N 个组合")
    parser.add_argument("--business-truth-tsv", type=Path, help="业务集真值表路径")
    parser.add_argument("--business-pure-list", type=Path, help="业务集纯净测试清单路径")
    parser.add_argument("--strict-enroll-dir", type=Path, help="标准严格集注册目录")
    parser.add_argument("--strict-test-dir", type=Path, help="标准严格集测试目录")
    parser.add_argument("--output-dir", required=True, type=Path, help="输出目录")
    parser.add_argument("--run-name", required=True, help="运行名称")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="项目根目录",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.combination_size <= 0:
        parser.error("--combination-size 必须大于 0")
    if args.top_n <= 0:
        parser.error("--top-n 必须大于 0")
    if (args.strict_enroll_dir is None) != (args.strict_test_dir is None):
        parser.error("--strict-enroll-dir 和 --strict-test-dir 必须同时提供")

    scoring_configs = [load_scoring_preset(path) for path in args.scoring_config]

    container = FrameworkContainer()
    register_default_backends(project_root=args.project_root, registry=container.backend_registry)
    model_config = args.project_root / "configs" / "models" / "default.yaml"
    backend_name = load_simple_yaml_map(model_config).get(
        "bootstrap_backend",
        "wespeaker-ecapa1024-lm-onnx",
    )
    backend = container.backend_registry.get(backend_name)
    backend.load()
    strict_validation = validate_scoring_configs_on_strict_set(
        backend=backend,
        scoring_configs=scoring_configs,
        strict_enroll_dir=args.strict_enroll_dir,
        strict_test_dir=args.strict_test_dir,
    )

    clips = load_business_benchmark_clips(
        args.business_dataset_dir,
        truth_tsv_path=args.business_truth_tsv,
        pure_list_path=args.business_pure_list,
    )
    speaker_clips = [clip for clip in clips if clip.expected_label == args.speaker_id]
    if len(speaker_clips) < args.combination_size:
        parser.error(
            f"说话人 `{args.speaker_id}` 的候选片段不足 {args.combination_size} 条，无法搜索组合"
        )

    clip_embeddings = extract_clip_embeddings(backend=backend, clips=clips)
    combination_rows = search_best_combinations(
        speaker_id=args.speaker_id,
        speaker_clips=speaker_clips,
        all_clips=clips,
        clip_embeddings=clip_embeddings,
        scoring_configs=scoring_configs,
        combination_size=args.combination_size,
        strict_validation=strict_validation,
    )
    top_rows = combination_rows[: min(args.top_n, len(combination_rows))]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_combination_tsv(
        rows=top_rows,
        output_path=args.output_dir / f"{args.run_name}_组合搜索.tsv",
    )
    write_combination_markdown(
        speaker_id=args.speaker_id,
        combination_size=args.combination_size,
        total_candidates=len(speaker_clips),
        rows=top_rows,
        output_path=args.output_dir / f"{args.run_name}_组合搜索报告.md",
    )
    write_combination_json(
        speaker_id=args.speaker_id,
        combination_size=args.combination_size,
        total_candidates=len(speaker_clips),
        rows=top_rows,
        output_path=args.output_dir / f"{args.run_name}_组合搜索摘要.json",
    )

    if top_rows:
        print(
            {
                "speaker_id": args.speaker_id,
                "combination_size": args.combination_size,
                "searched_combinations": combination_count(len(speaker_clips), args.combination_size),
                "best_pack": top_rows[0]["enroll_files"],
                "best_accuracy": top_rows[0]["accuracy"],
                "best_profile": top_rows[0]["config_name"],
                "strict_accuracy": top_rows[0].get("strict_accuracy"),
            }
        )
    return 0


def load_scoring_preset(config_path: Path) -> dict[str, str | float]:
    raw = load_simple_yaml_map(config_path)
    strategy = raw.get("strategy", ScoringStrategy.CENTER.value)
    threshold = float(raw.get("threshold", "0.41"))
    aggregation = raw.get("profile_aggregation_strategy", "center")
    return {
        "config_name": config_path.stem,
        "config_path": str(config_path),
        "strategy": strategy,
        "threshold": threshold,
        "profile_aggregation_strategy": aggregation,
    }


def extract_clip_embeddings(
    *,
    backend,
    clips: list,
) -> dict[str, EmbeddingResult]:
    results: dict[str, EmbeddingResult] = {}
    for index, clip in enumerate(clips, start=1):
        audio_chunk = load_audio_chunk(clip.audio_path)
        results[clip.clip_id] = backend.extract_embedding(
            EmbeddingRequest(sample_id=f"clip-{index}", audio=audio_chunk)
        )
    return results


def validate_scoring_configs_on_strict_set(
    *,
    backend,
    scoring_configs: list[dict[str, str | float]],
    strict_enroll_dir: Path | None,
    strict_test_dir: Path | None,
) -> dict[str, dict[str, float | int | str]]:
    if strict_enroll_dir is None or strict_test_dir is None:
        return {}
    service = IdentificationService(backend)
    runner = BenchmarkRunner(service)
    enrollments = load_enrollments_from_directory(strict_enroll_dir)
    clips = load_benchmark_clips_from_directory(strict_test_dir)
    results: dict[str, dict[str, float | int | str]] = {}
    for config in scoring_configs:
        benchmark = runner.run(
            config=BenchmarkRunConfig(
                run_name=f"strict_validation_{config['config_name']}",
                dataset_name="voice-benchmark-strict",
                dataset_version="eval",
                backend_name=backend.backend_name,
                scoring_strategy=ScoringStrategy(str(config["strategy"])),
                threshold_value=float(config["threshold"]),
                profile_aggregation_strategy=str(config["profile_aggregation_strategy"]),
            ),
            enrollments=enrollments,
            clips=clips,
        )
        results[str(config["config_name"])] = {
            "strict_total": benchmark.total,
            "strict_correct": benchmark.correct,
            "strict_accuracy": benchmark.accuracy,
            "strict_positive_recall": benchmark.positive_recall,
            "strict_unknown_reject_rate": benchmark.unknown_reject_rate,
        }
    return results


def search_best_combinations(
    *,
    speaker_id: str,
    speaker_clips: list,
    all_clips: list,
    clip_embeddings: dict[str, EmbeddingResult],
    scoring_configs: list[dict[str, str | float]],
    combination_size: int,
    strict_validation: dict[str, dict[str, float | int | str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    clip_catalog = {clip.clip_id: clip for clip in all_clips}
    for clip_group in itertools.combinations(speaker_clips, combination_size):
        embedding_results = [clip_embeddings[clip.clip_id] for clip in clip_group]
        enroll_files = [clip.audio_path.name for clip in clip_group]
        for config in scoring_configs:
            profile = build_speaker_profile(
                speaker_id,
                embedding_results,
                aggregation_strategy=str(config["profile_aggregation_strategy"]),
            )
            rows.append(
                {
                    "config_name": config["config_name"],
                    "config_path": config["config_path"],
                    "strategy": config["strategy"],
                    "threshold": float(config["threshold"]),
                    "profile_aggregation_strategy": config["profile_aggregation_strategy"],
                    "enroll_files": enroll_files,
                    **strict_validation.get(str(config["config_name"]), {}),
                    **evaluate_profile_with_clipset(
                        speaker_id=speaker_id,
                        profile=profile,
                        clips_to_score=list(clip_embeddings.keys()),
                        clip_catalog=clip_catalog,
                        clip_embeddings=clip_embeddings,
                        strategy=ScoringStrategy(str(config["strategy"])),
                        threshold=float(config["threshold"]),
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            float(row["accuracy"]),
            float(row.get("strict_accuracy", 0.0)),
            float(row["positive_recall"]),
            float(row["unknown_reject_rate"]),
            float(row["avg_positive_score"]) - float(row["avg_unknown_score"]),
        ),
        reverse=True,
    )
    return rows


def evaluate_profile_with_clipset(
    *,
    speaker_id: str,
    profile,
    clips_to_score: list[str],
    clip_catalog: dict[str, object],
    clip_embeddings: dict[str, EmbeddingResult],
    strategy: ScoringStrategy,
    threshold: float,
) -> dict[str, object]:
    correct = 0
    total = len(clips_to_score)
    positive_total = 0
    positive_correct = 0
    unknown_total = 0
    unknown_correct = 0
    positive_scores: list[float] = []
    unknown_scores: list[float] = []
    false_accepts: list[str] = []
    false_rejects: list[str] = []

    for clip_id in clips_to_score:
        result = clip_embeddings[clip_id]
        clip = clip_catalog[clip_id]
        expected_label = str(clip.expected_label)
        decision = build_decision(
            result.embedding,
            [profile],
            threshold_value=threshold,
            scoring_strategy=strategy,
        )
        final_label = decision.final_label
        score = float(decision.best_score)
        if expected_label == "UNKNOWN":
            unknown_total += 1
            unknown_scores.append(score)
            if final_label == "UNKNOWN":
                unknown_correct += 1
            else:
                false_accepts.append(clip_id)
        else:
            positive_total += 1
            positive_scores.append(score)
            if final_label == speaker_id:
                positive_correct += 1
            else:
                false_rejects.append(clip_id)
        correct += int(final_label == expected_label)

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "positive_total": positive_total,
        "positive_correct": positive_correct,
        "positive_recall": positive_correct / positive_total if positive_total else 0.0,
        "unknown_total": unknown_total,
        "unknown_correct": unknown_correct,
        "unknown_reject_rate": unknown_correct / unknown_total if unknown_total else 0.0,
        "avg_positive_score": average_or_zero(positive_scores),
        "avg_unknown_score": average_or_zero(unknown_scores),
        "false_accept_count": len(false_accepts),
        "false_reject_count": len(false_rejects),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
    }


def combination_count(candidate_count: int, combination_size: int) -> int:
    if combination_size > candidate_count:
        return 0
    numerator = 1
    denominator = 1
    for value in range(combination_size):
        numerator *= candidate_count - value
        denominator *= value + 1
    return numerator // denominator


def average_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def write_combination_tsv(*, rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(
            [
                "配置名",
                "打分策略",
                "阈值",
                "画像聚合策略",
                "注册样本",
                "正确数",
                "总数",
                "准确率",
                "本人召回",
                "非本人拒识率",
                "标准集正确数",
                "标准集总数",
                "标准集准确率",
                "标准集本人召回",
                "标准集拒识率",
                "平均本人分",
                "平均非本人分",
                "误认数",
                "漏认数",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["config_name"],
                    row["strategy"],
                    f"{float(row['threshold']):.4f}",
                    row["profile_aggregation_strategy"],
                    " | ".join(row["enroll_files"]),
                    row["correct"],
                    row["total"],
                    f"{float(row['accuracy']):.2%}",
                    f"{float(row['positive_recall']):.2%}",
                    f"{float(row['unknown_reject_rate']):.2%}",
                    row.get("strict_correct", ""),
                    row.get("strict_total", ""),
                    (
                        f"{float(row['strict_accuracy']):.2%}"
                        if row.get("strict_accuracy") is not None
                        else ""
                    ),
                    (
                        f"{float(row['strict_positive_recall']):.2%}"
                        if row.get("strict_positive_recall") is not None
                        else ""
                    ),
                    (
                        f"{float(row['strict_unknown_reject_rate']):.2%}"
                        if row.get("strict_unknown_reject_rate") is not None
                        else ""
                    ),
                    f"{float(row['avg_positive_score']):.4f}",
                    f"{float(row['avg_unknown_score']):.4f}",
                    row["false_accept_count"],
                    row["false_reject_count"],
                ]
            )


def write_combination_markdown(
    *,
    speaker_id: str,
    combination_size: int,
    total_candidates: int,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    lines = [
        f"# {speaker_id} 注册组合搜索报告",
        "",
        "## 搜索口径",
        "",
        f"- 候选说话人：`{speaker_id}`",
        f"- 候选片段数：`{total_candidates}`",
        f"- 组合大小：`{combination_size}`",
        f"- 穷举组合数：`{combination_count(total_candidates, combination_size)}`",
        "",
        "## Top 结果",
        "",
        "| 排名 | 配置名 | 注册样本 | 业务准确率 | 本人召回 | 非本人拒识率 | 标准集准确率 |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | {row['config_name']} | {' / '.join(row['enroll_files'])} | "
            f"{float(row['accuracy']):.2%} | {float(row['positive_recall']):.2%} | "
            f"{float(row['unknown_reject_rate']):.2%} | "
            f"{(f'{float(row['strict_accuracy']):.2%}' if row.get('strict_accuracy') is not None else '-')} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combination_json(
    *,
    speaker_id: str,
    combination_size: int,
    total_candidates: int,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    payload = {
        "speaker_id": speaker_id,
        "combination_size": combination_size,
        "total_candidates": total_candidates,
        "searched_combinations": combination_count(total_candidates, combination_size),
        "rows": rows,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
