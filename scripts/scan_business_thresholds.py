"""Scan thresholds for flattened business benchmark datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from app.benchmark.business import load_business_benchmark_clips
from app.common.config import load_simple_yaml_map
from app.scoring.models import ScoringStrategy
from app.services.bootstrap import register_default_backends
from app.services.container import FrameworkContainer
from app.services.identification import EnrollmentRecord, IdentificationService


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="扫描业务集阈值，找出更适合当前业务的配置。")
    parser.add_argument("--business-dataset-dir", required=True, type=Path, help="平铺业务集目录")
    parser.add_argument("--business-truth-tsv", type=Path, help="业务集真值表路径")
    parser.add_argument("--business-pure-list", type=Path, help="业务集纯净测试清单路径")
    parser.add_argument("--enroll-speaker", required=True, help="注册说话人 ID")
    parser.add_argument(
        "--enroll-file",
        action="append",
        required=True,
        type=Path,
        help="注册音频路径，可重复传入",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        required=True,
        choices=[strategy.value for strategy in ScoringStrategy],
        help="要扫描的打分策略，可重复传入",
    )
    parser.add_argument("--threshold-start", type=float, required=True, help="阈值起点")
    parser.add_argument("--threshold-end", type=float, required=True, help="阈值终点")
    parser.add_argument("--threshold-step", type=float, required=True, help="阈值步长")
    parser.add_argument("--output-dir", required=True, type=Path, help="输出目录")
    parser.add_argument("--run-name", required=True, help="运行名称")
    parser.add_argument("--dataset-name", required=True, help="数据集名称")
    parser.add_argument("--dataset-version", default="v1", help="数据集版本")
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

    thresholds = build_threshold_values(
        start=args.threshold_start,
        end=args.threshold_end,
        step=args.threshold_step,
    )
    if not thresholds:
        parser.error("阈值范围为空，请检查起点、终点和步长")

    container = FrameworkContainer()
    register_default_backends(project_root=args.project_root, registry=container.backend_registry)
    model_config = args.project_root / "configs" / "models" / "default.yaml"
    backend_name = load_simple_yaml_map(model_config).get(
        "bootstrap_backend",
        "wespeaker-ecapa1024-lm-onnx",
    )
    backend = container.backend_registry.get(backend_name)
    backend.load()

    clips = load_business_benchmark_clips(
        args.business_dataset_dir,
        truth_tsv_path=args.business_truth_tsv,
        pure_list_path=args.business_pure_list,
    )
    enrollments = [
        EnrollmentRecord(
            speaker_id=args.enroll_speaker,
            audio_paths=[
                resolve_audio_path(audio_path, base_dir=args.business_dataset_dir)
                for audio_path in args.enroll_file
            ],
        )
    ]

    service = IdentificationService(backend)
    score_rows_by_strategy: dict[str, list[dict[str, str | float | bool]]] = {}
    sweep_rows_by_strategy: dict[str, list[dict[str, str | float | int]]] = {}
    best_rows: dict[str, dict[str, str | float | int]] = {}
    for strategy_value in args.strategy:
        strategy = ScoringStrategy(strategy_value)
        profile_aggregation_strategy = default_profile_aggregation_for(strategy)
        profiles = service.build_profiles(
            enrollments,
            profile_aggregation_strategy=profile_aggregation_strategy,
        )
        score_rows = collect_clip_scores(
            service=service,
            clips=clips,
            profiles=profiles,
            strategy=strategy,
        )
        score_rows_by_strategy[strategy.value] = score_rows
        sweep_rows = [
            evaluate_threshold(
                score_rows=score_rows,
                threshold=threshold,
                strategy=strategy,
                profile_aggregation_strategy=profile_aggregation_strategy,
                speaker_id=args.enroll_speaker,
            )
            for threshold in thresholds
        ]
        sweep_rows_by_strategy[strategy.value] = sweep_rows
        best_rows[strategy.value] = select_best_row(sweep_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_threshold_scan_tsv(
        sweep_rows_by_strategy=sweep_rows_by_strategy,
        output_path=args.output_dir / f"{args.run_name}_阈值扫描.tsv",
    )
    write_threshold_scan_markdown(
        args=args,
        backend_name=backend.backend_name,
        best_rows=best_rows,
        sweep_rows_by_strategy=sweep_rows_by_strategy,
        output_path=args.output_dir / f"{args.run_name}_阈值扫描报告.md",
    )
    write_threshold_scan_json(
        args=args,
        backend_name=backend.backend_name,
        best_rows=best_rows,
        score_rows_by_strategy=score_rows_by_strategy,
        sweep_rows_by_strategy=sweep_rows_by_strategy,
        output_path=args.output_dir / f"{args.run_name}_阈值扫描摘要.json",
    )

    for strategy_value in args.strategy:
        best_row = best_rows[strategy_value]
        print(
            {
                "strategy": strategy_value,
                "profile_aggregation_strategy": best_row["profile_aggregation_strategy"],
                "best_threshold": best_row["threshold_value"],
                "accuracy": best_row["accuracy"],
                "positive_recall": best_row["positive_recall"],
                "unknown_reject_rate": best_row["unknown_reject_rate"],
                "correct": best_row["correct"],
                "total": best_row["total"],
            }
        )
    return 0


def build_threshold_values(*, start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("threshold step must be positive")
    values: list[float] = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def resolve_audio_path(audio_path: Path, *, base_dir: Path) -> Path:
    if audio_path.is_absolute():
        return audio_path
    return (base_dir / audio_path).resolve()


def default_profile_aggregation_for(strategy: ScoringStrategy) -> str:
    if strategy == ScoringStrategy.QUALITY_WEIGHTED_CENTER:
        return "quality_weighted_center"
    return "center"


def collect_clip_scores(
    *,
    service: IdentificationService,
    clips: list,
    profiles: list,
    strategy: ScoringStrategy,
) -> list[dict[str, str | float | bool]]:
    rows: list[dict[str, str | float | bool]] = []
    for clip in clips:
        decision = service.identify(
            clip.audio_path,
            profiles,
            threshold_value=-1.0,
            scoring_strategy=strategy,
        )
        rows.append(
            {
                "clip_id": clip.clip_id,
                "audio_path": str(clip.audio_path),
                "truth_label": clip.truth_label,
                "expected_label": clip.expected_label,
                "best_score": float(decision.best_score),
                "score_breakdown": json.dumps(decision.score_breakdown, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def evaluate_threshold(
    *,
    score_rows: list[dict[str, str | float | bool]],
    threshold: float,
    strategy: ScoringStrategy,
    profile_aggregation_strategy: str,
    speaker_id: str,
) -> dict[str, str | float | int]:
    total = len(score_rows)
    correct = 0
    positive_total = 0
    positive_correct = 0
    unknown_total = 0
    unknown_correct = 0
    for row in score_rows:
        expected_label = str(row["expected_label"])
        best_score = float(row["best_score"])
        final_label = speaker_id if best_score >= threshold else "UNKNOWN"
        is_correct = final_label == expected_label
        correct += int(is_correct)
        if expected_label == "UNKNOWN":
            unknown_total += 1
            unknown_correct += int(is_correct)
        else:
            positive_total += 1
            positive_correct += int(is_correct)

    return {
        "strategy": strategy.value,
        "profile_aggregation_strategy": profile_aggregation_strategy,
        "threshold_value": threshold,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "positive_total": positive_total,
        "positive_correct": positive_correct,
        "positive_recall": positive_correct / positive_total if positive_total else 0.0,
        "unknown_total": unknown_total,
        "unknown_correct": unknown_correct,
        "unknown_reject_rate": unknown_correct / unknown_total if unknown_total else 0.0,
    }


def select_best_row(rows: list[dict[str, str | float | int]]) -> dict[str, str | float | int]:
    return max(
        rows,
        key=lambda row: (
            int(row["correct"]),
            int(row["positive_correct"]),
            int(row["unknown_correct"]),
            -abs(float(row["threshold_value"]) - 0.41),
        ),
    )


def write_threshold_scan_tsv(
    *,
    sweep_rows_by_strategy: dict[str, list[dict[str, str | float | int]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(
            [
                "打分策略",
                "画像聚合策略",
                "阈值",
                "正确数",
                "总数",
                "准确率",
                "本人样本数",
                "本人命中数",
                "本人召回",
                "非本人样本数",
                "非本人拒识数",
                "非本人拒识率",
            ]
        )
        for strategy_value, rows in sweep_rows_by_strategy.items():
            for row in rows:
                writer.writerow(
                    [
                        strategy_value,
                        row["profile_aggregation_strategy"],
                        f"{float(row['threshold_value']):.4f}",
                        row["correct"],
                        row["total"],
                        f"{float(row['accuracy']):.2%}",
                        row["positive_total"],
                        row["positive_correct"],
                        f"{float(row['positive_recall']):.2%}",
                        row["unknown_total"],
                        row["unknown_correct"],
                        f"{float(row['unknown_reject_rate']):.2%}",
                    ]
                )


def write_threshold_scan_markdown(
    *,
    args: argparse.Namespace,
    backend_name: str,
    best_rows: dict[str, dict[str, str | float | int]],
    sweep_rows_by_strategy: dict[str, list[dict[str, str | float | int]]],
    output_path: Path,
) -> None:
    lines = [
        f"# {args.run_name} 阈值扫描报告",
        "",
        "## 测试口径",
        "",
        f"- 数据集：`{args.dataset_name}` / `{args.dataset_version}`",
        f"- Backend：`{backend_name}`",
        f"- 注册说话人：`{args.enroll_speaker}`",
        f"- 阈值范围：`{args.threshold_start:.4f}` 到 `{args.threshold_end:.4f}`，步长 `{args.threshold_step:.4f}`",
        "",
        "## 最优结果",
        "",
        "| 打分策略 | 画像聚合策略 | 最优阈值 | 正确数 | 总数 | 准确率 | 本人召回 | 非本人拒识率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy_value in args.strategy:
        row = best_rows[strategy_value]
        lines.append(
            f"| {strategy_value} | {row['profile_aggregation_strategy']} | "
            f"{float(row['threshold_value']):.4f} | {row['correct']} | {row['total']} | "
            f"{float(row['accuracy']):.2%} | {float(row['positive_recall']):.2%} | "
            f"{float(row['unknown_reject_rate']):.2%} |"
        )
    lines.append("")
    lines.append("## 全量扫描")
    lines.append("")
    for strategy_value in args.strategy:
        lines.append(f"### {strategy_value}")
        lines.append("")
        lines.append("| 阈值 | 正确数 | 总数 | 准确率 | 本人命中数 | 本人召回 | 非本人拒识数 | 非本人拒识率 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in sweep_rows_by_strategy[strategy_value]:
            lines.append(
                f"| {float(row['threshold_value']):.4f} | {row['correct']} | {row['total']} | "
                f"{float(row['accuracy']):.2%} | {row['positive_correct']} | "
                f"{float(row['positive_recall']):.2%} | {row['unknown_correct']} | "
                f"{float(row['unknown_reject_rate']):.2%} |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_threshold_scan_json(
    *,
    args: argparse.Namespace,
    backend_name: str,
    best_rows: dict[str, dict[str, str | float | int]],
    score_rows_by_strategy: dict[str, list[dict[str, str | float | bool]]],
    sweep_rows_by_strategy: dict[str, list[dict[str, str | float | int]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": args.run_name,
        "dataset_name": args.dataset_name,
        "dataset_version": args.dataset_version,
        "backend_name": backend_name,
        "enroll_speaker": args.enroll_speaker,
        "threshold_range": {
            "start": args.threshold_start,
            "end": args.threshold_end,
            "step": args.threshold_step,
        },
        "best_rows": best_rows,
        "sweep_rows_by_strategy": sweep_rows_by_strategy,
        "score_rows_by_strategy": score_rows_by_strategy,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
