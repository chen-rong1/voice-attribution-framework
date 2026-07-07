"""Run a benchmark from enrollment and test directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.benchmark.business import load_business_benchmark_clips
from app.benchmark.filesystem import (
    load_benchmark_clips_from_directory,
    load_enrollments_from_directory,
)
from app.benchmark.manifest import load_from_manifest
from app.benchmark.models import BenchmarkRunConfig
from app.benchmark.runner import BenchmarkRunner
from app.common.config import load_simple_yaml_map
from app.reporting.json_summary import write_benchmark_json_summary
from app.reporting.markdown import write_benchmark_markdown
from app.reporting.tsv import write_benchmark_tsv
from app.scoring.models import ScoringStrategy
from app.services.bootstrap import register_default_backends
from app.services.container import FrameworkContainer
from app.services.identification import EnrollmentRecord, IdentificationService


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run voice attribution benchmark from directories.")
    parser.add_argument("--enroll-dir", type=Path, help="注册样本目录")
    parser.add_argument("--test-dir", type=Path, help="测试样本目录")
    parser.add_argument("--business-dataset-dir", type=Path, help="平铺业务集目录")
    parser.add_argument("--business-truth-tsv", type=Path, help="业务集真值表路径")
    parser.add_argument("--business-pure-list", type=Path, help="业务集纯净测试清单路径")
    parser.add_argument("--enroll-speaker", help="business 模式下的注册说话人 ID")
    parser.add_argument("--enroll-list", type=Path, help="business 模式下的注册清单文件")
    parser.add_argument(
        "--enroll-file",
        action="append",
        type=Path,
        default=[],
        help="business 模式下的注册音频路径，可重复传入",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="输出目录")
    parser.add_argument("--run-name", required=True, help="运行名称")
    parser.add_argument("--dataset-name", required=True, help="数据集名称")
    parser.add_argument("--dataset-version", default="v1", help="数据集版本")
    parser.add_argument("--manifest-path", type=Path, help="可选的 CSV 清单路径")
    parser.add_argument("--dataset-root", type=Path, help="manifest 模式下的数据根目录")
    parser.add_argument("--scoring-config", type=Path, help="打分配置文件路径")
    parser.add_argument("--threshold", type=float, help="判定阈值")
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in ScoringStrategy],
        help="打分策略",
    )
    parser.add_argument(
        "--profile-aggregation-strategy",
        choices=["center", "quality_weighted_center"],
        help="画像聚合策略",
    )
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
    scoring_config = _load_scoring_config(
        args.scoring_config or args.project_root / "configs" / "scoring" / "default.yaml"
    )
    threshold_value = float(scoring_config.get("threshold", "0.41"))
    if args.threshold is not None:
        threshold_value = args.threshold
    strategy_value = scoring_config.get("strategy", ScoringStrategy.CENTER.value)
    if args.strategy is not None:
        strategy_value = args.strategy
    profile_aggregation_strategy = scoring_config.get("profile_aggregation_strategy", "center")
    if args.profile_aggregation_strategy is not None:
        profile_aggregation_strategy = args.profile_aggregation_strategy
    if profile_aggregation_strategy not in {"center", "quality_weighted_center"}:
        parser.error("profile_aggregation_strategy 只支持 center 或 quality_weighted_center")
    if strategy_value not in {strategy.value for strategy in ScoringStrategy}:
        parser.error("strategy 配置无效")

    container = FrameworkContainer()
    register_default_backends(project_root=args.project_root, registry=container.backend_registry)
    model_config = args.project_root / "configs" / "models" / "default.yaml"
    backend_name = load_simple_yaml_map(model_config).get(
        "bootstrap_backend",
        "wespeaker-ecapa1024-lm-onnx",
    )

    backend = container.backend_registry.get(backend_name)
    backend.load()

    service = IdentificationService(backend)
    runner = BenchmarkRunner(service)
    data_mode_count = sum(
        (
            args.manifest_path is not None,
            args.business_dataset_dir is not None,
            args.enroll_dir is not None or args.test_dir is not None,
        )
    )
    if data_mode_count != 1:
        parser.error("必须且只能选择一种输入模式：目录模式、manifest 模式或 business 模式")

    if args.manifest_path is not None:
        if args.dataset_root is None:
            parser.error("使用 --manifest-path 时必须同时提供 --dataset-root")
        enrollments, clips = load_from_manifest(
            dataset_root=args.dataset_root,
            manifest_path=args.manifest_path,
        )
    elif args.business_dataset_dir is not None:
        if not args.enroll_speaker:
            parser.error("business 模式下必须提供 --enroll-speaker")
        enroll_files = _resolve_business_enroll_files(
            base_dir=args.business_dataset_dir,
            enroll_list=args.enroll_list,
            enroll_files=args.enroll_file,
        )
        if not enroll_files:
            parser.error("business 模式下至少需要一个 --enroll-file 或 --enroll-list")
        enrollments = [
            EnrollmentRecord(
                speaker_id=args.enroll_speaker,
                audio_paths=enroll_files,
            )
        ]
        clips = load_business_benchmark_clips(
            args.business_dataset_dir,
            truth_tsv_path=args.business_truth_tsv,
            pure_list_path=args.business_pure_list,
        )
    else:
        if args.enroll_dir is None or args.test_dir is None:
            parser.error("目录模式下必须同时提供 --enroll-dir 和 --test-dir")
        enrollments = load_enrollments_from_directory(args.enroll_dir)
        clips = load_benchmark_clips_from_directory(args.test_dir)
    result = runner.run(
        config=BenchmarkRunConfig(
            run_name=args.run_name,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            backend_name=backend.backend_name,
            scoring_strategy=ScoringStrategy(strategy_value),
            threshold_value=threshold_value,
            profile_aggregation_strategy=profile_aggregation_strategy,
        ),
        enrollments=enrollments,
        clips=clips,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_benchmark_tsv(result, args.output_dir / f"{args.run_name}_测试总表.tsv")
    write_benchmark_markdown(result, args.output_dir / f"{args.run_name}_正式报告.md")
    write_benchmark_json_summary(result, args.output_dir / f"{args.run_name}_摘要.json")
    print(result.summary)
    return 0


def _load_scoring_config(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        return {}
    return load_simple_yaml_map(config_path)


def _resolve_audio_path(audio_path: Path, *, base_dir: Path) -> Path:
    if audio_path.is_absolute():
        return audio_path
    return (base_dir / audio_path).resolve()


def _load_path_list(list_path: Path) -> list[Path]:
    paths: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(Path(line))
    return paths


def _resolve_business_enroll_files(
    *,
    base_dir: Path,
    enroll_list: Path | None,
    enroll_files: list[Path],
) -> list[Path]:
    resolved_files = [_resolve_audio_path(audio_path, base_dir=base_dir) for audio_path in enroll_files]
    if enroll_list is not None:
        for audio_path in _load_path_list(enroll_list):
            resolved_files.append(_resolve_audio_path(audio_path, base_dir=base_dir))
    return resolved_files


if __name__ == "__main__":
    raise SystemExit(main())
