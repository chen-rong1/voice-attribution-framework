"""Run the current verified Liaoning 0222 business solution."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行当前已验证的辽宁0222官方业务方案。")
    parser.add_argument(
        "--business-dataset-dir",
        type=Path,
        default=Path(
            "/Users/工作/声纹识别/voice-benchmark-strict/processed/attribution/"
            "辽宁0222_前5分钟_diarization_thr1.0_merge_gap1.2"
        ),
        help="业务集目录",
    )
    parser.add_argument(
        "--strict-enroll-dir",
        type=Path,
        default=Path("/Users/工作/声纹识别/voice-benchmark-strict/processed/enroll/eval"),
        help="标准严格集注册目录",
    )
    parser.add_argument(
        "--strict-test-dir",
        type=Path,
        default=Path("/Users/工作/声纹识别/voice-benchmark-strict/processed/attribution/eval"),
        help="标准严格集测试目录",
    )
    parser.add_argument(
        "--enroll-speaker",
        default="xiaoli",
        help="业务集注册说话人 ID",
    )
    parser.add_argument(
        "--enroll-list",
        type=Path,
        default=Path("configs/enrollment_packs/liaoning0222_xiaoli_best_verified.txt"),
        help="注册清单文件",
    )
    parser.add_argument(
        "--scoring-config",
        type=Path,
        default=Path("configs/scoring/business_best_verified.yaml"),
        help="打分配置文件",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/benchmark/official_liaoning0222_solution"),
        help="输出根目录",
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

    project_root = args.project_root.resolve()
    output_root = _resolve_config_path(project_root, args.output_root)
    business_output_dir = (output_root / "business").resolve()
    strict_output_dir = (output_root / "strict").resolve()
    enroll_list = _resolve_config_path(project_root, args.enroll_list)
    scoring_config = _resolve_config_path(project_root, args.scoring_config)

    business_run_name = "liaoning0222_official_business"
    strict_run_name = "liaoning0222_official_strict"

    _run_command(
        [
            sys.executable,
            str(project_root / "scripts" / "run_benchmark.py"),
            "--business-dataset-dir",
            str(args.business_dataset_dir),
            "--enroll-speaker",
            args.enroll_speaker,
            "--enroll-list",
            str(enroll_list),
            "--scoring-config",
            str(scoring_config),
            "--output-dir",
            str(business_output_dir),
            "--run-name",
            business_run_name,
            "--dataset-name",
            "liaoning0222_business",
            "--dataset-version",
            "thr1.0_merge_gap1.2",
            "--project-root",
            str(project_root),
        ],
        cwd=project_root,
    )
    _run_command(
        [
            sys.executable,
            str(project_root / "scripts" / "run_benchmark.py"),
            "--enroll-dir",
            str(args.strict_enroll_dir),
            "--test-dir",
            str(args.strict_test_dir),
            "--scoring-config",
            str(scoring_config),
            "--output-dir",
            str(strict_output_dir),
            "--run-name",
            strict_run_name,
            "--dataset-name",
            "voice-benchmark-strict",
            "--dataset-version",
            "eval",
            "--project-root",
            str(project_root),
        ],
        cwd=project_root,
    )

    business_summary = _load_summary(business_output_dir / f"{business_run_name}_摘要.json")
    strict_summary = _load_summary(strict_output_dir / f"{strict_run_name}_摘要.json")
    summary_payload = {
        "official_solution_name": "liaoning0222_xiaoli_best_verified",
        "enroll_speaker": args.enroll_speaker,
        "enroll_list": str(enroll_list),
        "scoring_config": str(scoring_config),
        "business_summary": business_summary,
        "strict_summary": strict_summary,
    }
    write_combined_summary(
        output_root=output_root,
        summary_payload=summary_payload,
    )
    print(summary_payload)
    return 0


def _resolve_config_path(project_root: Path, config_path: Path) -> Path:
    if config_path.is_absolute():
        return config_path
    return (project_root / config_path).resolve()


def _run_command(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _load_summary(summary_path: Path) -> dict[str, object]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return dict(payload["summary"])


def write_combined_summary(*, output_root: Path, summary_payload: dict[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "官方方案汇总.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    business = summary_payload["business_summary"]
    strict = summary_payload["strict_summary"]
    markdown = [
        "# 辽宁0222官方方案汇总",
        "",
        "## 方案组成",
        "",
        f"- 注册清单：`{summary_payload['enroll_list']}`",
        f"- 打分配置：`{summary_payload['scoring_config']}`",
        "",
        "## 业务集结果",
        "",
        f"- 准确率：`{float(business['accuracy']):.2%}`",
        f"- 正样本召回：`{float(business['positive_recall']):.2%}`",
        f"- 非本人拒识率：`{float(business['unknown_reject_rate']):.2%}`",
        "",
        "## 标准集结果",
        "",
        f"- 准确率：`{float(strict['accuracy']):.2%}`",
        f"- 正样本召回：`{float(strict['positive_recall']):.2%}`",
        f"- 非本人拒识率：`{float(strict['unknown_reject_rate']):.2%}`",
        "",
    ]
    (output_root / "官方方案汇总.md").write_text("\n".join(markdown), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
