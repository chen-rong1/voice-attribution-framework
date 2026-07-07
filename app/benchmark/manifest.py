"""CSV manifest loaders for real benchmark datasets."""

from __future__ import annotations

import csv
from pathlib import Path

from app.benchmark.models import BenchmarkClip
from app.common.errors import InvalidManifestError, ValidationIssue
from app.common.constants import DEFAULT_REJECT_LABEL
from app.services.identification import EnrollmentRecord

REQUIRED_MANIFEST_COLUMNS = {"kind", "output_rel_path", "agent"}
SUPPORTED_KINDS = {"enroll", "attribution"}
EXTERNAL_KNOWN_TRIAL_ROLES = {
    "external_known_query",
    "known_query",
    "positive_query",
    "positive",
    "pos",
    "known",
    "external_known",
}
EXTERNAL_UNKNOWN_TRIAL_ROLES = {
    "external_unknown_query",
    "unknown_query",
    "negative_query",
    "unknown",
    "external_unknown",
}


def load_from_manifest(
    *,
    dataset_root: Path,
    manifest_path: Path,
) -> tuple[list[EnrollmentRecord], list[BenchmarkClip]]:
    """Load enrollments and benchmark clips from a manifest CSV."""

    enrollment_map: dict[str, list[Path]] = {}
    clip_rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        _validate_manifest_header(
            manifest_path=manifest_path,
            fieldnames=reader.fieldnames,
        )
        for row in reader:
            row_number = (reader.line_num or 1)
            kind = _require_value(
                row,
                column_name="kind",
                manifest_path=manifest_path,
                row_number=row_number,
            )
            if kind not in SUPPORTED_KINDS:
                raise InvalidManifestError(
                    [
                        ValidationIssue(
                            code="invalid_kind",
                            row_number=row_number,
                            column_name="kind",
                            message=(
                                f"{manifest_path}: 第 {row_number} 行的 `kind` 非法：`{kind}`。"
                                f" 仅支持 {sorted(SUPPORTED_KINDS)}。"
                            ),
                        )
                    ]
                )
            output_rel_path = _require_value(
                row,
                column_name="output_rel_path",
                manifest_path=manifest_path,
                row_number=row_number,
            )
            agent = _require_value(
                row,
                column_name="agent",
                manifest_path=manifest_path,
                row_number=row_number,
            )
            if kind == "attribution":
                trial_role = _resolve_trial_role(row)
                _validate_trial_role(
                    trial_role=trial_role,
                    manifest_path=manifest_path,
                    row_number=row_number,
                )
            audio_path = dataset_root / output_rel_path
            _validate_audio_path_exists(
                audio_path=audio_path,
                manifest_path=manifest_path,
                row_number=row_number,
                column_name="output_rel_path",
            )
            _validate_time_range(
                row=row,
                manifest_path=manifest_path,
                row_number=row_number,
            )
            if kind == "enroll":
                enrollment_map.setdefault(agent, []).append(audio_path)
            elif kind == "attribution":
                clip_rows.append(
                    {
                        **row,
                        "agent": agent,
                        "output_rel_path": output_rel_path,
                        "__row_number__": str(row_number),
                    }
                )

    enrolled_agents = set(enrollment_map.keys())
    enrollments = [
        EnrollmentRecord(speaker_id=speaker_id, audio_paths=audio_paths)
        for speaker_id, audio_paths in sorted(enrollment_map.items())
    ]
    clips: list[BenchmarkClip] = []
    for row in clip_rows:
        agent = row["agent"]
        trial_role = _resolve_trial_role(row)
        _validate_external_known_agent(
            agent=agent,
            enrolled_agents=enrolled_agents,
            trial_role=trial_role,
            manifest_path=manifest_path,
            row_number=int(row.get("__row_number__", 0) or 0),
        )
        expected_label = _resolve_expected_label(
            agent=agent,
            enrolled_agents=enrolled_agents,
            trial_role=trial_role,
        )
        audio_path = dataset_root / row["output_rel_path"]
        clips.append(
            BenchmarkClip(
                clip_id=audio_path.stem,
                audio_path=audio_path,
                truth_label=agent,
                expected_label=expected_label,
                evaluation_group=_resolve_evaluation_group(
                    expected_label=expected_label,
                    trial_role=trial_role,
                ),
                metadata={
                    "trial_role": trial_role,
                    "trial_label": row.get("trial_label", ""),
                    "source_file": row.get("source_file", ""),
                    "note": row.get("note", ""),
                    "start_sec": row.get("start_sec", ""),
                    "end_sec": row.get("end_sec", ""),
                },
            )
        )
    return enrollments, clips


def _resolve_trial_role(row: dict[str, str]) -> str:
    trial_role = row.get("trial_role", "").strip().lower()
    if trial_role:
        if trial_role.startswith("neg_for_"):
            return "known"
        return trial_role
    legacy_trial_label = row.get("trial_label", "").strip().lower()
    if legacy_trial_label.startswith("neg_for_"):
        return "known"
    return legacy_trial_label


def _resolve_expected_label(
    *,
    agent: str,
    enrolled_agents: set[str],
    trial_role: str,
) -> str:
    if trial_role in EXTERNAL_UNKNOWN_TRIAL_ROLES:
        return DEFAULT_REJECT_LABEL
    if trial_role in EXTERNAL_KNOWN_TRIAL_ROLES:
        return agent
    return agent if agent in enrolled_agents else DEFAULT_REJECT_LABEL


def _resolve_evaluation_group(*, expected_label: str, trial_role: str) -> str:
    normalized = trial_role.strip().lower()
    if normalized in EXTERNAL_UNKNOWN_TRIAL_ROLES or expected_label == DEFAULT_REJECT_LABEL:
        return "external_unknown"
    if normalized in EXTERNAL_KNOWN_TRIAL_ROLES:
        return "external_known"
    return "internal_known"


def _validate_manifest_header(
    *,
    manifest_path: Path,
    fieldnames: list[str] | None,
) -> None:
    if not fieldnames:
        raise InvalidManifestError(
            [
                ValidationIssue(
                    code="missing_header",
                    message=f"{manifest_path}: manifest 缺少表头。",
                )
            ]
        )
    normalized_fieldnames = {fieldname.strip() for fieldname in fieldnames if fieldname.strip()}
    missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - normalized_fieldnames)
    if missing_columns:
        raise InvalidManifestError(
            [
                ValidationIssue(
                    code="missing_required_columns",
                    message=f"{manifest_path}: manifest 缺少必填列 {missing_columns}。",
                )
            ]
        )


def _require_value(
    row: dict[str, str],
    *,
    column_name: str,
    manifest_path: Path,
    row_number: int,
) -> str:
    value = row.get(column_name, "")
    normalized = value.strip()
    if normalized:
        return normalized
    raise InvalidManifestError(
        [
            ValidationIssue(
                code="missing_required_value",
                row_number=row_number,
                column_name=column_name,
                message=f"{manifest_path}: 第 {row_number} 行缺少必填字段 `{column_name}`。",
            )
        ]
    )


def _validate_trial_role(
    *,
    trial_role: str,
    manifest_path: Path,
    row_number: int,
) -> None:
    if not trial_role:
        raise InvalidManifestError(
            [
                ValidationIssue(
                    code="missing_trial_role",
                    row_number=row_number,
                    column_name="trial_role",
                    message=(
                        f"{manifest_path}: 第 {row_number} 行的 attribution 样本缺少 `trial_role`"
                        "，也没有可兼容的 `trial_label`。"
                    ),
                )
            ]
        )
    supported_trial_roles = sorted(EXTERNAL_KNOWN_TRIAL_ROLES | EXTERNAL_UNKNOWN_TRIAL_ROLES)
    if trial_role not in EXTERNAL_KNOWN_TRIAL_ROLES and trial_role not in EXTERNAL_UNKNOWN_TRIAL_ROLES:
        raise InvalidManifestError(
            [
                ValidationIssue(
                    code="invalid_trial_role",
                    row_number=row_number,
                    column_name="trial_role",
                    message=(
                        f"{manifest_path}: 第 {row_number} 行的 `trial_role` 非法：`{trial_role}`。"
                        f" 支持值包括 {supported_trial_roles}，或使用兼容旧值"
                        " `pos` / `positive` / `known` / `unknown`。"
                    ),
                )
            ]
        )


def _validate_audio_path_exists(
    *,
    audio_path: Path,
    manifest_path: Path,
    row_number: int,
    column_name: str,
) -> None:
    if audio_path.exists():
        return
    raise InvalidManifestError(
        [
            ValidationIssue(
                code="missing_audio_file",
                row_number=row_number,
                column_name=column_name,
                message=(
                    f"{manifest_path}: 第 {row_number} 行引用的音频文件不存在："
                    f"`{audio_path}`。"
                ),
            )
        ]
    )


def _validate_time_range(
    *,
    row: dict[str, str],
    manifest_path: Path,
    row_number: int,
) -> None:
    start_raw = row.get("start_sec", "").strip()
    end_raw = row.get("end_sec", "").strip()
    if not start_raw and not end_raw:
        return
    try:
        start_value = float(start_raw) if start_raw else 0.0
        end_value = float(end_raw) if end_raw else 0.0
    except ValueError as exc:
        raise InvalidManifestError(
            [
                ValidationIssue(
                    code="invalid_time_value",
                    row_number=row_number,
                    column_name="start_sec/end_sec",
                    message=(
                        f"{manifest_path}: 第 {row_number} 行的 `start_sec/end_sec` "
                        "必须是数字。"
                    ),
                )
            ]
        ) from exc
    if start_value < 0.0 or end_value <= start_value:
        raise InvalidManifestError(
            [
                ValidationIssue(
                    code="invalid_time_range",
                    row_number=row_number,
                    column_name="start_sec/end_sec",
                    message=(
                        f"{manifest_path}: 第 {row_number} 行的时间范围非法："
                        f" start_sec={start_value}, end_sec={end_value}。"
                    ),
                )
            ]
        )


def _validate_external_known_agent(
    *,
    agent: str,
    enrolled_agents: set[str],
    trial_role: str,
    manifest_path: Path,
    row_number: int,
) -> None:
    if trial_role not in EXTERNAL_KNOWN_TRIAL_ROLES:
        return
    if agent in enrolled_agents:
        return
    raise InvalidManifestError(
        [
            ValidationIssue(
                code="unenrolled_external_known_agent",
                row_number=row_number,
                column_name="agent",
                message=(
                    f"{manifest_path}: 第 {row_number} 行声明为 external known，但 `agent` "
                    f"`{agent}` 不在 enrollment 集合中。"
                ),
            )
        ]
    )
