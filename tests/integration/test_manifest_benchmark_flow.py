from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from app.benchmark.manifest import load_from_manifest
from app.common.errors import InvalidManifestError


def _write_tone(audio_path: Path, frequency: float, *, duration_sec: float = 1.5) -> None:
    sample_rate = 16000
    timeline = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * frequency * timeline).astype(np.float32)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, waveform, sample_rate)


def _write_manifest(manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "kind",
                "output_rel_path",
                "source_file",
                "start_sec",
                "end_sec",
                "agent",
                "trial_role",
                "trial_label",
                "note",
            ]
        )
        writer.writerow(
            [
                "enroll",
                "processed/enroll/eval/enroll_A_1.wav",
                "raw/a.wav",
                "0",
                "1.5",
                "A",
                "",
                "enroll",
                "demo",
            ]
        )
        writer.writerow(
            [
                "enroll",
                "processed/enroll/eval/enroll_B_1.wav",
                "raw/b.wav",
                "0",
                "1.5",
                "B",
                "",
                "enroll",
                "demo",
            ]
        )
        writer.writerow(
            [
                "attribution",
                "processed/attribution/eval/clip_A.wav",
                "raw/a.wav",
                "0",
                "1.5",
                "A",
                "external_known_query",
                "pos",
                "demo",
            ]
        )
        writer.writerow(
            [
                "attribution",
                "processed/attribution/eval/clip_C.wav",
                "raw/c.wav",
                "0",
                "1.5",
                "C",
                "external_unknown_query",
                "unknown",
                "demo",
            ]
        )


def _write_legacy_manifest(manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["kind", "output_rel_path", "source_file", "start_sec", "end_sec", "agent", "trial_label", "note"]
        )
        writer.writerow(
            ["enroll", "processed/enroll/eval/enroll_A_1.wav", "raw/a.wav", "0", "1.5", "A", "enroll", "demo"]
        )
        writer.writerow(
            ["attribution", "processed/attribution/eval/clip_A.wav", "raw/a.wav", "0", "1.5", "A", "pos", "demo"]
        )
        writer.writerow(
            ["attribution", "processed/attribution/eval/clip_C.wav", "raw/c.wav", "0", "1.5", "C", "unknown", "demo"]
        )


def test_manifest_loader_and_cli_script(tmp_path: Path) -> None:
    dataset_root = tmp_path / "strict_dataset"
    manifest_path = dataset_root / "meta" / "manifest.csv"
    _write_manifest(manifest_path)
    _write_tone(dataset_root / "processed" / "enroll" / "eval" / "enroll_A_1.wav", 220.0)
    _write_tone(dataset_root / "processed" / "enroll" / "eval" / "enroll_B_1.wav", 440.0)
    _write_tone(dataset_root / "processed" / "attribution" / "eval" / "clip_A.wav", 220.0)
    _write_tone(dataset_root / "processed" / "attribution" / "eval" / "clip_C.wav", 660.0)

    enrollments, clips = load_from_manifest(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
    )
    assert len(enrollments) == 2
    assert len(clips) == 2
    assert clips[0].evaluation_group == "external_known"
    assert clips[1].expected_label == "UNKNOWN"
    assert clips[1].evaluation_group == "external_unknown"

    project_root = Path(__file__).resolve().parents[2]
    output_root = tmp_path / "outputs"
    command = [
        sys.executable,
        str(project_root / "scripts" / "run_benchmark.py"),
        "--output-dir",
        str(output_root),
        "--run-name",
        "manifest_smoke",
        "--dataset-name",
        "strict_demo",
        "--manifest-path",
        str(manifest_path),
        "--dataset-root",
        str(dataset_root),
        "--dataset-role",
        "external_holdout",
        "--project-root",
        str(project_root),
        "--threshold",
        "0.5",
    ]
    result = subprocess.run(
        command,
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "strict_demo" in result.stdout
    assert (output_root / "manifest_smoke_测试总表.tsv").exists()
    assert (output_root / "manifest_smoke_正式报告.md").exists()
    assert (output_root / "manifest_smoke_摘要.json").exists()
    summary_payload = json.loads((output_root / "manifest_smoke_摘要.json").read_text(encoding="utf-8"))
    assert summary_payload["summary"]["dataset_role"] == "external_holdout"
    assert summary_payload["summary"]["external_known_total"] == 1
    assert summary_payload["summary"]["external_unknown_total"] == 1
    assert "latency_ms" in summary_payload["items"][0]
    assert summary_payload["items"][0]["metadata"]["trial_role"] == "external_known_query"
    assert summary_payload["items"][1]["metadata"]["trial_role"] == "external_unknown_query"


def test_manifest_loader_keeps_legacy_trial_label_compatibility(tmp_path: Path) -> None:
    dataset_root = tmp_path / "legacy_dataset"
    manifest_path = dataset_root / "meta" / "manifest_legacy.csv"
    _write_legacy_manifest(manifest_path)
    _write_tone(dataset_root / "processed" / "enroll" / "eval" / "enroll_A_1.wav", 220.0)
    _write_tone(dataset_root / "processed" / "attribution" / "eval" / "clip_A.wav", 220.0)
    _write_tone(dataset_root / "processed" / "attribution" / "eval" / "clip_C.wav", 660.0)

    enrollments, clips = load_from_manifest(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
    )

    assert len(enrollments) == 1
    assert len(clips) == 2
    assert clips[0].evaluation_group == "external_known"
    assert clips[0].metadata["trial_role"] == "pos"
    assert clips[1].expected_label == "UNKNOWN"
    assert clips[1].evaluation_group == "external_unknown"
    assert clips[1].metadata["trial_role"] == "unknown"


def test_manifest_loader_rejects_invalid_trial_role(tmp_path: Path) -> None:
    dataset_root = tmp_path / "invalid_role_dataset"
    manifest_path = dataset_root / "meta" / "manifest_invalid_role.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "output_rel_path", "agent", "trial_role"])
        writer.writerow(["enroll", "processed/enroll/eval/enroll_A_1.wav", "A", ""])
        writer.writerow(["attribution", "processed/attribution/eval/clip_A.wav", "A", "bad_role"])

    _write_tone(dataset_root / "processed" / "enroll" / "eval" / "enroll_A_1.wav", 220.0)
    _write_tone(dataset_root / "processed" / "attribution" / "eval" / "clip_A.wav", 220.0)

    try:
        load_from_manifest(dataset_root=dataset_root, manifest_path=manifest_path)
    except InvalidManifestError as error:
        issue = error.issues[0]
    else:
        raise AssertionError("Expected InvalidManifestError")

    assert issue.code == "invalid_trial_role"
    assert issue.row_number == 3
    assert issue.column_name == "trial_role"
    assert "bad_role" in issue.message


def test_manifest_loader_rejects_missing_required_field(tmp_path: Path) -> None:
    dataset_root = tmp_path / "missing_field_dataset"
    manifest_path = dataset_root / "meta" / "manifest_missing_agent.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "output_rel_path", "agent", "trial_role"])
        writer.writerow(["attribution", "processed/attribution/eval/clip_A.wav", "", "external_known_query"])

    _write_tone(dataset_root / "processed" / "attribution" / "eval" / "clip_A.wav", 220.0)

    try:
        load_from_manifest(dataset_root=dataset_root, manifest_path=manifest_path)
    except InvalidManifestError as error:
        issue = error.issues[0]
    else:
        raise AssertionError("Expected InvalidManifestError")

    assert issue.code == "missing_required_value"
    assert issue.row_number == 2
    assert issue.column_name == "agent"
    assert "缺少必填字段 `agent`" in issue.message


def test_manifest_loader_rejects_missing_header(tmp_path: Path) -> None:
    dataset_root = tmp_path / "missing_header_dataset"
    manifest_path = dataset_root / "meta" / "manifest_missing_header.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("", encoding="utf-8")

    try:
        load_from_manifest(dataset_root=dataset_root, manifest_path=manifest_path)
    except InvalidManifestError as error:
        issue = error.issues[0]
    else:
        raise AssertionError("Expected InvalidManifestError")

    assert issue.code == "missing_header"


def test_manifest_loader_rejects_missing_required_columns(tmp_path: Path) -> None:
    dataset_root = tmp_path / "missing_columns_dataset"
    manifest_path = dataset_root / "meta" / "manifest_missing_columns.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "agent"])
        writer.writerow(["enroll", "A"])

    try:
        load_from_manifest(dataset_root=dataset_root, manifest_path=manifest_path)
    except InvalidManifestError as error:
        issue = error.issues[0]
    else:
        raise AssertionError("Expected InvalidManifestError")

    assert issue.code == "missing_required_columns"


def test_manifest_loader_rejects_invalid_kind(tmp_path: Path) -> None:
    dataset_root = tmp_path / "invalid_kind_dataset"
    manifest_path = dataset_root / "meta" / "manifest_invalid_kind.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "output_rel_path", "agent"])
        writer.writerow(["bad_kind", "processed/enroll/eval/enroll_A_1.wav", "A"])

    try:
        load_from_manifest(dataset_root=dataset_root, manifest_path=manifest_path)
    except InvalidManifestError as error:
        issue = error.issues[0]
    else:
        raise AssertionError("Expected InvalidManifestError")

    assert issue.code == "invalid_kind"
    assert issue.row_number == 2


def test_manifest_loader_rejects_missing_trial_role(tmp_path: Path) -> None:
    dataset_root = tmp_path / "missing_trial_role_dataset"
    manifest_path = dataset_root / "meta" / "manifest_missing_trial_role.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "output_rel_path", "agent", "trial_role"])
        writer.writerow(["enroll", "processed/enroll/eval/enroll_A_1.wav", "A", ""])
        writer.writerow(["attribution", "processed/attribution/eval/clip_A.wav", "A", ""])

    _write_tone(dataset_root / "processed" / "enroll" / "eval" / "enroll_A_1.wav", 220.0)
    _write_tone(dataset_root / "processed" / "attribution" / "eval" / "clip_A.wav", 220.0)

    try:
        load_from_manifest(dataset_root=dataset_root, manifest_path=manifest_path)
    except InvalidManifestError as error:
        issue = error.issues[0]
    else:
        raise AssertionError("Expected InvalidManifestError")

    assert issue.code == "missing_trial_role"
    assert issue.row_number == 3


def test_manifest_loader_rejects_missing_audio_file(tmp_path: Path) -> None:
    dataset_root = tmp_path / "missing_audio_dataset"
    manifest_path = dataset_root / "meta" / "manifest_missing_audio.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "output_rel_path", "agent", "trial_role"])
        writer.writerow(["enroll", "processed/enroll/eval/enroll_A_1.wav", "A", ""])

    try:
        load_from_manifest(dataset_root=dataset_root, manifest_path=manifest_path)
    except InvalidManifestError as error:
        issue = error.issues[0]
    else:
        raise AssertionError("Expected InvalidManifestError")

    assert issue.code == "missing_audio_file"
    assert issue.row_number == 2


def test_manifest_loader_rejects_invalid_time_range(tmp_path: Path) -> None:
    dataset_root = tmp_path / "invalid_time_dataset"
    manifest_path = dataset_root / "meta" / "manifest_invalid_time.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "output_rel_path", "agent", "trial_role", "start_sec", "end_sec"])
        writer.writerow(["enroll", "processed/enroll/eval/enroll_A_1.wav", "A", "", "1.5", "1.0"])

    _write_tone(dataset_root / "processed" / "enroll" / "eval" / "enroll_A_1.wav", 220.0)

    try:
        load_from_manifest(dataset_root=dataset_root, manifest_path=manifest_path)
    except InvalidManifestError as error:
        issue = error.issues[0]
    else:
        raise AssertionError("Expected InvalidManifestError")

    assert issue.code == "invalid_time_range"
    assert issue.row_number == 2


def test_manifest_loader_rejects_external_known_agent_without_enrollment(tmp_path: Path) -> None:
    dataset_root = tmp_path / "external_known_dataset"
    manifest_path = dataset_root / "meta" / "manifest_external_known.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "output_rel_path", "agent", "trial_role"])
        writer.writerow(["enroll", "processed/enroll/eval/enroll_A_1.wav", "A", ""])
        writer.writerow(["attribution", "processed/attribution/eval/clip_B.wav", "B", "external_known_query"])

    _write_tone(dataset_root / "processed" / "enroll" / "eval" / "enroll_A_1.wav", 220.0)
    _write_tone(dataset_root / "processed" / "attribution" / "eval" / "clip_B.wav", 330.0)

    try:
        load_from_manifest(dataset_root=dataset_root, manifest_path=manifest_path)
    except InvalidManifestError as error:
        issue = error.issues[0]
    else:
        raise AssertionError("Expected InvalidManifestError")

    assert issue.code == "unenrolled_external_known_agent"
    assert issue.row_number == 3


def test_manifest_loader_treats_neg_for_legacy_labels_as_known_queries(tmp_path: Path) -> None:
    dataset_root = tmp_path / "legacy_neg_for_dataset"
    manifest_path = dataset_root / "meta" / "manifest_legacy_neg_for.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kind", "output_rel_path", "agent", "trial_label"])
        writer.writerow(["enroll", "processed/enroll/eval/enroll_A_1.wav", "A", "enroll"])
        writer.writerow(["attribution", "processed/attribution/eval/clip_A_neg_for_B.wav", "A", "neg_for_B"])

    _write_tone(dataset_root / "processed" / "enroll" / "eval" / "enroll_A_1.wav", 220.0)
    _write_tone(dataset_root / "processed" / "attribution" / "eval" / "clip_A_neg_for_B.wav", 220.0)

    enrollments, clips = load_from_manifest(dataset_root=dataset_root, manifest_path=manifest_path)

    assert len(enrollments) == 1
    assert len(clips) == 1
    assert clips[0].expected_label == "A"
    assert clips[0].evaluation_group == "external_known"
    assert clips[0].metadata["trial_role"] == "known"
