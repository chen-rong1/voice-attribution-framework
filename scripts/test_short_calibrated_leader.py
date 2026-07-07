#!/usr/bin/env python3
"""Ad hoc tester for the short_calibrated_leader rescue path."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_AUDIT_JSON = WORKSPACE_ROOT / "tmp_candidate_audit_max035_branchfix_external_known_v3.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding_backends.models import EmbeddingResult
from app.profiles.models import SpeakerEmbeddingSample, SpeakerProfile
from app.scoring.models import ScoringStrategy
from app.scoring.strategies import build_decision


def _profile(speaker_id: str, raw_score: float) -> SpeakerProfile:
    vector = np.asarray([raw_score, math.sqrt(1.0 - raw_score**2)], dtype=np.float32)
    result = EmbeddingResult(
        sample_id=f"{speaker_id}-sample",
        backend_name="debug",
        backend_version="0.1.0",
        feature_version="fbank80_v1",
        embedding=vector,
        embedding_dim=vector.shape[0],
    )
    return SpeakerProfile(
        speaker_id=speaker_id,
        profile_name="default",
        backend_name="debug",
        backend_version="0.1.0",
        feature_version="fbank80_v1",
        aggregation_strategy="center",
        vector=vector,
        members=[SpeakerEmbeddingSample(speaker_id=speaker_id, embedding_result=result)],
    )


def run_synthetic_case() -> dict[str, object]:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    huangbaichao = _profile("huangbaichao", 0.2865)
    passerby_a = _profile("passerby_a", 0.1812)
    chenrong = _profile("chenrong", 0.0700)

    for profile, mean, std, floor in (
        (huangbaichao, 0.01, 0.05, 0.18),
        (passerby_a, 0.03, 0.11, 0.18),
        (chenrong, 0.02, 0.10, 0.16),
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
    return {
        "decision": result.decision.value,
        "final_label": result.final_label,
        "accept_reason": result.metadata.get("accept_reason", ""),
        "reject_reason": result.metadata.get("reject_reason", ""),
        "accept_score_space": result.metadata.get("accept_score_space", ""),
        "top1_raw_score": result.metadata.get("top1_raw_score"),
        "top2_score": result.metadata.get("top2_score"),
        "calibrated_score": result.metadata.get("calibrated_score"),
        "effective_threshold_value": result.metadata.get("effective_threshold_value"),
        "effective_calibrated_threshold_value": result.metadata.get(
            "effective_calibrated_threshold_value"
        ),
    }


def audit_candidate_window(audit_json: Path) -> dict[str, object]:
    payload = json.loads(audit_json.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    counts = payload.get("counts", {})
    return {
        "audit_json": str(audit_json),
        "matched_total": payload.get("matched_total", len(rows)),
        "counts": counts,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=DEFAULT_AUDIT_JSON,
        help="Path to the candidate audit JSON produced from the max035 v3 suite.",
    )
    parser.add_argument(
        "--mode",
        choices=("synthetic", "audit", "both"),
        default="both",
        help="Which checks to run.",
    )
    args = parser.parse_args()

    output: dict[str, object] = {}
    if args.mode in {"synthetic", "both"}:
        output["synthetic_case"] = run_synthetic_case()
    if args.mode in {"audit", "both"}:
        output["candidate_window_audit"] = audit_candidate_window(args.audit_json)

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
