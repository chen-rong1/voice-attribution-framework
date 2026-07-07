"""Recommend enrollment clips for flattened business datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from app.audio.io import load_audio_chunk
from app.benchmark.business import load_business_benchmark_clips
from app.common.config import load_simple_yaml_map
from app.embedding_backends.models import EmbeddingRequest
from app.scoring.strategies import cosine_similarity
from app.services.bootstrap import register_default_backends
from app.services.container import FrameworkContainer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为真实业务集推荐更稳的注册样本组合。")
    parser.add_argument("--business-dataset-dir", required=True, type=Path, help="平铺业务集目录")
    parser.add_argument("--speaker-id", required=True, help="目标说话人 ID")
    parser.add_argument("--top-k", type=int, default=4, help="推荐输出前 K 条")
    parser.add_argument("--business-truth-tsv", type=Path, help="业务集真值表路径")
    parser.add_argument("--business-pure-list", type=Path, help="业务集纯净测试清单路径")
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
    if args.top_k <= 0:
        parser.error("--top-k 必须大于 0")

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
    speaker_clips = [clip for clip in clips if clip.expected_label == args.speaker_id]
    unknown_clips = [clip for clip in clips if clip.expected_label == "UNKNOWN"]
    if not speaker_clips:
        parser.error(f"没有找到说话人 `{args.speaker_id}` 的候选片段")

    candidates = []
    for index, clip in enumerate(speaker_clips, start=1):
        audio_chunk = load_audio_chunk(clip.audio_path)
        embedding_result = backend.extract_embedding(
            EmbeddingRequest(sample_id=f"{args.speaker_id}-candidate-{index}", audio=audio_chunk)
        )
        candidates.append(
            {
                "clip_id": clip.clip_id,
                "audio_path": clip.audio_path,
                "duration_sec": float(audio_chunk.duration_sec),
                "quality_score": float(embedding_result.quality_score or 0.0),
                "embedding": embedding_result.embedding,
                "source_segment_count": int(clip.metadata.get("source_segment_count", 0)),
            }
        )
    unknown_embeddings = []
    for index, clip in enumerate(unknown_clips, start=1):
        audio_chunk = load_audio_chunk(clip.audio_path)
        embedding_result = backend.extract_embedding(
            EmbeddingRequest(sample_id=f"unknown-candidate-{index}", audio=audio_chunk)
        )
        unknown_embeddings.append(embedding_result.embedding)

    annotated_rows = build_candidate_rows(candidates, unknown_embeddings=unknown_embeddings)
    recommended_rows = sorted(annotated_rows, key=lambda row: row["recommend_score"], reverse=True)
    recommended_rows = recommended_rows[: min(args.top_k, len(recommended_rows))]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_recommendation_tsv(
        rows=annotated_rows,
        output_path=args.output_dir / f"{args.run_name}_注册样本审计.tsv",
    )
    write_recommendation_markdown(
        speaker_id=args.speaker_id,
        top_k=args.top_k,
        rows=annotated_rows,
        recommended_rows=recommended_rows,
        output_path=args.output_dir / f"{args.run_name}_注册样本建议.md",
    )
    write_recommendation_json(
        speaker_id=args.speaker_id,
        top_k=args.top_k,
        rows=annotated_rows,
        recommended_rows=recommended_rows,
        output_path=args.output_dir / f"{args.run_name}_注册样本建议.json",
    )

    print(
        {
            "speaker_id": args.speaker_id,
            "candidate_count": len(annotated_rows),
            "recommended_files": [row["clip_filename"] for row in recommended_rows],
        }
    )
    return 0


def build_candidate_rows(
    candidates: list[dict[str, object]],
    *,
    unknown_embeddings: list[np.ndarray],
) -> list[dict[str, object]]:
    embeddings = [np.asarray(candidate["embedding"], dtype=np.float32) for candidate in candidates]
    if len(embeddings) == 1:
        only = candidates[0]
        return [
            {
                "clip_filename": Path(str(only["audio_path"])).name,
                "clip_id": str(only["clip_id"]),
                "duration_sec": float(only["duration_sec"]),
                "quality_score": float(only["quality_score"]),
                "cohort_similarity": 1.0,
                "center_similarity": 1.0,
                "unknown_avg_similarity": 0.0,
                "unknown_max_similarity": 0.0,
                "discrimination_margin": 1.0,
                "source_segment_count": int(only["source_segment_count"]),
                "recommend_score": float(only["quality_score"]),
            }
        ]

    center = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        current_embedding = embeddings[index]
        peer_similarities = [
            cosine_similarity(current_embedding, other_embedding)
            for peer_index, other_embedding in enumerate(embeddings)
            if peer_index != index
        ]
        cohort_similarity = float(np.mean(peer_similarities, dtype=np.float32)) if peer_similarities else 0.0
        center_similarity = float(cosine_similarity(current_embedding, center))
        unknown_similarities = [
            cosine_similarity(current_embedding, np.asarray(unknown_embedding, dtype=np.float32))
            for unknown_embedding in unknown_embeddings
        ]
        unknown_avg_similarity = (
            float(np.mean(unknown_similarities, dtype=np.float32)) if unknown_similarities else 0.0
        )
        unknown_max_similarity = max(unknown_similarities) if unknown_similarities else 0.0
        discrimination_margin = cohort_similarity - unknown_max_similarity
        quality_score = float(candidate["quality_score"])
        duration_sec = float(candidate["duration_sec"])
        duration_bonus = min(duration_sec / 8.0, 1.0)
        recommend_score = (
            0.30 * quality_score
            + 0.25 * max(cohort_similarity, 0.0)
            + 0.15 * max(center_similarity, 0.0) * duration_bonus
            + 0.30 * discrimination_margin
        )
        rows.append(
            {
                "clip_filename": Path(str(candidate["audio_path"])).name,
                "clip_id": str(candidate["clip_id"]),
                "duration_sec": duration_sec,
                "quality_score": quality_score,
                "cohort_similarity": cohort_similarity,
                "center_similarity": center_similarity,
                "unknown_avg_similarity": unknown_avg_similarity,
                "unknown_max_similarity": unknown_max_similarity,
                "discrimination_margin": discrimination_margin,
                "source_segment_count": int(candidate["source_segment_count"]),
                "recommend_score": float(recommend_score),
            }
        )
    rows.sort(key=lambda row: row["recommend_score"], reverse=True)
    return rows


def write_recommendation_tsv(*, rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(
            [
                "片段文件",
                "片段编号",
                "时长(秒)",
                "质量分",
                "同群相似度",
                "中心相似度",
                "平均负样本相似度",
                "最大负样本相似度",
                "区分度边际",
                "合并片段数",
                "推荐分",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["clip_filename"],
                    row["clip_id"],
                    f"{float(row['duration_sec']):.3f}",
                    f"{float(row['quality_score']):.4f}",
                    f"{float(row['cohort_similarity']):.4f}",
                    f"{float(row['center_similarity']):.4f}",
                    f"{float(row['unknown_avg_similarity']):.4f}",
                    f"{float(row['unknown_max_similarity']):.4f}",
                    f"{float(row['discrimination_margin']):.4f}",
                    row["source_segment_count"],
                    f"{float(row['recommend_score']):.4f}",
                ]
            )


def write_recommendation_markdown(
    *,
    speaker_id: str,
    top_k: int,
    rows: list[dict[str, object]],
    recommended_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    lines = [
        f"# {speaker_id} 注册样本建议",
        "",
        "## 一句话结论",
        "",
        f"- 当前共审计 `{len(rows)}` 条候选片段，建议优先使用前 `{top_k}` 条作为注册底库。",
        "",
        "## 推荐样本",
        "",
        "| 排名 | 片段文件 | 时长(秒) | 质量分 | 同群相似度 | 最大负样本相似度 | 区分度边际 | 推荐分 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(recommended_rows, start=1):
        lines.append(
            f"| {index} | {row['clip_filename']} | {float(row['duration_sec']):.3f} | "
            f"{float(row['quality_score']):.4f} | {float(row['cohort_similarity']):.4f} | "
            f"{float(row['unknown_max_similarity']):.4f} | {float(row['discrimination_margin']):.4f} | "
            f"{float(row['recommend_score']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 全量审计",
            "",
            "| 片段文件 | 时长(秒) | 质量分 | 同群相似度 | 中心相似度 | 最大负样本相似度 | 区分度边际 | 合并片段数 | 推荐分 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['clip_filename']} | {float(row['duration_sec']):.3f} | "
            f"{float(row['quality_score']):.4f} | {float(row['cohort_similarity']):.4f} | "
            f"{float(row['center_similarity']):.4f} | {float(row['unknown_max_similarity']):.4f} | "
            f"{float(row['discrimination_margin']):.4f} | {row['source_segment_count']} | "
            f"{float(row['recommend_score']):.4f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommendation_json(
    *,
    speaker_id: str,
    top_k: int,
    rows: list[dict[str, object]],
    recommended_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    payload = {
        "speaker_id": speaker_id,
        "top_k": top_k,
        "recommended_files": [row["clip_filename"] for row in recommended_rows],
        "rows": rows,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
