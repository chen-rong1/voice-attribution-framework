"""Framework bootstrap helpers for default backend wiring."""

from __future__ import annotations

from pathlib import Path

from app.common.config import load_simple_yaml_map
from app.embedding_backends.ecapa_onnx import EcapaTdnnOnnxBackend
from app.embedding_backends.registry import EmbeddingBackendRegistry


def register_default_backends(
    *,
    project_root: Path,
    registry: EmbeddingBackendRegistry,
    model_config_path: Path | None = None,
) -> EmbeddingBackendRegistry:
    """Register the default bootstrap backend declared in the model config."""

    resolved_config_path = model_config_path or project_root / "configs" / "models" / "default.yaml"
    config = load_simple_yaml_map(resolved_config_path)
    backend_name = config["bootstrap_backend"]
    model_dir = project_root / config["bootstrap_model_dir"]
    feature_version = config.get("feature_version", "fbank80_v1")
    registry.register(
        EcapaTdnnOnnxBackend(
            backend_name=backend_name,
            model_dir=model_dir,
            feature_version=feature_version,
        )
    )
    return registry
