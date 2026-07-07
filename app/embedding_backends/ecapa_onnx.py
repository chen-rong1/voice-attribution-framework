"""Generic ECAPA-TDNN ONNX backend implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.audio.io import load_audio_chunk
from app.audio.quality import estimate_audio_quality_score
from app.common.constants import DEFAULT_FEATURE_VERSION
from app.common.errors import EmbeddingExtractionError
from app.embedding_backends.base import EmbeddingBackend
from app.embedding_backends.models import EmbeddingRequest, EmbeddingResult
from app.features.fbank import compute_fbank


class EcapaTdnnOnnxBackend(EmbeddingBackend):
    """A generic ONNX backend for ECAPA-TDNN models that consume fbank features."""

    def __init__(
        self,
        *,
        backend_name: str,
        model_dir: Path,
        backend_version: str = "0.1.0",
        feature_version: str = DEFAULT_FEATURE_VERSION,
        embedding_dim: int = 0,
    ) -> None:
        super().__init__()
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.feature_version = feature_version
        self.embedding_dim = embedding_dim
        self.model_dir = model_dir
        self._session: ort.InferenceSession | None = None
        self._input_name = "feats"
        self._output_name = "embs"

    @property
    def onnx_path(self) -> Path:
        return self.model_dir / "avg_model.onnx"

    def load(self) -> None:
        if self._loaded:
            return
        if not self.onnx_path.exists():
            raise EmbeddingExtractionError(f"ONNX model does not exist: {self.onnx_path}")
        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._loaded = True

    def unload(self) -> None:
        self._session = None
        super().unload()

    def extract_embedding(self, request: EmbeddingRequest) -> EmbeddingResult:
        self.ensure_loaded()
        if request.features is not None:
            features = request.features
            duration_sec = None
        elif request.audio is not None:
            features = compute_fbank(
                request.audio,
                feature_version=self.feature_version,
            )
            duration_sec = request.audio.duration_sec
            rms = float(request.audio.metadata.get("rms", 0.0))
        elif "audio_path" in request.metadata:
            audio_chunk = load_audio_chunk(Path(str(request.metadata["audio_path"])))
            features = compute_fbank(
                audio_chunk,
                feature_version=self.feature_version,
            )
            duration_sec = audio_chunk.duration_sec
            rms = float(audio_chunk.metadata.get("rms", 0.0))
        else:
            raise EmbeddingExtractionError(
                "EmbeddingRequest requires features, audio, or metadata['audio_path']."
            )
        if request.features is not None:
            rms = None

        if self._session is None:
            raise EmbeddingExtractionError(f"ONNX session for `{self.backend_name}` is not ready.")

        output = self._session.run(
            output_names=[self._output_name],
            input_feed={self._input_name: features.values},
        )[0]
        embedding = np.asarray(output, dtype=np.float32).reshape(-1)
        self.embedding_dim = int(embedding.shape[0])
        return EmbeddingResult(
            sample_id=request.sample_id,
            backend_name=self.backend_name,
            backend_version=self.backend_version,
            feature_version=features.feature_version,
            embedding=embedding,
            embedding_dim=self.embedding_dim,
            duration_sec=duration_sec,
            quality_score=estimate_audio_quality_score(duration_sec=duration_sec, rms=rms),
            metadata={"model_dir": str(self.model_dir)},
        )
