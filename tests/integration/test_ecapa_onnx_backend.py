from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio.io import load_audio_chunk
from app.embedding_backends.ecapa_onnx import EcapaTdnnOnnxBackend
from app.embedding_backends.models import EmbeddingRequest


def test_ecapa_tdnn_onnx_backend_can_extract_embedding(tmp_path: Path) -> None:
    model_dir = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "ecapa_tdnn"
        / "wespeaker_ecapa1024_lm"
    )
    assert (model_dir / "avg_model.onnx").exists()

    sample_rate = 16000
    duration_sec = 1.5
    timeline = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = 0.2 * np.sin(2 * np.pi * 220 * timeline).astype(np.float32)
    audio_path = tmp_path / "synthetic.wav"
    sf.write(audio_path, waveform, sample_rate)

    backend = EcapaTdnnOnnxBackend(
        backend_name="wespeaker-ecapa1024-lm-onnx",
        model_dir=model_dir,
    )
    backend.load()
    audio_chunk = load_audio_chunk(audio_path)
    result = backend.extract_embedding(
        EmbeddingRequest(sample_id="synthetic", audio=audio_chunk)
    )

    assert result.backend_name == "wespeaker-ecapa1024-lm-onnx"
    assert result.embedding_dim > 0
    assert result.embedding.shape[0] == result.embedding_dim
