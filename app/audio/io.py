"""Audio loading and normalization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from app.audio.models import AudioChunk
from app.common.constants import DEFAULT_SAMPLE_RATE
from app.common.errors import InvalidAudioInputError


def load_audio_chunk(
    audio_path: Path,
    *,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> AudioChunk:
    """Load an audio file and normalize it into mono float32 waveform."""

    if not audio_path.exists():
        raise InvalidAudioInputError(f"Audio file does not exist: {audio_path}")

    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise InvalidAudioInputError(f"Audio file is empty: {audio_path}")

    waveform = np.mean(audio, axis=1, dtype=np.float32)
    waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
    if sample_rate != target_sample_rate:
        waveform_tensor = torchaudio.functional.resample(
            waveform_tensor,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        sample_rate = target_sample_rate

    normalized_waveform = waveform_tensor.squeeze(0).numpy().astype(np.float32)
    duration_sec = float(normalized_waveform.shape[0] / sample_rate)
    rms = float(np.sqrt(np.mean(np.square(normalized_waveform)))) if normalized_waveform.size else 0.0
    peak = float(np.max(np.abs(normalized_waveform))) if normalized_waveform.size else 0.0
    return AudioChunk(
        waveform=normalized_waveform,
        sample_rate=sample_rate,
        source_path=audio_path,
        duration_sec=duration_sec,
        metadata={
            "loader": "soundfile+torchaudio",
            "rms": rms,
            "peak": peak,
        },
    )
