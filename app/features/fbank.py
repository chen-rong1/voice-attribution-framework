"""Fbank feature extraction helpers for ECAPA-style backends."""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

from app.audio.models import AudioChunk
from app.common.constants import DEFAULT_NUM_MEL_BINS, DEFAULT_FEATURE_VERSION
from app.features.models import FeatureMatrix


def compute_fbank(
    audio_chunk: AudioChunk,
    *,
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS,
    feature_version: str = DEFAULT_FEATURE_VERSION,
) -> FeatureMatrix:
    """Compute CMN-normalized Kaldi fbank features for one audio chunk."""

    waveform = torch.from_numpy(audio_chunk.waveform).unsqueeze(0)
    waveform = waveform * (1 << 15)
    feats = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=audio_chunk.sample_rate,
        window_type="hamming",
        use_energy=False,
    )
    feats = feats - torch.mean(feats, dim=0, keepdim=True)
    values = feats.unsqueeze(0).numpy().astype(np.float32)
    return FeatureMatrix(
        values=values,
        feature_version=feature_version,
        num_frames=int(feats.shape[0]),
        metadata={
            "num_mel_bins": num_mel_bins,
            "sample_rate": audio_chunk.sample_rate,
        },
    )
