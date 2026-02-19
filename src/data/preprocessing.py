from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio

from src.config.audio import AudioConfig


@dataclass(frozen=True, slots=True)
class SpectrogramBatch:
    noisy_mag: torch.Tensor
    clean_mag: torch.Tensor
    noisy_complex: torch.Tensor
    clean_waveform: torch.Tensor
    noisy_waveform: torch.Tensor


def _hann_window(win_length: int, reference: torch.Tensor) -> torch.Tensor:
    if win_length <= 0:
        raise ValueError("win_length must be positive.")
    base = reference.real if torch.is_complex(reference) else reference
    return torch.hann_window(
        win_length,
        device=base.device,
        dtype=base.dtype,
    )


def resample_if_needed(waveform: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    if source_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive.")
    if source_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=target_sr)
    return resampler(waveform)


def peak_normalize(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform rank 2 [channels, samples], got {waveform.shape}.")
    max_abs = waveform.abs().max()
    scale = torch.clamp(max_abs, min=eps)
    return waveform / scale


def compute_stft_magnitude(
    waveform: torch.Tensor,
    audio_cfg: AudioConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    audio_cfg.validate()
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform rank 2 [channels, samples], got {waveform.shape}.")
    mono = waveform.mean(dim=0)
    window = _hann_window(audio_cfg.win_length, mono)
    complex_spec = torch.stft(
        mono,
        n_fft=audio_cfg.n_fft,
        hop_length=audio_cfg.hop_length,
        win_length=audio_cfg.win_length,
        window=window,
        return_complex=True,
        normalized=audio_cfg.normalized_stft,
        center=audio_cfg.center,
    )
    magnitude = complex_spec.abs().unsqueeze(0)
    return magnitude, complex_spec


def invert_stft(complex_spec: torch.Tensor, audio_cfg: AudioConfig) -> torch.Tensor:
    if complex_spec.ndim != 2:
        raise ValueError(f"Expected complex STFT rank 2 [freq, time], got {complex_spec.shape}.")
    window = _hann_window(audio_cfg.win_length, complex_spec.real)
    return torch.istft(
        complex_spec,
        n_fft=audio_cfg.n_fft,
        hop_length=audio_cfg.hop_length,
        win_length=audio_cfg.win_length,
        window=window,
        normalized=audio_cfg.normalized_stft,
        center=audio_cfg.center,
    )

