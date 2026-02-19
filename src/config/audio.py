from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Global audio and transform settings used across the project."""

    sample_rate_hz: int = 16_000
    clip_duration_sec: float = 2.0
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    normalized_stft: bool = False
    center: bool = True
    min_snr_db: float = -5.0
    max_snr_db: float = 20.0

    def validate(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        if self.clip_duration_sec <= 0.0:
            raise ValueError("clip_duration_sec must be positive.")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive.")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive.")
        if self.win_length <= 0:
            raise ValueError("win_length must be positive.")
        if self.win_length > self.n_fft:
            raise ValueError("win_length cannot be larger than n_fft.")
        if self.max_snr_db <= self.min_snr_db:
            raise ValueError("max_snr_db must be greater than min_snr_db.")

    @property
    def clip_num_samples(self) -> int:
        self.validate()
        return int(self.clip_duration_sec * self.sample_rate_hz)

