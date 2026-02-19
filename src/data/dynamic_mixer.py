from __future__ import annotations

from dataclasses import dataclass
from random import Random

import torch
import torchaudio
from torch.utils.data import Dataset

from src.config.audio import AudioConfig
from src.data.manifests import ManifestRecord
from src.data.preprocessing import (
    compute_stft_magnitude,
    peak_normalize,
    resample_if_needed,
)


@dataclass(frozen=True, slots=True)
class DenoiseBatchItem:
    noisy_mag: torch.Tensor
    clean_mag: torch.Tensor
    noisy_complex: torch.Tensor
    clean_waveform: torch.Tensor
    noisy_waveform: torch.Tensor
    snr_db: float


def _random_crop_1d(signal: torch.Tensor, target_length: int, rng: Random) -> torch.Tensor:
    if signal.ndim != 1:
        raise ValueError(f"Expected rank 1 signal for crop, got shape {signal.shape}.")
    if target_length <= 0:
        raise ValueError("target_length must be positive.")
    if signal.numel() >= target_length:
        max_offset = signal.numel() - target_length
        offset = rng.randint(0, max_offset) if max_offset > 0 else 0
        return signal[offset : offset + target_length]
    pad_len = target_length - signal.numel()
    return torch.nn.functional.pad(signal, (0, pad_len))


def _compute_noise_scale_for_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    target_snr_db: float,
) -> torch.Tensor:
    clean_power = torch.mean(clean.square())
    noise_power = torch.mean(noise.square())
    if clean_power <= 0:
        raise ValueError("Clean waveform has zero power, cannot mix.")
    if noise_power <= 0:
        raise ValueError("Noise waveform has zero power, cannot mix.")
    desired_noise_power = clean_power / (10.0 ** (target_snr_db / 10.0))
    scale = torch.sqrt(desired_noise_power / noise_power)
    return scale


class DynamicDenoiseDataset(Dataset[DenoiseBatchItem]):
    """On-the-fly clean+noise mixing dataset for speech denoising."""

    def __init__(
        self,
        speech_records: list[ManifestRecord],
        noise_records: list[ManifestRecord],
        audio_cfg: AudioConfig,
        split: str,
        seed: int = 17,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test.")
        audio_cfg.validate()
        self._audio_cfg = audio_cfg
        self._rng = Random(seed)
        self._speech = [record for record in speech_records if record.split == split]
        self._noise = [record for record in noise_records if record.split == split]
        self._power_epsilon = 1e-10
        self._max_mix_retries = 32
        self._invalid_noise_paths: set[str] = set()
        if not self._speech:
            raise ValueError(f"No speech records found for split={split}.")
        if not self._noise:
            raise ValueError(f"No noise records found for split={split}.")

    def __len__(self) -> int:
        return len(self._speech)

    def __getitem__(self, index: int) -> DenoiseBatchItem:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset size {len(self)}.")
        speech_record = self._speech[index]
        clean_waveform, clean_sr = torchaudio.load(speech_record.file_path)
        clean_waveform = resample_if_needed(
            clean_waveform,
            clean_sr,
            self._audio_cfg.sample_rate_hz,
        )
        clean_mono = peak_normalize(clean_waveform).mean(dim=0)
        clip_len = self._audio_cfg.clip_num_samples
        for _ in range(self._max_mix_retries):
            clean_clip = _random_crop_1d(clean_mono, clip_len, self._rng)
            if torch.mean(clean_clip.square()).item() <= self._power_epsilon:
                continue
            noise_record = self._noise[self._rng.randrange(0, len(self._noise))]
            if noise_record.file_path in self._invalid_noise_paths:
                continue
            noise_waveform, noise_sr = torchaudio.load(noise_record.file_path)
            noise_waveform = resample_if_needed(
                noise_waveform,
                noise_sr,
                self._audio_cfg.sample_rate_hz,
            )
            noise_mono = peak_normalize(noise_waveform).mean(dim=0)
            noise_clip = _random_crop_1d(noise_mono, clip_len, self._rng)
            if torch.mean(noise_clip.square()).item() <= self._power_epsilon:
                self._invalid_noise_paths.add(noise_record.file_path)
                continue
            snr_db = self._rng.uniform(self._audio_cfg.min_snr_db, self._audio_cfg.max_snr_db)
            noise_scale = _compute_noise_scale_for_snr(clean_clip, noise_clip, snr_db)
            noisy_clip = clean_clip + noise_scale * noise_clip
            clean_rank2 = clean_clip.unsqueeze(0)
            noisy_rank2 = noisy_clip.unsqueeze(0)
            clean_mag, _ = compute_stft_magnitude(clean_rank2, self._audio_cfg)
            noisy_mag, noisy_complex = compute_stft_magnitude(noisy_rank2, self._audio_cfg)
            return DenoiseBatchItem(
                noisy_mag=noisy_mag,
                clean_mag=clean_mag,
                noisy_complex=noisy_complex,
                clean_waveform=clean_rank2,
                noisy_waveform=noisy_rank2,
                snr_db=snr_db,
            )
        raise RuntimeError(
            "Failed to create a valid clean/noise mixture after retries. "
            f"Invalid noise clips seen: {len(self._invalid_noise_paths)} / {len(self._noise)}."
        )


def collate_denoise_batch(items: list[DenoiseBatchItem]) -> DenoiseBatchItem:
    if not items:
        raise ValueError("Cannot collate empty batch.")
    return DenoiseBatchItem(
        noisy_mag=torch.stack([item.noisy_mag for item in items], dim=0),
        clean_mag=torch.stack([item.clean_mag for item in items], dim=0),
        noisy_complex=torch.stack([item.noisy_complex for item in items], dim=0),
        clean_waveform=torch.stack([item.clean_waveform for item in items], dim=0),
        noisy_waveform=torch.stack([item.noisy_waveform for item in items], dim=0),
        snr_db=sum(item.snr_db for item in items) / float(len(items)),
    )

