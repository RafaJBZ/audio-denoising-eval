from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

import torch
import torchaudio

from src.config.audio import AudioConfig
from src.data.manifests import ManifestRecord, read_manifest_csv
from src.data.preprocessing import peak_normalize, resample_if_needed


@dataclass(frozen=True, slots=True)
class EvalSampleMetadata:
    sample_id: str
    split: str
    clean_path: str
    noisy_path: str
    clean_source: str
    noise_source: str
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
    return torch.sqrt(desired_noise_power / noise_power)


def _load_mono_waveform(record: ManifestRecord, target_sr: int) -> torch.Tensor:
    waveform, source_sr = torchaudio.load(record.file_path)
    waveform = resample_if_needed(waveform, source_sr, target_sr)
    waveform = peak_normalize(waveform).mean(dim=0)
    return waveform


def _filter_records(
    records: list[ManifestRecord],
    split: str,
    domain_label: str,
) -> list[ManifestRecord]:
    filtered = [
        record
        for record in records
        if record.split == split and record.domain_label == domain_label
    ]
    if not filtered:
        raise ValueError(f"No records found for split={split}, domain_label={domain_label}.")
    return filtered


def generate_fixed_eval_set(
    manifest_csv_path: Path,
    output_dir: Path,
    audio_cfg: AudioConfig,
    num_samples: int,
    split: str = "test",
    seed: int = 17,
) -> list[EvalSampleMetadata]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    audio_cfg.validate()
    records = read_manifest_csv(manifest_csv_path)
    speech_records = _filter_records(records, split=split, domain_label="speech")
    noise_records = _filter_records(records, split=split, domain_label="noise")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = Random(seed)
    metadata: list[EvalSampleMetadata] = []

    for idx in range(num_samples):
        speech_record = speech_records[idx % len(speech_records)]
        noise_record = noise_records[rng.randrange(0, len(noise_records))]
        clean_mono = _load_mono_waveform(speech_record, target_sr=audio_cfg.sample_rate_hz)
        noise_mono = _load_mono_waveform(noise_record, target_sr=audio_cfg.sample_rate_hz)
        clip_len = audio_cfg.clip_num_samples
        clean_clip = _random_crop_1d(clean_mono, clip_len, rng)
        noise_clip = _random_crop_1d(noise_mono, clip_len, rng)
        snr_db = rng.uniform(audio_cfg.min_snr_db, audio_cfg.max_snr_db)
        noise_scale = _compute_noise_scale_for_snr(clean_clip, noise_clip, snr_db)
        noisy_clip = clean_clip + noise_scale * noise_clip

        sample_id = f"sample_{idx:05d}"
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        clean_path = sample_dir / "clean.wav"
        noisy_path = sample_dir / "noisy.wav"
        torchaudio.save(
            clean_path.as_posix(),
            clean_clip.unsqueeze(0),
            sample_rate=audio_cfg.sample_rate_hz,
        )
        torchaudio.save(
            noisy_path.as_posix(),
            noisy_clip.unsqueeze(0),
            sample_rate=audio_cfg.sample_rate_hz,
        )
        metadata.append(
            EvalSampleMetadata(
                sample_id=sample_id,
                split=split,
                clean_path=clean_path.as_posix(),
                noisy_path=noisy_path.as_posix(),
                clean_source=speech_record.file_path,
                noise_source=noise_record.file_path,
                snr_db=snr_db,
            )
        )

    manifest_path = output_dir / "eval_set_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as out_file:
        for row in metadata:
            out_file.write(json.dumps(asdict(row)))
            out_file.write("\n")
    return metadata

