from __future__ import annotations

import hashlib
from pathlib import Path

import torchaudio

from src.data.manifests import ManifestRecord, write_manifest_csv, write_manifest_jsonl

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".wav", ".flac", ".mp3")


def _list_audio_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _duration_seconds(path: Path) -> tuple[float, int]:
    waveform, sample_rate = torchaudio.load(path.as_posix())
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in audio metadata: {path}")
    if waveform.numel() <= 0:
        raise ValueError(f"Invalid frame count in audio metadata: {path}")
    num_frames = waveform.shape[-1]
    return num_frames / sample_rate, sample_rate


def _default_split_from_hash(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _noise_source_id(path: Path) -> str:
    # For downloaded AudioSet clips we usually have names like:
    # <ytid>_<start>_<end>.wav inside a shared "clips" directory.
    if path.parent.name == "clips":
        stem_parts = path.stem.split("_")
        if stem_parts and stem_parts[0].strip() != "":
            return stem_parts[0]
    if path.parent.name.strip() != "":
        return path.parent.name
    if path.stem.strip() != "":
        return path.stem
    return "unknown_noise_source"


def build_speech_manifest(librispeech_root: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for path in _list_audio_files(librispeech_root):
        duration_sec, sample_rate = _duration_seconds(path)
        # LibriSpeech folders are generally speaker/chapter/file
        speaker_id = path.parent.parent.name if path.parent.parent.name != "" else "unknown_speaker"
        split = _default_split_from_hash(speaker_id)
        records.append(
            ManifestRecord(
                file_path=path.as_posix(),
                duration_sec=duration_sec,
                sample_rate=sample_rate,
                split=split,
                domain_label="speech",
                speaker_or_noise_id=speaker_id,
            )
        )
    return records


def build_noise_manifest(audioset_root: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    for path in _list_audio_files(audioset_root):
        duration_sec, sample_rate = _duration_seconds(path)
        source_id = _noise_source_id(path)
        split = _default_split_from_hash(source_id)
        records.append(
            ManifestRecord(
                file_path=path.as_posix(),
                duration_sec=duration_sec,
                sample_rate=sample_rate,
                split=split,
                domain_label="noise",
                speaker_or_noise_id=source_id,
            )
        )
    return records


def write_manifests(
    speech_records: list[ManifestRecord],
    noise_records: list[ManifestRecord],
    output_dir: Path,
) -> None:
    if output_dir.as_posix().strip() == "":
        raise ValueError("output_dir cannot be empty.")
    all_records = speech_records + noise_records
    csv_path = output_dir / "dataset_manifest.csv"
    jsonl_path = output_dir / "dataset_manifest.jsonl"
    write_manifest_csv(all_records, csv_path)
    write_manifest_jsonl(all_records, jsonl_path)

