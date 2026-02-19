from __future__ import annotations

import csv
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

LIBRISPEECH_TRAIN_CLEAN_100_URL = (
    "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
)

AUDIOSET_BALANCED_CSV_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
)
AUDIOSET_EVAL_CSV_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
)

# Environmental and non-stationary labels from AudioSet ontology.
DEFAULT_NOISE_LABEL_IDS: tuple[str, ...] = (
    "/m/0jbk",  # Animal
    "/m/07yv9",  # Vehicle
    "/m/01j3sz",  # Wind
    "/m/06mb1",  # Rain
    "/m/07r04",  # Thunderstorm
    "/m/09x0r",  # Fire
    "/m/01b_21",  # Water
    "/m/01hsr_",  # Crowd
    "/m/0dgw9r",  # Traffic noise
    "/m/0f8s22",  # Siren
)


@dataclass(frozen=True, slots=True)
class AudioSetSegment:
    ytid: str
    start_seconds: float
    end_seconds: float
    labels: tuple[str, ...]


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    urlretrieve(url, destination.as_posix())


def download_librispeech_train_clean_100(librispeech_root: Path) -> Path:
    librispeech_root.mkdir(parents=True, exist_ok=True)
    archive_path = librispeech_root / "train-clean-100.tar.gz"
    _download_file(LIBRISPEECH_TRAIN_CLEAN_100_URL, archive_path)
    target_dir = librispeech_root / "LibriSpeech" / "train-clean-100"
    if target_dir.exists():
        return target_dir
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive.extractall(path=librispeech_root)
    if not target_dir.exists():
        raise RuntimeError(f"Expected extracted LibriSpeech path not found: {target_dir}")
    return target_dir


def download_audioset_metadata(audioset_root: Path) -> tuple[Path, Path]:
    metadata_dir = audioset_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    balanced_csv = metadata_dir / "balanced_train_segments.csv"
    eval_csv = metadata_dir / "eval_segments.csv"
    _download_file(AUDIOSET_BALANCED_CSV_URL, balanced_csv)
    _download_file(AUDIOSET_EVAL_CSV_URL, eval_csv)
    return balanced_csv, eval_csv


def _parse_audioset_csv(csv_path: Path) -> list[AudioSetSegment]:
    if not csv_path.exists():
        raise FileNotFoundError(f"AudioSet CSV does not exist: {csv_path}")
    segments: list[AudioSetSegment] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        for raw_line in csv_file:
            if raw_line.startswith("#") or raw_line.strip() == "":
                continue
            fields = next(csv.reader([raw_line]))
            if len(fields) < 4:
                continue
            ytid = fields[0].strip()
            start_seconds = float(fields[1].strip())
            end_seconds = float(fields[2].strip())
            labels = tuple(label.strip() for label in fields[3].strip().strip('"').split(","))
            segments.append(
                AudioSetSegment(
                    ytid=ytid,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    labels=labels,
                )
            )
    return segments


def filter_segments_by_labels(
    segments: list[AudioSetSegment],
    target_labels: tuple[str, ...] = DEFAULT_NOISE_LABEL_IDS,
) -> list[AudioSetSegment]:
    target_set = set(target_labels)
    return [segment for segment in segments if any(label in target_set for label in segment.labels)]


def _download_single_clip(
    segment: AudioSetSegment,
    output_dir: Path,
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_name = f"{segment.ytid}_{int(segment.start_seconds)}_{int(segment.end_seconds)}"
    target_wav = output_dir / f"{clip_name}.wav"
    if target_wav.exists():
        return True
    clip_url = f"https://www.youtube.com/watch?v={segment.ytid}"
    output_template = (output_dir / f"{clip_name}.%(ext)s").as_posix()
    command = [
        "yt-dlp",
        "-q",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--download-sections",
        f"*{segment.start_seconds:.3f}-{segment.end_seconds:.3f}",
        "-o",
        output_template,
        clip_url,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.returncode == 0 and target_wav.exists()


def download_audioset_noise_subset(
    audioset_root: Path,
    max_clips: int = 200,
    target_labels: tuple[str, ...] = DEFAULT_NOISE_LABEL_IDS,
) -> Path:
    if max_clips <= 0:
        raise ValueError("max_clips must be positive.")
    balanced_csv, eval_csv = download_audioset_metadata(audioset_root)
    all_segments = _parse_audioset_csv(balanced_csv) + _parse_audioset_csv(eval_csv)
    filtered = filter_segments_by_labels(all_segments, target_labels=target_labels)
    output_dir = audioset_root / "clips"
    downloaded = 0
    for segment in filtered:
        if downloaded >= max_clips:
            break
        if _download_single_clip(segment, output_dir=output_dir):
            downloaded += 1
    if downloaded == 0:
        raise RuntimeError(
            "No AudioSet clips downloaded. Install ffmpeg + yt-dlp and verify network access."
        )
    return output_dir

