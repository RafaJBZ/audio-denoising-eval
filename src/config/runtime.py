from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.config.audio import AudioConfig
from src.config.experiment import DataPaths


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    data_paths: DataPaths
    output_root: Path


def _load_dotenv_file(dotenv_path: Path = Path(".env")) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "" or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"Invalid .env entry: {line}")
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if key == "":
            raise RuntimeError(f"Invalid .env key in line: {line}")
        os.environ.setdefault(key, value)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer value for {name}: {raw}") from exc


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"Invalid float value for {name}: {raw}") from exc


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean value for {name}: {raw}")


def load_audio_config() -> AudioConfig:
    _load_dotenv_file()
    base = AudioConfig()
    cfg = AudioConfig(
        sample_rate_hz=_get_env_int("SAMPLE_RATE_HZ", base.sample_rate_hz),
        clip_duration_sec=_get_env_float("CLIP_DURATION_SEC", base.clip_duration_sec),
        n_fft=_get_env_int("N_FFT", base.n_fft),
        hop_length=_get_env_int("HOP_LENGTH", base.hop_length),
        win_length=_get_env_int("WIN_LENGTH", base.win_length),
        normalized_stft=_get_env_bool("STFT_NORMALIZED", base.normalized_stft),
        center=_get_env_bool("STFT_CENTER", base.center),
        min_snr_db=_get_env_float("MIN_SNR_DB", base.min_snr_db),
        max_snr_db=_get_env_float("MAX_SNR_DB", base.max_snr_db),
    )
    cfg.validate()
    return cfg


def load_runtime_paths() -> RuntimePaths:
    _load_dotenv_file()
    librispeech_root = Path(_require_env("LIBRISPEECH_ROOT"))
    audioset_root = Path(_require_env("AUDIOSET_ROOT"))
    manifest_dir = Path(os.getenv("MANIFEST_DIR", "data/manifests"))
    output_root = Path(os.getenv("OUTPUT_ROOT", "experiments/runs"))
    data_paths = DataPaths(
        librispeech_root=librispeech_root,
        audioset_root=audioset_root,
        manifest_dir=manifest_dir,
    )
    data_paths.validate()
    return RuntimePaths(data_paths=data_paths, output_root=output_root)

