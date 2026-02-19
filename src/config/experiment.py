from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DataPaths:
    librispeech_root: Path
    audioset_root: Path
    manifest_dir: Path

    def validate(self) -> None:
        for name, path_value in (
            ("librispeech_root", self.librispeech_root),
            ("audioset_root", self.audioset_root),
            ("manifest_dir", self.manifest_dir),
        ):
            if path_value.as_posix().strip() == "":
                raise ValueError(f"{name} cannot be empty.")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 16
    num_workers: int = 4
    max_epochs: int = 80
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0
    early_stopping_patience: int = 8
    use_compile: bool = True
    use_bfloat16: bool = True
    seed: int = 17

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    mlflow_tracking_uri: str = "file:./experiments/mlruns"
    experiment_name: str = "audio-denoising-comparison"

    def validate(self) -> None:
        if self.mlflow_tracking_uri.strip() == "":
            raise ValueError("mlflow_tracking_uri cannot be empty.")
        if self.experiment_name.strip() == "":
            raise ValueError("experiment_name cannot be empty.")

