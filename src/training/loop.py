from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config.audio import AudioConfig
from src.config.experiment import DataPaths, TrackingConfig, TrainingConfig
from src.data.acquisition import build_noise_manifest, build_speech_manifest
from src.data.dynamic_mixer import DynamicDenoiseDataset, collate_denoise_batch
from src.models.base import BaseDenoiseModel
from src.training.trainer import DenoiseTrainer, EpochStats
from src.utils.seeding import set_global_seed
from src.utils.tracking import MlflowTracker


def _resolve_training_device() -> torch.device:
    require_cuda = os.getenv("REQUIRE_CUDA", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
    has_cuda = torch.cuda.is_available()
    if require_cuda and not has_cuda:
        raise RuntimeError(
            "CUDA is required but unavailable. Set REQUIRE_CUDA=0 to allow CPU fallback."
        )
    return torch.device("cuda" if has_cuda else "cpu")


def _describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    index = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    return f"cuda:{index} ({name}, sm_{capability[0]}{capability[1]})"


def _build_loaders(
    data_paths: DataPaths,
    audio_cfg: AudioConfig,
    train_cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    speech_records = build_speech_manifest(data_paths.librispeech_root)
    noise_records = build_noise_manifest(data_paths.audioset_root)
    train_ds = DynamicDenoiseDataset(
        speech_records=speech_records,
        noise_records=noise_records,
        audio_cfg=audio_cfg,
        split="train",
        seed=train_cfg.seed,
    )
    val_ds = DynamicDenoiseDataset(
        speech_records=speech_records,
        noise_records=noise_records,
        audio_cfg=audio_cfg,
        split="val",
        seed=train_cfg.seed + 1,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_denoise_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_denoise_batch,
    )
    return train_loader, val_loader


def train_model(
    model: BaseDenoiseModel,
    run_name: str,
    output_dir: Path,
    data_paths: DataPaths,
    audio_cfg: AudioConfig,
    train_cfg: TrainingConfig,
    tracking_cfg: TrackingConfig,
) -> list[EpochStats]:
    set_global_seed(train_cfg.seed)
    device = _resolve_training_device()
    print(f"[train] using device: {_describe_device(device)}")
    print(f"[train] require_cuda={os.getenv('REQUIRE_CUDA', '0')}")
    tracker = MlflowTracker(
        tracking_uri=tracking_cfg.mlflow_tracking_uri,
        experiment_name=tracking_cfg.experiment_name,
    )
    tracker.start_run(run_name=run_name)
    tracker.log_runtime_metadata(seed=train_cfg.seed)
    tracker.log_config_dataclass("audio", audio_cfg)
    tracker.log_config_dataclass("training", train_cfg)
    tracker.log_config_dataclass("tracking", tracking_cfg)
    train_loader, val_loader = _build_loaders(
        data_paths=data_paths,
        audio_cfg=audio_cfg,
        train_cfg=train_cfg,
    )
    trainer = DenoiseTrainer(
        model=model,
        audio_cfg=audio_cfg,
        train_cfg=train_cfg,
        tracker=tracker,
        output_dir=output_dir,
        device=device,
    )
    history = trainer.train(train_loader=train_loader, val_loader=val_loader)
    tracker.end_run()
    return history

