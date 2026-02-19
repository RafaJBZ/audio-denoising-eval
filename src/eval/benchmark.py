from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from src.models.base import BaseDenoiseModel
from src.utils.profiling import LatencyStats, benchmark_latency_ms


@dataclass(frozen=True, slots=True)
class TrainingCost:
    total_seconds: float
    epochs: int
    seconds_per_epoch: float


def summarize_training_cost(epoch_seconds: list[float]) -> TrainingCost:
    if not epoch_seconds:
        raise ValueError("epoch_seconds cannot be empty.")
    total = sum(epoch_seconds)
    epochs = len(epoch_seconds)
    return TrainingCost(total_seconds=total, epochs=epochs, seconds_per_epoch=total / float(epochs))


def model_latency(
    model: BaseDenoiseModel,
    batch_size: int,
    freq_bins: int,
    time_frames: int,
    device: torch.device,
    warmup_steps: int = 30,
    timed_steps: int = 200,
) -> LatencyStats:
    if batch_size <= 0 or freq_bins <= 0 or time_frames <= 0:
        raise ValueError("batch_size, freq_bins, and time_frames must be positive.")
    input_tensor = torch.randn(batch_size, 1, freq_bins, time_frames, device=device)
    model = model.to(device=device).eval()

    def _infer(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(batch).denoised_mag

    return benchmark_latency_ms(
        fn=_infer,
        input_tensor=input_tensor,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
    )


def convergence_epoch(loss_curve: list[float], threshold: float) -> int:
    if not loss_curve:
        raise ValueError("loss_curve cannot be empty.")
    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    for epoch_idx, loss_value in enumerate(loss_curve, start=1):
        if loss_value <= threshold:
            return epoch_idx
    return len(loss_curve)


def timed_forward_seconds(
    model: BaseDenoiseModel,
    batch: torch.Tensor,
    device: torch.device,
) -> float:
    model = model.to(device=device).eval()
    batch = batch.to(device=device)
    start = perf_counter()
    with torch.no_grad():
        _ = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    return perf_counter() - start

