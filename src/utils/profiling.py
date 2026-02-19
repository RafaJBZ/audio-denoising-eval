from __future__ import annotations

import statistics
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import torch


class InferenceCallable(Protocol):
    def __call__(self, batch: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class LatencyStats:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values cannot be empty.")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1].")
    sorted_values = sorted(values)
    idx = int(q * (len(sorted_values) - 1))
    return sorted_values[idx]


def benchmark_latency_ms(
    fn: InferenceCallable,
    input_tensor: torch.Tensor,
    warmup_steps: int = 20,
    timed_steps: int = 100,
) -> LatencyStats:
    if warmup_steps < 0 or timed_steps <= 0:
        raise ValueError("warmup_steps must be >= 0 and timed_steps must be > 0.")
    for _ in range(warmup_steps):
        _ = fn(input_tensor)
    timings: list[float] = []
    for _ in range(timed_steps):
        start = perf_counter()
        _ = fn(input_tensor)
        if input_tensor.is_cuda:
            torch.cuda.synchronize(device=input_tensor.device)
        elapsed_ms = (perf_counter() - start) * 1000.0
        timings.append(elapsed_ms)
    return LatencyStats(
        p50_ms=percentile(timings, 0.50),
        p95_ms=percentile(timings, 0.95),
        p99_ms=percentile(timings, 0.99),
        mean_ms=statistics.fmean(timings),
        std_ms=statistics.pstdev(timings),
    )

