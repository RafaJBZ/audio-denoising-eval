from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    pesq: float
    stoi: float
    si_sdr: float


def _validate_waveforms(reference: torch.Tensor, estimate: torch.Tensor) -> None:
    if reference.shape != estimate.shape:
        raise ValueError(f"Waveform shapes must match, got {reference.shape} vs {estimate.shape}.")
    if reference.ndim not in {1, 2}:
        raise ValueError("Waveforms must have shape [samples] or [batch, samples].")


def compute_si_sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> float:
    _validate_waveforms(reference, estimate)
    ref = reference.reshape(-1).float()
    est = estimate.reshape(-1).float()
    scale = torch.dot(est, ref) / (torch.dot(ref, ref) + eps)
    projection = scale * ref
    noise = est - projection
    ratio = torch.sum(projection.square()) / (torch.sum(noise.square()) + eps)
    return float(10.0 * torch.log10(ratio + eps).item())


def compute_pesq(reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int) -> float:
    _validate_waveforms(reference, estimate)
    if sample_rate not in {8000, 16000}:
        raise ValueError("PESQ expects sample_rate in {8000, 16000}.")
    try:
        from pesq import pesq
    except ImportError as exc:
        raise RuntimeError("PESQ dependency missing. Install with: uv add pesq") from exc
    ref = reference.reshape(-1).detach().cpu().numpy()
    est = estimate.reshape(-1).detach().cpu().numpy()
    mode = "wb" if sample_rate == 16000 else "nb"
    return float(pesq(sample_rate, ref, est, mode))


def compute_stoi(reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int) -> float:
    _validate_waveforms(reference, estimate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    try:
        from pystoi import stoi
    except ImportError as exc:
        raise RuntimeError("STOI dependency missing. Install with: uv add pystoi") from exc
    ref = reference.reshape(-1).detach().cpu().numpy()
    est = estimate.reshape(-1).detach().cpu().numpy()
    return float(stoi(ref, est, sample_rate, extended=False))


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def safe_quality_metrics(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    sample_rate: int,
) -> QualityMetrics:
    si_sdr = compute_si_sdr(reference, estimate)
    pesq = compute_pesq(reference, estimate, sample_rate=sample_rate)
    stoi = compute_stoi(reference, estimate, sample_rate=sample_rate)
    if not all(math.isfinite(value) for value in (si_sdr, pesq, stoi)):
        raise RuntimeError("Encountered non-finite metric result.")
    return QualityMetrics(pesq=pesq, stoi=stoi, si_sdr=si_sdr)

