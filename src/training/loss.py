from __future__ import annotations

import torch
from torch import nn


class SpectralL1Loss(nn.Module):
    """Simple and stable baseline loss on magnitude spectra."""

    def __init__(self) -> None:
        super().__init__()
        self._loss = nn.L1Loss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target must match shapes, "
                f"got {prediction.shape} vs {target.shape}."
            )
        return self._loss(prediction, target)

