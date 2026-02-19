from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ModelOutput:
    denoised_mag: torch.Tensor
    mask: torch.Tensor
    attention_map: torch.Tensor | None = None


class BaseDenoiseModel(nn.Module, ABC):
    """Strict base model contract for fair cross-model evaluation."""

    @abstractmethod
    def forward(self, noisy_mag: torch.Tensor) -> ModelOutput:
        """
        Args:
            noisy_mag: [batch, channels=1, freq_bins, time_frames]
        """
        raise NotImplementedError

    def inference_step(self, noisy_mag: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(noisy_mag).denoised_mag

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @staticmethod
    def validate_input(noisy_mag: torch.Tensor) -> None:
        if noisy_mag.ndim != 4:
            raise ValueError(
                "Expected noisy_mag shape [batch, channels, freq, time], "
                f"got {tuple(noisy_mag.shape)}."
            )
        if noisy_mag.shape[1] != 1:
            raise ValueError(f"Expected channels=1 for comparability, got {noisy_mag.shape[1]}.")

