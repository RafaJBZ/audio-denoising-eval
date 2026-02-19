from __future__ import annotations

import torch
from torch import nn

from src.config.model import EfficientDfConfig
from src.models.base import BaseDenoiseModel, ModelOutput


class EfficientDeepFilterNetDenoiser(BaseDenoiseModel):
    """
    Native low-complexity DeepFilterNet-inspired model.

    This implementation focuses on a compact parameter footprint and low-latency blocks.
    """

    def __init__(self, config: EfficientDfConfig) -> None:
        super().__init__()
        config.validate()
        pad = config.conv_kernel_size // 2
        self.frontend = nn.Sequential(
            nn.Conv2d(
                1,
                config.bottleneck_channels,
                kernel_size=config.conv_kernel_size,
                padding=pad,
            ),
            nn.GELU(),
            nn.Conv2d(
                config.bottleneck_channels,
                config.bottleneck_channels,
                kernel_size=config.conv_kernel_size,
                padding=pad,
                groups=config.bottleneck_channels,
            ),
            nn.GELU(),
            nn.Conv2d(config.bottleneck_channels, config.bottleneck_channels, kernel_size=1),
            nn.GELU(),
        )
        self.rnn = nn.GRU(
            input_size=config.bottleneck_channels,
            hidden_size=config.recurrent_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.proj = nn.Linear(config.recurrent_hidden, config.bottleneck_channels)
        self.mask_head = nn.Sequential(
            nn.Conv2d(config.bottleneck_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, noisy_mag: torch.Tensor) -> ModelOutput:
        self.validate_input(noisy_mag)
        features = self.frontend(noisy_mag)
        batch, channels, freq, time = features.shape
        seq = features.permute(0, 2, 3, 1).reshape(batch * freq, time, channels)
        recurrent_out, _ = self.rnn(seq)
        projected = self.proj(recurrent_out)
        restored = projected.reshape(batch, freq, time, channels).permute(0, 3, 1, 2)
        mask = self.mask_head(restored)
        denoised_mag = noisy_mag * mask
        return ModelOutput(denoised_mag=denoised_mag, mask=mask)

