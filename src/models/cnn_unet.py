from __future__ import annotations

import torch
from torch import nn

from src.config.model import CnnUnetConfig
from src.models.base import BaseDenoiseModel, ModelOutput


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CnnUnetDenoiser(BaseDenoiseModel):
    def __init__(self, config: CnnUnetConfig) -> None:
        super().__init__()
        config.validate()
        channels = [config.base_channels * (2**i) for i in range(config.depth)]
        self.down_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        in_ch = 1
        for out_ch in channels:
            self.down_blocks.append(ConvBlock(in_ch, out_ch, config.dropout))
            in_ch = out_ch
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, config.dropout)
        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        up_channels = list(reversed(channels))
        current = channels[-1] * 2
        for ch in up_channels:
            self.up_convs.append(nn.ConvTranspose2d(current, ch, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(ch * 2, ch, config.dropout))
            current = ch
        self.mask_head = nn.Sequential(
            nn.Conv2d(channels[0], 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, noisy_mag: torch.Tensor) -> ModelOutput:
        self.validate_input(noisy_mag)
        skips: list[torch.Tensor] = []
        x = noisy_mag
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up_conv, up_block, skip in zip(
            self.up_convs,
            self.up_blocks,
            reversed(skips),
            strict=True,
        ):
            x = up_conv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
            x = up_block(x)
        mask = self.mask_head(x)
        denoised_mag = noisy_mag * mask
        return ModelOutput(denoised_mag=denoised_mag, mask=mask)

