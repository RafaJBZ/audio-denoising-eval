from __future__ import annotations

import math

import torch
from torch import nn

from src.config.model import TransUnetConfig
from src.models.base import BaseDenoiseModel, ModelOutput


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _AttentionEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, attn_weights = self.attn(
            query=x,
            key=x,
            value=x,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attention_weights = attn_weights
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class AudioTransUnetDenoiser(BaseDenoiseModel):
    def __init__(self, config: TransUnetConfig) -> None:
        super().__init__()
        config.validate()
        self.max_attention_tokens = config.max_attention_tokens
        self.encoder1 = _EncoderBlock(1, config.base_channels)
        self.encoder2 = _EncoderBlock(config.base_channels, config.base_channels * 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bridge_proj = nn.Linear(config.base_channels * 2, config.transformer_dim)
        self.transformer_layers = nn.ModuleList(
            [
                _AttentionEncoderLayer(
                    embed_dim=config.transformer_dim,
                    n_heads=config.transformer_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.transformer_layers)
            ]
        )
        self.bridge_back = nn.Linear(config.transformer_dim, config.base_channels * 2)
        self.up = nn.ConvTranspose2d(
            config.base_channels * 2,
            config.base_channels,
            kernel_size=2,
            stride=2,
        )
        self.decoder = _EncoderBlock(config.base_channels * 2, config.base_channels)
        self.mask_head = nn.Sequential(
            nn.Conv2d(config.base_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def _collect_attention_map(self) -> torch.Tensor | None:
        maps: list[torch.Tensor] = []
        for layer in self.transformer_layers:
            attention_weights = layer.last_attention_weights
            if isinstance(attention_weights, torch.Tensor):
                # [batch, heads, tokens, tokens] -> head-average
                maps.append(attention_weights.mean(dim=1))
        if not maps:
            return None
        stacked = torch.stack(maps, dim=0)  # [layers, batch, tokens, tokens]
        return stacked.mean(dim=0)

    def _downsample_for_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, freq, time = x.shape
        total_tokens = freq * time
        if total_tokens <= self.max_attention_tokens:
            return x, (freq, time)

        aspect = freq / max(time, 1)
        target_freq = max(1, int(math.sqrt(self.max_attention_tokens * aspect)))
        target_freq = min(target_freq, freq)
        target_time = max(1, self.max_attention_tokens // target_freq)
        target_time = min(target_time, time)
        pooled = nn.functional.adaptive_avg_pool2d(x, output_size=(target_freq, target_time))
        return pooled, (freq, time)

    def forward(self, noisy_mag: torch.Tensor) -> ModelOutput:
        self.validate_input(noisy_mag)
        skip = self.encoder1(noisy_mag)
        x = self.pool(skip)
        x = self.encoder2(x)
        x_for_attention, original_shape = self._downsample_for_attention(x)
        batch, channels, freq, time = x_for_attention.shape
        tokens = x_for_attention.permute(0, 2, 3, 1).reshape(batch, freq * time, channels)
        tokens = self.bridge_proj(tokens)
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        tokens = self.bridge_back(tokens)
        x = tokens.reshape(batch, freq, time, channels).permute(0, 3, 1, 2)
        if x.shape[-2:] != original_shape:
            x = nn.functional.interpolate(
                x,
                size=original_shape,
                mode="bilinear",
                align_corners=False,
            )
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([x, skip], dim=1)
        x = self.decoder(x)
        mask = self.mask_head(x)
        attention_map = self._collect_attention_map()
        denoised_mag = noisy_mag * mask
        return ModelOutput(denoised_mag=denoised_mag, mask=mask, attention_map=attention_map)

