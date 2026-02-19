from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommonModelConfig:
    """Shared shape contract across all denoising models."""

    in_channels: int = 1
    out_channels: int = 1
    freq_bins: int = 257

    def validate(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        if self.freq_bins <= 0:
            raise ValueError("freq_bins must be positive.")


@dataclass(frozen=True, slots=True)
class CnnUnetConfig:
    base_channels: int = 32
    depth: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if self.depth < 2:
            raise ValueError("depth must be >= 2.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


@dataclass(frozen=True, slots=True)
class TransUnetConfig:
    base_channels: int = 32
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4
    dropout: float = 0.1
    max_attention_tokens: int = 1024

    def validate(self) -> None:
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if self.transformer_dim <= 0:
            raise ValueError("transformer_dim must be positive.")
        if self.transformer_heads <= 0:
            raise ValueError("transformer_heads must be positive.")
        if self.transformer_dim % self.transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads.")
        if self.transformer_layers <= 0:
            raise ValueError("transformer_layers must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.max_attention_tokens <= 0:
            raise ValueError("max_attention_tokens must be positive.")


@dataclass(frozen=True, slots=True)
class EfficientDfConfig:
    bottleneck_channels: int = 48
    conv_kernel_size: int = 3
    recurrent_hidden: int = 96
    deep_filter_taps: int = 5

    def validate(self) -> None:
        if self.bottleneck_channels <= 0:
            raise ValueError("bottleneck_channels must be positive.")
        if self.conv_kernel_size <= 0 or self.conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be a positive odd integer.")
        if self.recurrent_hidden <= 0:
            raise ValueError("recurrent_hidden must be positive.")
        if self.deep_filter_taps <= 0:
            raise ValueError("deep_filter_taps must be positive.")

