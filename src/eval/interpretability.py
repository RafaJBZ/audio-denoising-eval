from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.models.transunet_audio import AudioTransUnetDenoiser


def extract_attention_map(model: AudioTransUnetDenoiser, noisy_mag: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        output = model(noisy_mag)
    if output.attention_map is None:
        raise RuntimeError(
            "No attention map found. Ensure TransUNet forward path has attention layers enabled."
        )
    return output.attention_map


def save_attention_heatmap(attention_map: torch.Tensor, output_path: Path) -> None:
    if attention_map.ndim != 3:
        raise ValueError(
            "Expected attention map shape [batch, tokens, tokens], "
            f"got {tuple(attention_map.shape)}."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_2d = attention_map[0].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(map_2d, aspect="auto", origin="lower")
    ax.set_title("TransUNet Attention Map (Token Space)")
    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path.as_posix(), dpi=180)
    plt.close(fig)

