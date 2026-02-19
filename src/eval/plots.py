from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_spectrogram_panel(
    noisy_mag: np.ndarray,
    clean_mag: np.ndarray,
    denoised_mag: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title in zip(
        axes,
        (noisy_mag, clean_mag, denoised_mag),
        ("Noisy", "Clean", "Denoised"),
        strict=True,
    ):
        image = ax.imshow(data, aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Frequency Bins")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path.as_posix(), dpi=180)
    plt.close(fig)

