from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

from src.config.audio import AudioConfig
from src.data.preprocessing import (
    compute_stft_magnitude,
    invert_stft,
    peak_normalize,
    resample_if_needed,
)
from src.models.base import BaseDenoiseModel, ModelOutput


@dataclass(frozen=True, slots=True)
class InferenceResult:
    model_name: str
    denoised_waveform: torch.Tensor
    denoised_magnitude: torch.Tensor
    attention_map: torch.Tensor | None


class UnifiedInferenceService:
    def __init__(
        self,
        audio_cfg: AudioConfig,
        cnn_model: BaseDenoiseModel,
        hybrid_model: BaseDenoiseModel,
        efficient_model: BaseDenoiseModel,
        device: torch.device | None = None,
    ) -> None:
        self.audio_cfg = audio_cfg
        self.audio_cfg.validate()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: dict[str, BaseDenoiseModel] = {
            "cnn_unet": cnn_model.to(self.device).eval(),
            "transunet_audio": hybrid_model.to(self.device).eval(),
            "deepfilternet_eff": efficient_model.to(self.device).eval(),
        }

    def _prepare_input(self, wav_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        if not wav_path.exists():
            raise FileNotFoundError(f"Input wav file does not exist: {wav_path}")
        waveform, sample_rate = torchaudio.load(wav_path.as_posix())
        waveform = resample_if_needed(waveform, sample_rate, self.audio_cfg.sample_rate_hz)
        waveform = peak_normalize(waveform)
        magnitude, noisy_complex = compute_stft_magnitude(waveform, self.audio_cfg)
        return magnitude.unsqueeze(0).to(self.device), noisy_complex

    def _reconstruct(self, output: ModelOutput, noisy_complex: torch.Tensor) -> torch.Tensor:
        mask = output.mask.squeeze(0).squeeze(0)
        denoised_complex = noisy_complex.to(mask.device) * mask
        waveform = invert_stft(denoised_complex, self.audio_cfg)
        return waveform.unsqueeze(0).cpu()

    def run_all(self, wav_path: Path) -> list[InferenceResult]:
        input_mag, noisy_complex = self._prepare_input(wav_path)
        results: list[InferenceResult] = []
        for model_name, model in self.models.items():
            with torch.no_grad():
                output = model(input_mag)
            denoised_wave = self._reconstruct(output, noisy_complex)
            results.append(
                # Keep optional attention output for hybrid interpretability.
                InferenceResult(
                    model_name=model_name,
                    denoised_waveform=denoised_wave,
                    denoised_magnitude=output.denoised_mag.squeeze(0).cpu(),
                    attention_map=(
                        output.attention_map.cpu()
                        if output.attention_map is not None
                        else None
                    ),
                )
            )
        return results

