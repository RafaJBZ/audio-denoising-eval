from __future__ import annotations

from src.config.model import CnnUnetConfig, EfficientDfConfig, TransUnetConfig
from src.models.base import BaseDenoiseModel
from src.models.cnn_unet import CnnUnetDenoiser
from src.models.deepfilternet_eff import EfficientDeepFilterNetDenoiser
from src.models.transunet_audio import AudioTransUnetDenoiser


def build_model(model_name: str) -> BaseDenoiseModel:
    if model_name == "cnn":
        return CnnUnetDenoiser(CnnUnetConfig())
    if model_name == "hybrid":
        return AudioTransUnetDenoiser(TransUnetConfig())
    if model_name == "efficient":
        return EfficientDeepFilterNetDenoiser(EfficientDfConfig())
    raise ValueError("model_name must be one of: cnn, hybrid, efficient.")

