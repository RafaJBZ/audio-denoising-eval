from __future__ import annotations

from pathlib import Path

import torch

from src.models.base import BaseDenoiseModel

MODEL_ORDER: tuple[str, ...] = ("cnn", "hybrid", "efficient")


def default_checkpoint_for_model(model_key: str) -> Path:
    if model_key not in MODEL_ORDER:
        raise ValueError("model_key must be one of: cnn, hybrid, efficient.")
    return Path(f"experiments/runs/{model_key}/best_model.pt")


def _normalize_state_dict_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("_orig_mod.") for key in state):
        return {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    return state


def load_checkpoint_if_available(
    model: BaseDenoiseModel,
    model_key: str,
    device: torch.device,
) -> tuple[BaseDenoiseModel, Path, bool]:
    checkpoint_path = default_checkpoint_for_model(model_key)
    if not checkpoint_path.exists():
        return model, checkpoint_path, False
    state = torch.load(checkpoint_path.as_posix(), map_location=device)
    state = _normalize_state_dict_keys(state)
    model.load_state_dict(state, strict=True)
    return model, checkpoint_path, True


def parse_models(raw_models: str) -> list[str]:
    selected = [value.strip() for value in raw_models.split(",") if value.strip() != ""]
    if not selected:
        raise ValueError("No models selected. Use a comma-separated list like 'cnn,hybrid'.")
    invalid = [value for value in selected if value not in MODEL_ORDER]
    if invalid:
        raise ValueError(f"Invalid model key(s): {invalid}. Valid values: {MODEL_ORDER}.")
    return selected

