from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT_STR = PROJECT_ROOT.as_posix()
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all denoising models on one wav file.")
    parser.add_argument("--input", type=Path, required=True, help="Path to input noisy wav file.")
    parser.add_argument(
        "--models",
        type=str,
        default="cnn,hybrid,efficient",
        help="Comma-separated model keys: cnn,hybrid,efficient",
    )
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="If set, run selected models even when checkpoint is missing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/inference_outputs"),
        help="Directory for output wav files.",
    )
    return parser.parse_args()


def main() -> None:
    from src.config.runtime import load_audio_config
    from src.inference.api import UnifiedInferenceService
    from src.inference.checkpoints import load_checkpoint_if_available, parse_models
    from src.models.base import BaseDenoiseModel
    from src.models.factory import build_model

    args = parse_args()
    cfg = load_audio_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_keys = parse_models(args.models)

    loaded_models: dict[str, BaseDenoiseModel] = {}
    for model_key in model_keys:
        model, ckpt_path, loaded = load_checkpoint_if_available(
            model=build_model(model_key),
            model_key=model_key,
            device=device,
        )
        print(f"[infer] {model_key} checkpoint loaded={loaded} path={ckpt_path}")
        if loaded:
            loaded_models[model_key] = model
            continue
        if args.allow_untrained:
            loaded_models[model_key] = model
            continue

    if not loaded_models:
        raise RuntimeError(
            "No selected models have checkpoints. "
            "Train at least one model or pass --allow-untrained."
        )

    cnn_model = loaded_models.get("cnn", build_model("cnn"))
    hybrid_model = loaded_models.get("hybrid", build_model("hybrid"))
    efficient_model = loaded_models.get("efficient", build_model("efficient"))
    service = UnifiedInferenceService(
        audio_cfg=cfg,
        cnn_model=cnn_model,
        hybrid_model=hybrid_model,
        efficient_model=efficient_model,
        device=device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for result in service.run_all(args.input):
        model_key = {
            "cnn_unet": "cnn",
            "transunet_audio": "hybrid",
            "deepfilternet_eff": "efficient",
        }[result.model_name]
        if model_key not in loaded_models:
            continue
        output_path = args.output_dir / f"{result.model_name}.wav"
        torchaudio.save(
            output_path.as_posix(),
            result.denoised_waveform,
            sample_rate=cfg.sample_rate_hz,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

