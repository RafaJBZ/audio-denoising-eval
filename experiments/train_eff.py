from __future__ import annotations

from pathlib import Path

from _bootstrap import ensure_project_root_on_path


def main() -> None:
    ensure_project_root_on_path()
    from src.config.experiment import TrackingConfig, TrainingConfig
    from src.config.runtime import load_audio_config, load_runtime_paths
    from src.models.factory import build_model
    from src.training.loop import train_model

    runtime = load_runtime_paths()
    audio_cfg = load_audio_config()
    model = build_model("efficient")
    history = train_model(
        model=model,
        run_name="deepfilternet_efficient_native",
        output_dir=Path("experiments/runs/efficient"),
        data_paths=runtime.data_paths,
        audio_cfg=audio_cfg,
        train_cfg=TrainingConfig(),
        tracking_cfg=TrackingConfig(),
    )
    print(f"Completed efficient model training with {len(history)} epochs.")


if __name__ == "__main__":
    main()

