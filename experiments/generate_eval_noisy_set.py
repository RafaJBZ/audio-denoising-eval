from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root_on_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed noisy evaluation/listening set.",
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/eval_set"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Defaults to <MANIFEST_DIR>/dataset_manifest.csv from .env/runtime config.",
    )
    return parser.parse_args()


def main() -> None:
    ensure_project_root_on_path()
    from src.config.runtime import load_audio_config, load_runtime_paths
    from src.data.evalset import generate_fixed_eval_set

    args = parse_args()
    runtime = load_runtime_paths()
    audio_cfg = load_audio_config()
    manifest_path = args.manifest_path or (runtime.data_paths.manifest_dir / "dataset_manifest.csv")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run: uv run experiments/prepare_data.py"
        )
    metadata = generate_fixed_eval_set(
        manifest_csv_path=manifest_path,
        output_dir=args.output_dir,
        audio_cfg=audio_cfg,
        num_samples=args.num_samples,
        split=args.split,
        seed=args.seed,
    )
    print(f"Generated {len(metadata)} noisy samples in {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'eval_set_manifest.jsonl'}")
    print("To listen quickly: ffplay -autoexit <path_to_noisy.wav>")


if __name__ == "__main__":
    main()

