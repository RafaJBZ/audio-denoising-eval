from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root_on_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated LibriSpeech + AudioSet downloader.")
    parser.add_argument(
        "--audioset-mode",
        choices=("metadata", "clips"),
        default="clips",
        help="metadata: download CSV metadata only. clips: download filtered YouTube noise clips.",
    )
    parser.add_argument(
        "--max-audioset-clips",
        type=int,
        default=200,
        help="Maximum number of AudioSet clips to download when audioset-mode=clips.",
    )
    return parser.parse_args()


def _warn_if_paths_look_swapped(librispeech_root: Path, audioset_root: Path) -> None:
    lib_str = librispeech_root.as_posix().lower()
    audio_str = audioset_root.as_posix().lower()
    if "audioset" in lib_str and "librispeech" in audio_str:
        print("WARNING: Your .env paths look swapped:")
        print(f"  LIBRISPEECH_ROOT={librispeech_root}")
        print(f"  AUDIOSET_ROOT={audioset_root}")


def main() -> None:
    ensure_project_root_on_path()
    from src.config.runtime import load_runtime_paths
    from src.data.download import (
        download_audioset_metadata,
        download_audioset_noise_subset,
        download_librispeech_train_clean_100,
    )

    args = parse_args()
    runtime = load_runtime_paths()
    librispeech_root = runtime.data_paths.librispeech_root
    audioset_root = runtime.data_paths.audioset_root
    _warn_if_paths_look_swapped(librispeech_root, audioset_root)

    librispeech_dir = download_librispeech_train_clean_100(librispeech_root)
    print(f"LibriSpeech ready: {librispeech_dir}")

    if args.audioset_mode == "metadata":
        balanced_csv, eval_csv = download_audioset_metadata(audioset_root)
        print(f"AudioSet metadata ready: {balanced_csv}")
        print(f"AudioSet metadata ready: {eval_csv}")
        return

    clips_dir = download_audioset_noise_subset(
        audioset_root=audioset_root,
        max_clips=args.max_audioset_clips,
    )
    print(f"AudioSet clips ready: {clips_dir}")


if __name__ == "__main__":
    main()

