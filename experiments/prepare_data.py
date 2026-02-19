from __future__ import annotations

from collections import Counter
from typing import Any

from _bootstrap import ensure_project_root_on_path


def _print_split_summary(
    speech_records: list[Any],
    noise_records: list[Any],
) -> None:
    counter: Counter[tuple[str, str]] = Counter()
    for record in speech_records + noise_records:
        counter[(record.domain_label, record.split)] += 1

    print("Split summary:")
    for domain in ("speech", "noise"):
        train_count = counter[(domain, "train")]
        val_count = counter[(domain, "val")]
        test_count = counter[(domain, "test")]
        total_count = train_count + val_count + test_count
        print(
            f"  {domain:<6} total={total_count:<6} "
            f"train={train_count:<6} val={val_count:<6} test={test_count:<6}"
        )


def main() -> None:
    ensure_project_root_on_path()
    from src.config.runtime import load_runtime_paths
    from src.data.acquisition import (
        build_noise_manifest,
        build_speech_manifest,
        write_manifests,
    )

    runtime = load_runtime_paths()
    speech_records = build_speech_manifest(runtime.data_paths.librispeech_root)
    noise_records = build_noise_manifest(runtime.data_paths.audioset_root)
    write_manifests(
        speech_records=speech_records,
        noise_records=noise_records,
        output_dir=runtime.data_paths.manifest_dir,
    )
    print(
        f"Wrote manifests with {len(speech_records)} speech clips and "
        f"{len(noise_records)} noise clips to {runtime.data_paths.manifest_dir}."
    )
    _print_split_summary(speech_records=speech_records, noise_records=noise_records)


if __name__ == "__main__":
    main()

