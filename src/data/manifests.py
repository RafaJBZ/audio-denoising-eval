from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    file_path: str
    duration_sec: float
    sample_rate: int
    split: str
    domain_label: str
    speaker_or_noise_id: str

    def validate(self) -> None:
        if self.file_path.strip() == "":
            raise ValueError("file_path cannot be empty.")
        if self.duration_sec <= 0.0:
            raise ValueError("duration_sec must be positive.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test.")
        if self.domain_label not in {"speech", "noise"}:
            raise ValueError("domain_label must be one of: speech, noise.")
        if self.speaker_or_noise_id.strip() == "":
            raise ValueError("speaker_or_noise_id cannot be empty.")


def write_manifest_csv(records: list[ManifestRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "file_path",
                "duration_sec",
                "sample_rate",
                "split",
                "domain_label",
                "speaker_or_noise_id",
            ],
        )
        writer.writeheader()
        for record in records:
            record.validate()
            writer.writerow(
                {
                    "file_path": record.file_path,
                    "duration_sec": f"{record.duration_sec:.6f}",
                    "sample_rate": record.sample_rate,
                    "split": record.split,
                    "domain_label": record.domain_label,
                    "speaker_or_noise_id": record.speaker_or_noise_id,
                }
            )


def read_manifest_csv(manifest_path: Path) -> list[ManifestRecord]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")
    records: list[ManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            record = ManifestRecord(
                file_path=row["file_path"],
                duration_sec=float(row["duration_sec"]),
                sample_rate=int(row["sample_rate"]),
                split=row["split"],
                domain_label=row["domain_label"],
                speaker_or_noise_id=row["speaker_or_noise_id"],
            )
            record.validate()
            records.append(record)
    return records


def write_manifest_jsonl(records: list[ManifestRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            record.validate()
            payload = {
                "file_path": record.file_path,
                "duration_sec": record.duration_sec,
                "sample_rate": record.sample_rate,
                "split": record.split,
                "domain_label": record.domain_label,
                "speaker_or_noise_id": record.speaker_or_noise_id,
            }
            jsonl_file.write(json.dumps(payload))
            jsonl_file.write("\n")

