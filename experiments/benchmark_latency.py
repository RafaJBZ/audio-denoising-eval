from __future__ import annotations

import json
from pathlib import Path

import torch
from _bootstrap import ensure_project_root_on_path


def main() -> None:
    ensure_project_root_on_path()
    from src.eval.benchmark import model_latency
    from src.inference.checkpoints import load_checkpoint_if_available
    from src.models.factory import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, dict[str, float | bool | str]] = {}
    for model_name in ("cnn", "hybrid", "efficient"):
        model, checkpoint_path, loaded = load_checkpoint_if_available(
            model=build_model(model_name),
            model_key=model_name,
            device=device,
        )
        latency = model_latency(
            model=model,
            batch_size=1,
            freq_bins=257,
            time_frames=128,
            device=device,
        )
        report[model_name] = {
            "p50_ms": latency.p50_ms,
            "p95_ms": latency.p95_ms,
            "p99_ms": latency.p99_ms,
            "mean_ms": latency.mean_ms,
            "std_ms": latency.std_ms,
            "checkpoint_loaded": loaded,
            "checkpoint_path": checkpoint_path.as_posix(),
        }
    output_path = Path("experiments/latency_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved latency report to {output_path}")


if __name__ == "__main__":
    main()

