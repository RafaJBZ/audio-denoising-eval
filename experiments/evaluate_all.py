from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
import torchaudio
from _bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate selected denoising model(s) on a noisy/clean pair.",
    )
    parser.add_argument(
        "--noisy-path",
        type=Path,
        default=Path("data/sample_noisy.wav"),
        help="Path to noisy waveform.",
    )
    parser.add_argument(
        "--clean-path",
        type=Path,
        default=Path("data/sample_clean.wav"),
        help="Path to clean reference waveform.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="cnn",
        help="Comma-separated model keys. Valid: cnn,hybrid,efficient. Default: cnn.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("experiments/metrics_report.json"),
        help="Path to output JSON report.",
    )
    parser.add_argument(
        "--evalset-manifest",
        type=Path,
        default=Path("experiments/eval_set/eval_set_manifest.jsonl"),
        help="If present, evaluates over this full eval set manifest.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap for eval-set runs. 0 means all samples.",
    )
    return parser.parse_args()


def _load_reference(path: Path, sample_rate: int) -> torch.Tensor:
    waveform, source_sr = torchaudio.load(path.as_posix())
    if source_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sr, sample_rate)
    return waveform.mean(dim=0)


def _load_noisy_input(
    noisy_path: Path,
    sample_rate: int,
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    from src.data.preprocessing import compute_stft_magnitude, peak_normalize, resample_if_needed

    waveform, source_sr = torchaudio.load(noisy_path.as_posix())
    waveform = resample_if_needed(waveform, source_sr, sample_rate)
    waveform = peak_normalize(waveform)
    noisy_mag, noisy_complex = compute_stft_magnitude(waveform, cfg)
    return noisy_mag.unsqueeze(0), noisy_complex


def evaluate_single_pair(
    noisy_path: Path,
    clean_path: Path,
    model_keys: list[str],
) -> dict[str, dict[str, float | int | str]]:
    ensure_project_root_on_path()
    from src.config.runtime import load_audio_config
    from src.data.preprocessing import invert_stft
    from src.eval.metrics import parameter_count, safe_quality_metrics
    from src.inference.checkpoints import default_checkpoint_for_model, parse_models
    from src.models.factory import build_model

    cfg = load_audio_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_reference = _load_reference(clean_path, cfg.sample_rate_hz)
    noisy_mag, noisy_complex = _load_noisy_input(
        noisy_path=noisy_path,
        sample_rate=cfg.sample_rate_hz,
        cfg=cfg,
    )
    report: dict[str, dict[str, float | int | str]] = {}

    selected_models = parse_models(",".join(model_keys))
    for model_key in selected_models:
        model = build_model(model_key).to(device=device).eval()
        checkpoint_path = default_checkpoint_for_model(model_key)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for '{model_key}' at {checkpoint_path}. "
                "Train this model first or remove it from --models."
            )
        state = torch.load(checkpoint_path.as_posix(), map_location=device)
        if any(key.startswith("_orig_mod.") for key in state):
            state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            output = model(noisy_mag.to(device=device))
        mask = output.mask.squeeze(0).squeeze(0)
        denoised_complex = noisy_complex.to(mask.device) * mask
        estimate = invert_stft(denoised_complex, cfg).cpu()
        quality = safe_quality_metrics(
            reference=clean_reference,
            estimate=estimate,
            sample_rate=cfg.sample_rate_hz,
        )
        report[model_key] = {
            "pesq": quality.pesq,
            "stoi": quality.stoi,
            "si_sdr": quality.si_sdr,
            "params": parameter_count(model),
            "checkpoint_path": checkpoint_path.as_posix(),
        }
    return report


def evaluate_evalset(
    evalset_manifest: Path,
    model_keys: list[str],
    max_samples: int,
) -> dict[str, dict[str, float | int | str]]:
    ensure_project_root_on_path()
    from src.inference.checkpoints import parse_models

    if not evalset_manifest.exists():
        raise FileNotFoundError(
            f"Eval-set manifest not found at {evalset_manifest}. "
            "Run: uv run experiments/generate_eval_noisy_set.py --num-samples 100 --split test"
        )
    selected_models = parse_models(",".join(model_keys))
    sample_rows: list[dict[str, Any]] = []
    with evalset_manifest.open("r", encoding="utf-8") as in_file:
        for line in in_file:
            if line.strip() == "":
                continue
            sample_rows.append(json.loads(line))
    if max_samples > 0:
        sample_rows = sample_rows[:max_samples]
    if not sample_rows:
        raise ValueError("Eval-set manifest is empty.")

    aggregated: dict[str, dict[str, float | int | str]] = {}
    per_model_metrics: dict[str, dict[str, list[float]]] = {
        key: {"pesq": [], "stoi": [], "si_sdr": []} for key in selected_models
    }
    per_model_static: dict[str, dict[str, int | str]] = {}
    for row in sample_rows:
        pair_report = evaluate_single_pair(
            noisy_path=Path(row["noisy_path"]),
            clean_path=Path(row["clean_path"]),
            model_keys=selected_models,
        )
        for model_key in selected_models:
            per_model_metrics[model_key]["pesq"].append(float(pair_report[model_key]["pesq"]))
            per_model_metrics[model_key]["stoi"].append(float(pair_report[model_key]["stoi"]))
            per_model_metrics[model_key]["si_sdr"].append(float(pair_report[model_key]["si_sdr"]))
            if model_key not in per_model_static:
                per_model_static[model_key] = {
                    "params": int(pair_report[model_key]["params"]),
                    "checkpoint_path": str(pair_report[model_key]["checkpoint_path"]),
                }

    for model_key in selected_models:
        pesq_vals = per_model_metrics[model_key]["pesq"]
        stoi_vals = per_model_metrics[model_key]["stoi"]
        si_sdr_vals = per_model_metrics[model_key]["si_sdr"]
        static_values = per_model_static.get(model_key)
        if static_values is None:
            raise RuntimeError(f"Missing static model metadata for '{model_key}'.")
        aggregated[model_key] = {
            "num_samples": len(sample_rows),
            "pesq_mean": statistics.fmean(pesq_vals),
            "pesq_std": statistics.pstdev(pesq_vals) if len(pesq_vals) > 1 else 0.0,
            "stoi_mean": statistics.fmean(stoi_vals),
            "stoi_std": statistics.pstdev(stoi_vals) if len(stoi_vals) > 1 else 0.0,
            "si_sdr_mean": statistics.fmean(si_sdr_vals),
            "si_sdr_std": statistics.pstdev(si_sdr_vals) if len(si_sdr_vals) > 1 else 0.0,
            "params": int(static_values["params"]),
            "checkpoint_path": str(static_values["checkpoint_path"]),
        }
    return aggregated


def main() -> None:
    from src.inference.checkpoints import parse_models

    args = parse_args()
    noisy_path = args.noisy_path
    clean_path = args.clean_path
    model_keys = parse_models(args.models)
    if args.evalset_manifest.exists():
        report = evaluate_evalset(
            evalset_manifest=args.evalset_manifest,
            model_keys=model_keys,
            max_samples=args.max_samples,
        )
    else:
        if not noisy_path.exists() or not clean_path.exists():
            raise FileNotFoundError(
                f"Expected noisy/clean files at {noisy_path} and {clean_path}."
            )
        report = evaluate_single_pair(
            noisy_path=noisy_path,
            clean_path=clean_path,
            model_keys=model_keys,
        )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {args.output_path}")


if __name__ == "__main__":
    main()

