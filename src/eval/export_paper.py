from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt

MODEL_ORDER: tuple[str, ...] = ("cnn", "hybrid", "efficient")
MODEL_LABELS: Mapping[str, str] = {
    "cnn": "CNN U-Net",
    "hybrid": "Audio TransUNet",
    "efficient": "Efficient DF-Inspired",
}
RUN_NAME_BY_MODEL: Mapping[str, str] = {
    "cnn": "cnn_unet_baseline",
    "hybrid": "transunet_audio_hybrid",
    "efficient": "deepfilternet_efficient_native",
}


def _load_json_object(path: Path, description: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) == 0:
        raise ValueError(f"{description} must be a non-empty JSON object.")
    return payload


def _require_metric(
    values: Mapping[str, object],
    model_name: str,
    preferred_key: str,
    fallback_key: str,
) -> float:
    metric_value = values.get(preferred_key, values.get(fallback_key))
    if metric_value is None:
        raise ValueError(
            f"Missing metric for model '{model_name}': expected '{preferred_key}' or '{fallback_key}'."
        )
    try:
        return float(metric_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid metric value for model '{model_name}' key '{preferred_key}': {metric_value!r}"
        ) from exc


def _require_raw_metric(values: Mapping[str, object], model_name: str, key: str) -> float:
    metric_value = values.get(key)
    if metric_value is None:
        raise ValueError(f"Missing metric for model '{model_name}': expected '{key}'.")
    try:
        return float(metric_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid metric value for model '{model_name}' key '{key}': {metric_value!r}"
        ) from exc


def _format_mean_std(mean_value: float, std_value: float, decimals: int = 3) -> str:
    return f"{mean_value:.{decimals}f} $\\pm$ {std_value:.{decimals}f}"


def _sorted_models(report: Mapping[str, object]) -> list[str]:
    prioritized = [model for model in MODEL_ORDER if model in report]
    remaining = [model for model in report if model not in MODEL_ORDER]
    return [*prioritized, *sorted(remaining)]


def export_quality_table(metrics_json_path: Path, output_table_path: Path) -> None:
    report = _load_json_object(metrics_json_path, "Metrics report")
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & PESQ & STOI & SI-SDR & Params \\\\",
        "\\midrule",
    ]
    for model_name in _sorted_models(report):
        values = report[model_name]
        if not isinstance(values, dict):
            raise ValueError(f"Invalid model payload for {model_name}")
        pesq = _require_metric(values, model_name, "pesq_mean", "pesq")
        stoi = _require_metric(values, model_name, "stoi_mean", "stoi")
        si_sdr = _require_metric(values, model_name, "si_sdr_mean", "si_sdr")
        pesq_std = _require_raw_metric(values, model_name, "pesq_std")
        stoi_std = _require_raw_metric(values, model_name, "stoi_std")
        si_sdr_std = _require_raw_metric(values, model_name, "si_sdr_std")
        params_value = values.get("params")
        params_text = "-" if params_value is None else str(int(params_value))
        model_label = MODEL_LABELS.get(model_name, model_name)
        line = (
            f"{model_label} & "
            f"{_format_mean_std(pesq, pesq_std)} & "
            f"{_format_mean_std(stoi, stoi_std)} & "
            f"{_format_mean_std(si_sdr, si_sdr_std)} & "
            f"{params_text} \\\\"
        )
        lines.append(line)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    output_table_path.write_text("\n".join(lines), encoding="utf-8")


def export_metrics_table(metrics_json_path: Path, output_table_path: Path) -> None:
    export_quality_table(metrics_json_path=metrics_json_path, output_table_path=output_table_path)


def export_latency_table(latency_json_path: Path, output_table_path: Path) -> None:
    report = _load_json_object(latency_json_path, "Latency report")
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & P50 (ms) & P95 (ms) & P99 (ms) & Mean (ms) \\\\",
        "\\midrule",
    ]
    for model_name in _sorted_models(report):
        values = report[model_name]
        if not isinstance(values, dict):
            raise ValueError(f"Invalid model payload for {model_name}")
        p50 = _require_raw_metric(values, model_name, "p50_ms")
        p95 = _require_raw_metric(values, model_name, "p95_ms")
        p99 = _require_raw_metric(values, model_name, "p99_ms")
        mean_ms = _require_raw_metric(values, model_name, "mean_ms")
        model_label = MODEL_LABELS.get(model_name, model_name)
        lines.append(f"{model_label} & {p50:.3f} & {p95:.3f} & {p99:.3f} & {mean_ms:.3f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    output_table_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_simple_meta(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "" or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip().strip("\"'")
    return parsed


def _list_finished_runs(mlruns_root: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for meta_path in mlruns_root.glob("*/*/meta.yaml"):
        meta = _parse_simple_meta(meta_path)
        run_name = meta.get("run_name")
        status = meta.get("status")
        if run_name is None or status != "3":
            continue
        run_dir = meta_path.parent
        end_time_raw = meta.get("end_time", "0")
        try:
            end_time = int(end_time_raw)
        except ValueError:
            end_time = 0
        runs.append(
            {
                "run_name": run_name,
                "run_id": run_dir.name,
                "run_dir": run_dir,
                "end_time": end_time,
            }
        )
    return runs


def _metric_values(metric_path: Path) -> list[float]:
    if not metric_path.exists():
        return []
    values: list[float] = []
    for line in metric_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            values.append(float(parts[1]))
        except ValueError:
            continue
    return values


def export_training_cost_table(mlruns_root: Path, output_table_path: Path) -> None:
    if not mlruns_root.exists():
        raise FileNotFoundError(f"MLflow runs root does not exist: {mlruns_root}")
    finished_runs = _list_finished_runs(mlruns_root)
    latest_by_run_name: dict[str, dict[str, object]] = {}
    for run in finished_runs:
        run_name = str(run["run_name"])
        current = latest_by_run_name.get(run_name)
        if current is None or int(run["end_time"]) > int(current["end_time"]):
            latest_by_run_name[run_name] = run

    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Model & Run ID & Epochs & Best Val Loss & Total (s) & Sec/Epoch \\\\",
        "\\midrule",
    ]
    for model_name in MODEL_ORDER:
        model_label = MODEL_LABELS.get(model_name, model_name)
        run_name = RUN_NAME_BY_MODEL[model_name]
        run = latest_by_run_name.get(run_name)
        if run is None:
            lines.append(f"{model_label} & - & - & - & - & - \\\\")
            continue
        run_dir = Path(str(run["run_dir"]))
        run_id = str(run["run_id"])
        metric_dir = run_dir / "metrics"
        epoch_seconds = _metric_values(metric_dir / "epoch_seconds")
        best_val_loss_values = _metric_values(metric_dir / "best_val_loss")
        best_epoch_values = _metric_values(metric_dir / "best_epoch")
        epochs = len(epoch_seconds)
        total_seconds = sum(epoch_seconds)
        sec_per_epoch = (total_seconds / float(epochs)) if epochs > 0 else 0.0
        best_val_loss = best_val_loss_values[-1] if best_val_loss_values else 0.0
        best_epoch = int(best_epoch_values[-1]) if best_epoch_values else epochs
        if best_epoch > epochs and epochs > 0:
            best_epoch = epochs
        lines.append(
            f"{model_label} & {run_id[:8]} & {best_epoch} / {epochs} & "
            f"{best_val_loss:.4f} & {total_seconds:.1f} & {sec_per_epoch:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    output_table_path.parent.mkdir(parents=True, exist_ok=True)
    output_table_path.write_text("\n".join(lines), encoding="utf-8")


def export_quality_latency_tradeoff_figure(
    metrics_json_path: Path,
    latency_json_path: Path,
    output_figure_path: Path,
) -> None:
    metrics_report = _load_json_object(metrics_json_path, "Metrics report")
    latency_report = _load_json_object(latency_json_path, "Latency report")

    models = [model for model in MODEL_ORDER if model in metrics_report and model in latency_report]
    if not models:
        raise ValueError("No overlapping models found between metrics and latency reports.")

    x_values: list[float] = []
    y_values: list[float] = []
    labels: list[str] = []
    sizes: list[float] = []
    for model_name in models:
        metric_values = metrics_report[model_name]
        latency_values = latency_report[model_name]
        if not isinstance(metric_values, dict) or not isinstance(latency_values, dict):
            raise ValueError(f"Invalid payload for model '{model_name}'.")
        mean_ms = _require_raw_metric(latency_values, model_name, "mean_ms")
        pesq_mean = _require_metric(metric_values, model_name, "pesq_mean", "pesq")
        params = int(metric_values.get("params", 0))
        marker_size = max(40.0, min(800.0, (params / 10000.0)))
        x_values.append(mean_ms)
        y_values.append(pesq_mean)
        labels.append(MODEL_LABELS.get(model_name, model_name))
        sizes.append(marker_size)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    scatter = ax.scatter(x_values, y_values, s=sizes, alpha=0.8)
    _ = scatter
    for idx, label in enumerate(labels):
        ax.annotate(label, (x_values[idx], y_values[idx]), textcoords="offset points", xytext=(8, 4))
    ax.set_xlabel("Inference Mean Latency (ms)")
    ax.set_ylabel("PESQ Mean")
    ax.set_title("Quality-Latency Trade-off Across Architectures")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure_path.as_posix(), dpi=180)
    plt.close(fig)

