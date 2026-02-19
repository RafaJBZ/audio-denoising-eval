from __future__ import annotations

import platform
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path


def _require_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "mlflow is required for experiment tracking. Install with: uv add mlflow"
        ) from exc
    return mlflow


def get_git_commit_hash() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return output
    except Exception:
        return "unknown"


class MlflowTracker:
    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        if tracking_uri.strip() == "":
            raise ValueError("tracking_uri cannot be empty.")
        if experiment_name.strip() == "":
            raise ValueError("experiment_name cannot be empty.")
        self._mlflow = _require_mlflow()
        self._mlflow.set_tracking_uri(tracking_uri)
        self._mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str) -> None:
        if run_name.strip() == "":
            raise ValueError("run_name cannot be empty.")
        self._mlflow.start_run(run_name=run_name)

    def end_run(self) -> None:
        self._mlflow.end_run()

    def log_dict(self, payload: Mapping[str, object], artifact_file: str) -> None:
        self._mlflow.log_dict(dict(payload), artifact_file)

    def log_config_dataclass(self, name: str, config_obj: object) -> None:
        if not is_dataclass(config_obj) or isinstance(config_obj, type):
            raise ValueError(f"{name} must be a dataclass instance, not a class.")
        self._mlflow.log_dict(asdict(config_obj), f"configs/{name}.json")

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if step is None:
            self._mlflow.log_metric(key, value)
        else:
            self._mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Artifact does not exist: {path}")
        self._mlflow.log_artifact(path.as_posix())

    def log_runtime_metadata(self, seed: int) -> None:
        self._mlflow.log_params(
            {
                "seed": seed,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "git_commit": get_git_commit_hash(),
            }
        )

