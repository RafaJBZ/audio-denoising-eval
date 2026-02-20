from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Mapping

import matplotlib.pyplot as plt
import streamlit as st
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = PROJECT_ROOT.as_posix()
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)


@dataclass(frozen=True, slots=True)
class ModelUiMeta:
    report_key: str
    display_name: str


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    loaded: bool
    checkpoint_path: Path


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    service: object
    device: torch.device
    checkpoints: dict[str, CheckpointInfo]


MODEL_META: Mapping[str, ModelUiMeta] = {
    "cnn_unet": ModelUiMeta(report_key="cnn", display_name="CNN U-Net"),
    "transunet_audio": ModelUiMeta(report_key="hybrid", display_name="Audio TransUNet"),
    "deepfilternet_eff": ModelUiMeta(report_key="efficient", display_name="Efficient DF-Inspired"),
}


def _load_json_report(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}.")
    return payload


@st.cache_resource
def _load_runtime_bundle() -> RuntimeBundle:
    from src.config.runtime import load_audio_config
    from src.inference.api import UnifiedInferenceService
    from src.inference.checkpoints import load_checkpoint_if_available
    from src.models.factory import build_model

    audio_cfg = load_audio_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model, cnn_checkpoint, cnn_loaded = load_checkpoint_if_available(
        model=build_model("cnn"),
        model_key="cnn",
        device=device,
    )
    hybrid_model, hybrid_checkpoint, hybrid_loaded = load_checkpoint_if_available(
        model=build_model("hybrid"),
        model_key="hybrid",
        device=device,
    )
    efficient_model, efficient_checkpoint, efficient_loaded = load_checkpoint_if_available(
        model=build_model("efficient"),
        model_key="efficient",
        device=device,
    )
    service = UnifiedInferenceService(
        audio_cfg=audio_cfg,
        cnn_model=cnn_model,
        hybrid_model=hybrid_model,
        efficient_model=efficient_model,
        device=device,
    )
    return RuntimeBundle(
        service=service,
        device=device,
        checkpoints={
            "cnn_unet": CheckpointInfo(loaded=cnn_loaded, checkpoint_path=cnn_checkpoint),
            "transunet_audio": CheckpointInfo(loaded=hybrid_loaded, checkpoint_path=hybrid_checkpoint),
            "deepfilternet_eff": CheckpointInfo(loaded=efficient_loaded, checkpoint_path=efficient_checkpoint),
        },
    )


def _plot_magnitude(magnitude_tensor: torch.Tensor, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(
        magnitude_tensor.squeeze(0).detach().cpu().numpy(),
        aspect="auto",
        origin="lower",
    )
    ax.set_title(title)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Frequency Bins")
    fig.colorbar(image, ax=ax)
    return fig


def main() -> None:
    from src.config.runtime import load_audio_config

    audio_cfg = load_audio_config()
    st.set_page_config(page_title="Comparative Audio Denoising", layout="wide")
    st.title("Comparative Audio Denoising")
    runtime_bundle = _load_runtime_bundle()
    service = runtime_bundle.service
    metrics_report = _load_json_report(Path("experiments/metrics_report.json"))
    latency_report = _load_json_report(Path("experiments/latency_report.json"))

    if runtime_bundle.device.type != "cuda":
        st.warning(
            "CUDA is not available in this runtime. Latency in this UI can be slower than your "
            "GPU benchmark numbers."
        )

    upload = st.file_uploader("Upload noisy WAV file", type=["wav"])
    if upload is None:
        st.info("Upload a WAV file to run all three models.")
        return
    raw_bytes = upload.read()
    if len(raw_bytes) == 0:
        raise ValueError("Uploaded file is empty.")
    with tempfile.TemporaryDirectory(prefix="denoise_upload_") as temp_dir:
        input_path = Path(temp_dir) / "input.wav"
        input_path.write_bytes(raw_bytes)
        start = perf_counter()
        results = service.run_all(input_path)
        elapsed_ms = (perf_counter() - start) * 1000.0
        st.subheader("Original")
        st.audio(BytesIO(raw_bytes), format="audio/wav")
        st.caption(f"Total inference time (all models): {elapsed_ms:.1f} ms")
        cols = st.columns(3)
        for idx, result in enumerate(results):
            with cols[idx]:
                model_meta = MODEL_META.get(
                    result.model_name,
                    ModelUiMeta(report_key=result.model_name, display_name=result.model_name),
                )
                checkpoint_info = runtime_bundle.checkpoints.get(result.model_name)
                st.subheader(model_meta.display_name)
                if checkpoint_info is None:
                    st.info("Checkpoint status unavailable.")
                elif checkpoint_info.loaded:
                    st.success(f"Checkpoint loaded: `{checkpoint_info.checkpoint_path.as_posix()}`")
                else:
                    st.warning(
                        f"Checkpoint missing: `{checkpoint_info.checkpoint_path.as_posix()}` "
                        "(running random weights)"
                    )

                metric_payload = metrics_report.get(model_meta.report_key)
                latency_payload = latency_report.get(model_meta.report_key)
                if isinstance(metric_payload, dict) and isinstance(latency_payload, dict):
                    params = metric_payload.get("params", "-")
                    pesq = metric_payload.get("pesq_mean", "-")
                    stoi = metric_payload.get("stoi_mean", "-")
                    si_sdr = metric_payload.get("si_sdr_mean", "-")
                    mean_ms = latency_payload.get("mean_ms", "-")
                    st.markdown(
                        f"- Params: `{params}`  \n"
                        f"- PESQ: `{pesq}`  \n"
                        f"- STOI: `{stoi}`  \n"
                        f"- SI-SDR: `{si_sdr}`  \n"
                        f"- Mean latency: `{mean_ms} ms`"
                    )

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_wav_path = Path(temp_wav.name)
                torchaudio.save(
                    uri=temp_wav_path.as_posix(),
                    src=result.denoised_waveform,
                    sample_rate=audio_cfg.sample_rate_hz,
                    format="wav",
                )
                wav_bytes = temp_wav_path.read_bytes()
                st.audio(wav_bytes, format="audio/wav")
                st.download_button(
                    label=f"Download {model_meta.display_name} output",
                    data=wav_bytes,
                    file_name=f"{model_meta.report_key}_denoised.wav",
                    mime="audio/wav",
                )
                temp_wav_path.unlink(missing_ok=True)
                st.pyplot(
                    _plot_magnitude(
                        result.denoised_magnitude,
                        f"{model_meta.display_name} magnitude",
                    )
                )
                if result.attention_map is not None:
                    with st.expander("Attention map (hybrid only)", expanded=False):
                        fig, ax = plt.subplots(figsize=(4, 4))
                        attention = result.attention_map[0].detach().cpu().numpy()
                        image = ax.imshow(attention, aspect="auto", origin="lower")
                        ax.set_title("Attention Map")
                        fig.colorbar(image, ax=ax)
                        st.pyplot(fig)


if __name__ == "__main__":
    main()

