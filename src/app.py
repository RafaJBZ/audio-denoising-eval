from __future__ import annotations

import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = PROJECT_ROOT.as_posix()
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)


@st.cache_resource
def _load_service() -> Any:
    from src.config.runtime import load_audio_config
    from src.inference.api import UnifiedInferenceService
    from src.inference.checkpoints import load_checkpoint_if_available
    from src.models.factory import build_model

    audio_cfg = load_audio_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model, _, _ = load_checkpoint_if_available(
        model=build_model("cnn"),
        model_key="cnn",
        device=device,
    )
    hybrid_model, _, _ = load_checkpoint_if_available(
        model=build_model("hybrid"),
        model_key="hybrid",
        device=device,
    )
    efficient_model, _, _ = load_checkpoint_if_available(
        model=build_model("efficient"),
        model_key="efficient",
        device=device,
    )
    return UnifiedInferenceService(
        audio_cfg=audio_cfg,
        cnn_model=cnn_model,
        hybrid_model=hybrid_model,
        efficient_model=efficient_model,
        device=device,
    )


def _plot_magnitude(magnitude_tensor: Any, title: str):
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
        service = _load_service()
        results = service.run_all(input_path)
        st.subheader("Original")
        st.audio(BytesIO(raw_bytes), format="audio/wav")
        cols = st.columns(3)
        for idx, result in enumerate(results):
            with cols[idx]:
                st.subheader(result.model_name)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_wav_path = Path(temp_wav.name)
                torchaudio.save(
                    uri=temp_wav_path.as_posix(),
                    src=result.denoised_waveform,
                    sample_rate=audio_cfg.sample_rate_hz,
                    format="wav",
                )
                st.audio(temp_wav_path.read_bytes(), format="audio/wav")
                temp_wav_path.unlink(missing_ok=True)
                st.pyplot(
                    _plot_magnitude(
                        result.denoised_magnitude,
                        f"{result.model_name} magnitude",
                    )
                )
                if result.attention_map is not None:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    attention = result.attention_map[0].detach().cpu().numpy()
                    image = ax.imshow(attention, aspect="auto", origin="lower")
                    ax.set_title("Attention Map")
                    fig.colorbar(image, ax=ax)
                    st.pyplot(fig)


if __name__ == "__main__":
    main()

