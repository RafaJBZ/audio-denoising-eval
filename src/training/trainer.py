from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.config.audio import AudioConfig
from src.config.experiment import TrainingConfig
from src.data.dynamic_mixer import DenoiseBatchItem
from src.models.base import BaseDenoiseModel
from src.training.loss import SpectralL1Loss
from src.utils.tracking import MlflowTracker


@dataclass(frozen=True, slots=True)
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float
    epoch_seconds: float


class DenoiseTrainer:
    def __init__(
        self,
        model: BaseDenoiseModel,
        audio_cfg: AudioConfig,
        train_cfg: TrainingConfig,
        tracker: MlflowTracker,
        output_dir: Path,
        device: torch.device,
    ) -> None:
        audio_cfg.validate()
        train_cfg.validate()
        self.model = model.to(device=device)
        self.audio_cfg = audio_cfg
        self.train_cfg = train_cfg
        self.tracker = tracker
        self.output_dir = output_dir
        self.device = device
        self.loss_fn = SpectralL1Loss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(enabled=False)
        self.best_val_loss: float | None = None
        self.bad_epochs = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if train_cfg.use_compile and hasattr(torch, "compile"):
            compiled_model = torch.compile(self.model)
            self.model = cast(BaseDenoiseModel, compiled_model)

    def _batch_to_device(self, batch: DenoiseBatchItem) -> tuple[torch.Tensor, torch.Tensor]:
        noisy = batch.noisy_mag.to(device=self.device, non_blocking=True)
        clean = batch.clean_mag.to(device=self.device, non_blocking=True)
        return noisy, clean

    def _run_epoch(self, loader: DataLoader[DenoiseBatchItem], training: bool) -> float:
        if training:
            self.model.train()
        else:
            self.model.eval()
        cumulative_loss = 0.0
        total_batches = 0
        for batch in loader:
            noisy, clean = self._batch_to_device(batch)
            with torch.set_grad_enabled(training):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.train_cfg.use_bfloat16 and self.device.type == "cuda",
                ):
                    output = self.model(noisy)
                    loss = self.loss_fn(output.denoised_mag, clean)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip_norm)
                self.optimizer.step()
            cumulative_loss += float(loss.detach().cpu().item())
            total_batches += 1
        if total_batches == 0:
            raise ValueError("DataLoader produced zero batches.")
        return cumulative_loss / float(total_batches)

    def train(
        self,
        train_loader: DataLoader[DenoiseBatchItem],
        val_loader: DataLoader[DenoiseBatchItem],
    ) -> list[EpochStats]:
        history: list[EpochStats] = []
        print(
            f"[train] max_epochs={self.train_cfg.max_epochs}, "
            f"early_stopping_patience={self.train_cfg.early_stopping_patience}",
            flush=True,
        )
        for epoch in range(1, self.train_cfg.max_epochs + 1):
            start = perf_counter()
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss = self._run_epoch(val_loader, training=False)
            elapsed = perf_counter() - start
            stats = EpochStats(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch_seconds=elapsed,
            )
            history.append(stats)
            self.tracker.log_metric("train_loss", train_loss, step=epoch)
            self.tracker.log_metric("val_loss", val_loss, step=epoch)
            self.tracker.log_metric("epoch_seconds", elapsed, step=epoch)
            improved = self._update_early_stopping(val_loss, epoch)
            best_val = self.best_val_loss if self.best_val_loss is not None else val_loss
            print(
                f"[train] epoch={epoch}/{self.train_cfg.max_epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"best_val={best_val:.6f} improved={improved} "
                f"bad_epochs={self.bad_epochs}/{self.train_cfg.early_stopping_patience} "
                f"time_sec={elapsed:.2f}",
                flush=True,
            )
            if self.bad_epochs >= self.train_cfg.early_stopping_patience:
                print(
                    f"[train] early stopping at epoch={epoch} "
                    f"(patience={self.train_cfg.early_stopping_patience})",
                    flush=True,
                )
                break
        print(f"[train] finished with {len(history)} epoch(s).", flush=True)
        return history

    def _update_early_stopping(self, val_loss: float, epoch: int) -> bool:
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.bad_epochs = 0
            checkpoint_path = self.output_dir / "best_model.pt"
            state_dict = self.model.state_dict()
            cleaned_state_dict = {
                key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
            }
            torch.save(cleaned_state_dict, checkpoint_path)
            self.tracker.log_artifact(checkpoint_path)
            self.tracker.log_metric("best_epoch", float(epoch))
            self.tracker.log_metric("best_val_loss", float(val_loss))
            return True
        self.bad_epochs += 1
        return False

