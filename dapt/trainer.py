"""Training engine for Multilingual Hotel Domain-Adaptive Pretraining.

Implements a plain PyTorch loop around
:class:`transformers.MT5ForConditionalGeneration` with the full production
toolkit requested for Stage 0:

    * AdamW optimizer + linear warmup/decay scheduler
    * mixed precision (bf16, or fp16 with a GradScaler; fp32 fallback)
    * gradient accumulation and gradient clipping
    * a validation loop reporting the denoising cross-entropy
    * checkpoint saving with a rolling limit, and full resume
    * live logging of train loss / val loss / learning rate / epoch / step

The loss is the cross-entropy computed *inside* the model (we pass ``labels``),
so this file never defines a loss function of its own.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    MT5ForConditionalGeneration,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from utils import Config, count_parameters, resolve_precision, setup_logging

logger = setup_logging()

_CKPT_PREFIX = "checkpoint-"
_STATE_FILE = "training_state.pt"


class DAPTTrainer:
    """Drive denoising pretraining of mT5 on the hotel corpus.

    Args:
        cfg: run configuration.
        model: an ``MT5ForConditionalGeneration`` already on the target device.
        tokenizer: the matching tokenizer (saved alongside every checkpoint).
        train_loader / val_loader: DataLoaders yielding span-corruption batches.
        device: the torch device to train on.
    """

    def __init__(
        self,
        cfg: Config,
        model: MT5ForConditionalGeneration,
        tokenizer: PreTrainedTokenizerBase,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # ---- precision / autocast setup -----------------------------------
        self.precision = resolve_precision(cfg.precision)
        self.autocast_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.precision]
        self.use_autocast = self.precision in {"bf16", "fp16"} and device.type == "cuda"
        # A GradScaler is only needed for fp16 (bf16 has fp32 dynamic range).
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.precision == "fp16"))
        logger.info("Precision: %s (autocast=%s)", self.precision, self.use_autocast)

        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False  # incompatible with checkpointing

        # ---- optimizer with decoupled weight decay ------------------------
        self.optimizer = self._build_optimizer()

        # ---- total steps & scheduler --------------------------------------
        steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
        if cfg.max_steps is not None:
            self.total_steps = cfg.max_steps
        else:
            self.total_steps = steps_per_epoch * cfg.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=self.total_steps,
        )

        # ---- run state ----------------------------------------------------
        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        logger.info(
            "Trainable parameters: %.1fM | total optimizer steps: %d",
            count_parameters(model) / 1e6, self.total_steps,
        )

    # ------------------------------------------------------------------ #
    # Setup helpers                                                        #
    # ------------------------------------------------------------------ #
    def _build_optimizer(self) -> AdamW:
        """AdamW with weight decay disabled on biases and LayerNorm weights."""
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias") or "layer_norm" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
        groups = [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return AdamW(
            groups,
            lr=self.cfg.learning_rate,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            eps=self.cfg.adam_epsilon,
        )

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move a collated batch to the training device."""
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    def train(self) -> None:
        """Run the full training schedule (epochs x steps) until completion."""
        logger.info("Starting DAPT: %d epoch(s) from epoch %d, step %d",
                    self.cfg.num_epochs, self.start_epoch, self.global_step)
        self.model.train()

        for epoch in range(self.start_epoch, self.cfg.num_epochs):
            # Vary the temperature-sampled stream between epochs.
            if hasattr(self.train_loader.dataset, "set_epoch"):
                self.train_loader.dataset.set_epoch(epoch)

            running_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            progress = tqdm(self.train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)

            for micro_step, batch in enumerate(progress):
                batch = self._move_batch(batch)

                # ---- forward (loss computed inside the model) -------------
                with self._autocast():
                    outputs = self.model(**batch)
                    # Normalise so accumulated grads match a full-size batch.
                    loss = outputs.loss / self.cfg.gradient_accumulation_steps

                # ---- backward --------------------------------------------
                self.scaler.scale(loss).backward()
                running_loss += outputs.loss.item()

                # ---- optimizer step on accumulation boundary -------------
                is_boundary = (micro_step + 1) % self.cfg.gradient_accumulation_steps == 0
                is_last = (micro_step + 1) == len(self.train_loader)
                if is_boundary or is_last:
                    # Unscale before clipping so the norm is measured in fp32.
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    self._maybe_log(progress, running_loss, epoch)
                    running_loss = 0.0

                    self._maybe_eval_and_save(epoch)

                    if self.cfg.max_steps is not None and self.global_step >= self.cfg.max_steps:
                        logger.info("Reached max_steps=%d, stopping.", self.cfg.max_steps)
                        self._final_actions(epoch)
                        return

        self._final_actions(self.cfg.num_epochs - 1)

    def _autocast(self):
        """Return the appropriate autocast context (or a no-op for fp32/CPU)."""
        if self.use_autocast:
            return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
        return torch.autocast(device_type="cpu", enabled=False)

    def _maybe_log(self, progress: "tqdm", running_loss: float, epoch: int) -> None:
        """Emit train loss / lr / epoch / step at the configured cadence."""
        if self.global_step % self.cfg.logging_steps != 0:
            return
        avg_loss = running_loss  # already summed over one accumulation window
        lr = self.scheduler.get_last_lr()[0]
        progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", step=self.global_step)
        logger.info(
            "epoch %d | step %d/%d | train_loss %.4f | lr %.3e",
            epoch, self.global_step, self.total_steps, avg_loss, lr,
        )

    def _maybe_eval_and_save(self, epoch: int) -> None:
        """Run validation and/or checkpoint at their configured cadences."""
        if self.cfg.eval_steps and self.global_step % self.cfg.eval_steps == 0:
            val_loss = self.evaluate()
            logger.info("epoch %d | step %d | val_loss %.4f (best %.4f)",
                        epoch, self.global_step, val_loss, self.best_val_loss)
            self.best_val_loss = min(self.best_val_loss, val_loss)
            self.model.train()  # evaluate() switches to eval mode

        if self.cfg.save_steps and self.global_step % self.cfg.save_steps == 0:
            self.save_checkpoint(epoch)

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute the mean denoising cross-entropy over the validation set."""
        self.model.eval()
        total_loss, total_batches = 0.0, 0
        for batch in tqdm(self.val_loader, desc="validation", leave=False, dynamic_ncols=True):
            batch = self._move_batch(batch)
            with self._autocast():
                outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1
        return total_loss / max(total_batches, 1)

    # ------------------------------------------------------------------ #
    # Checkpointing & resume                                               #
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, epoch: int) -> str:
        """Save model, tokenizer and training state to ``checkpoint-<step>``."""
        ckpt_dir = os.path.join(self.cfg.output_dir, f"{_CKPT_PREFIX}{self.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # HF weights (safetensors) + config + generation_config + tokenizer.
        self.model.save_pretrained(ckpt_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(ckpt_dir)

        # Everything else needed to resume bit-for-bit.
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": epoch,
                "best_val_loss": self.best_val_loss,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_rng": np.random.get_state(),
            },
            os.path.join(ckpt_dir, _STATE_FILE),
        )
        logger.info("Saved checkpoint: %s", ckpt_dir)
        self._prune_checkpoints()
        return ckpt_dir

    def _prune_checkpoints(self) -> None:
        """Keep only the ``save_total_limit`` most recent checkpoints."""
        limit = self.cfg.save_total_limit
        if not limit or limit <= 0:
            return
        ckpts = _list_checkpoints(self.cfg.output_dir)
        for stale in ckpts[:-limit]:
            shutil.rmtree(stale, ignore_errors=True)
            logger.info("Pruned old checkpoint: %s", stale)

    def load_training_state(self, checkpoint_dir: str) -> None:
        """Restore optimizer/scheduler/scaler/step/RNG from a checkpoint dir.

        The model weights themselves are loaded by the entry-point via
        ``from_pretrained(checkpoint_dir)`` *before* the trainer is built.
        """
        state_path = os.path.join(checkpoint_dir, _STATE_FILE)
        state = torch.load(state_path, map_location=self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["global_step"]
        # Resume at the epoch after the one that was in progress.
        self.start_epoch = state["epoch"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        logger.info("Resumed from %s at step %d (epoch %d)",
                    checkpoint_dir, self.global_step, self.start_epoch)

    # ------------------------------------------------------------------ #
    # Finalisation                                                         #
    # ------------------------------------------------------------------ #
    def _final_actions(self, epoch: int) -> None:
        """Run a final validation and export the finished backbone."""
        val_loss = self.evaluate()
        logger.info("Final validation loss: %.4f", val_loss)
        self.save_pretrained_final()

    def save_pretrained_final(self) -> None:
        """Export the finished backbone to ``cfg.final_dir`` (``hotel-mt5/``).

        Produces ``config.json``, ``generation_config.json``, ``tokenizer.json``,
        the SentencePiece model (``spiece.model``) and ``model.safetensors`` —
        loadable via ``MT5ForConditionalGeneration.from_pretrained(final_dir)``.
        """
        os.makedirs(self.cfg.final_dir, exist_ok=True)
        self.model.save_pretrained(self.cfg.final_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(self.cfg.final_dir)
        logger.info("Saved final backbone to: %s", self.cfg.final_dir)


# --------------------------------------------------------------------------- #
# Module-level checkpoint helpers (used by the entry-point for resume)          #
# --------------------------------------------------------------------------- #
def _list_checkpoints(output_dir: str) -> List[str]:
    """Return checkpoint dirs sorted ascending by global step."""
    if not os.path.isdir(output_dir):
        return []
    ckpts = []
    for name in os.listdir(output_dir):
        match = re.fullmatch(rf"{_CKPT_PREFIX}(\d+)", name)
        if match and os.path.isdir(os.path.join(output_dir, name)):
            ckpts.append((int(match.group(1)), os.path.join(output_dir, name)))
    return [path for _, path in sorted(ckpts)]


def find_last_checkpoint(output_dir: str) -> Optional[str]:
    """Return the most recent checkpoint dir, or ``None`` if there are none."""
    ckpts = _list_checkpoints(output_dir)
    return ckpts[-1] if ckpts else None
