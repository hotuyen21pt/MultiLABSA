"""Stage 0.5 — ASQP fine-tuning of the DAPT-adapted mT5 backbone.

``hotel-mt5/`` (produced by ``dapt/train_dapt.py``) only knows span-corruption
denoising on hotel reviews — it has never seen the (aspect, opinion,
category, sentiment) task, nor the linearized target format
``models/mt5.py`` expects to parse. This script closes that gap with
supervised seq2seq fine-tuning on the gold ASQP annotations
(``data_final/labeled_data/.../train.json``): review text -> linearized quad
string. The result is what ``teacher/generative_teacher.py``'s "already
fine-tuned hotel-mt5" assumption actually requires.

Example
-------
    python train_asqp_mt5.py \\
        --labeled_dir data_final/labeled_data/hamos26 \\
        --base_model hotel-mt5 \\
        --num_epochs 8 --train_batch_size 8 \\
        --output_dir checkpoints/hotel-mt5-asqp --final_dir hotel-mt5-asqp

    # resume from the latest checkpoint in --output_dir
    python train_asqp_mt5.py --resume ...
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MT5ForConditionalGeneration, get_linear_schedule_with_warmup

from utils.asqp_data import ASQPCollator, ASQPDataset, load_asqp_split
from utils.common import get_device, set_seed, setup_logging

logger = setup_logging()

_CKPT_PREFIX = "checkpoint-"
_STATE_FILE = "training_state.pt"


@dataclass
class ASQPConfig:
    # ---- data -----------------------------------------------------------
    labeled_dir: str = "data_final/labeled_data/hamos26"

    # ---- model / tokenizer --------------------------------------------
    base_model: str = "hotel-mt5"           # the DAPT-adapted backbone to fine-tune
    max_source_length: int = 256
    max_target_length: int = 160

    # ---- optimisation -------------------------------------------------
    train_batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 3e-4             # higher than DAPT's 1e-4: short supervised run, small dataset
    weight_decay: float = 0.01
    num_epochs: int = 8
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1            # softens targets; a few thousand gold quads overfit easily otherwise

    # ---- runtime / IO ---------------------------------------------------
    output_dir: str = "checkpoints/hotel-mt5-asqp"
    final_dir: str = "hotel-mt5-asqp"
    logging_steps: int = 50
    save_steps: int = 500
    save_total_limit: int = 3
    preview_every_n_epochs: int = 2         # log a few generated examples periodically
    num_workers: int = 2
    seed: int = 42


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune hotel-mt5 to generate ASQP quads")

    p.add_argument("--labeled_dir", default="data_final/labeled_data/hamos26")
    p.add_argument("--base_model", default="hotel-mt5")
    p.add_argument("--max_source_length", type=int, default=256)
    p.add_argument("--max_target_length", type=int, default=160)

    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_epochs", type=int, default=8)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)

    p.add_argument("--output_dir", default="checkpoints/hotel-mt5-asqp")
    p.add_argument("--final_dir", default="hotel-mt5-asqp")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--preview_every_n_epochs", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in --output_dir")
    p.add_argument("--resume_from", default=None, help="Explicit checkpoint dir to resume from")
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> ASQPConfig:
    field_names = {f.name for f in dataclasses.fields(ASQPConfig)}
    kwargs = {k: v for k, v in vars(args).items() if k in field_names}
    return ASQPConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Checkpointing (mirrors dapt/trainer.py's convention: checkpoint-<step>/      #
# with model+tokenizer via save_pretrained + a training_state.pt sidecar)      #
# --------------------------------------------------------------------------- #
def find_last_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    steps = []
    for name in os.listdir(output_dir):
        if name.startswith(_CKPT_PREFIX):
            try:
                steps.append((int(name[len(_CKPT_PREFIX):]), name))
            except ValueError:
                continue
    if not steps:
        return None
    return os.path.join(output_dir, max(steps)[1])


def save_checkpoint(model, tokenizer, optimizer, scheduler, global_step: int, epoch: int,
                     output_dir: str, save_total_limit: int) -> None:
    ckpt_dir = os.path.join(output_dir, f"{_CKPT_PREFIX}{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir, safe_serialization=True)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
         "global_step": global_step, "epoch": epoch},
        os.path.join(ckpt_dir, _STATE_FILE),
    )
    _prune_checkpoints(output_dir, save_total_limit)


def _prune_checkpoints(output_dir: str, save_total_limit: int) -> None:
    steps = []
    for name in os.listdir(output_dir):
        if name.startswith(_CKPT_PREFIX):
            try:
                steps.append((int(name[len(_CKPT_PREFIX):]), name))
            except ValueError:
                continue
    steps.sort()
    while len(steps) > save_total_limit:
        _, name = steps.pop(0)
        shutil.rmtree(os.path.join(output_dir, name), ignore_errors=True)


# --------------------------------------------------------------------------- #
# Train / eval / preview                                                       #
# --------------------------------------------------------------------------- #
def compute_loss(model, batch: dict, label_smoothing: float) -> torch.Tensor:
    """Label-smoothed cross-entropy over the decoder logits.

    ``labels`` is still passed to ``model(...)`` (not omitted) so HF builds
    the correct shifted ``decoder_input_ids`` internally; only the returned
    ``loss`` (plain CE, no smoothing) is discarded in favour of recomputing
    it from ``logits`` with label smoothing applied.
    """
    outputs = model(**batch)
    logits = outputs.logits
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), batch["labels"].view(-1),
        ignore_index=-100, label_smoothing=label_smoothing,
    )


@torch.no_grad()
def evaluate(model, val_loader: DataLoader, device, label_smoothing: float) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total += float(compute_loss(model, batch, label_smoothing))
        count += 1
    model.train()
    return total / max(1, count)


@torch.no_grad()
def preview_generation(model, tokenizer, samples: List[Tuple[str, str]], device, cfg: ASQPConfig) -> None:
    """Log a handful of gold-vs-generated quad strings — a fast sanity check
    that training is actually converging on the target format, not just
    driving the loss down on padding/easy tokens."""
    model.eval()
    for source, gold_target in samples:
        inputs = tokenizer(source, truncation=True, max_length=cfg.max_source_length, return_tensors="pt").to(device)
        generated = model.generate(**inputs, max_new_tokens=cfg.max_target_length, num_beams=4)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        logger.info("  [preview] source: %s", source[:100])
        logger.info("  [preview] gold  : %s", gold_target)
        logger.info("  [preview] pred  : %s", decoded)
    model.train()


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    set_seed(cfg.seed)
    device = get_device()
    logger.info("Device: %s", device)

    resume_ckpt = args.resume_from
    if resume_ckpt is None and args.resume:
        resume_ckpt = find_last_checkpoint(cfg.output_dir)
        if resume_ckpt is None:
            logger.warning("No checkpoint found in %s; starting from scratch.", cfg.output_dir)

    load_from = resume_ckpt or cfg.base_model
    logger.info("Loading model & tokenizer from: %s", load_from)
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    model = MT5ForConditionalGeneration.from_pretrained(load_from)
    model.to(device)

    train_pairs = load_asqp_split(cfg.labeled_dir, "train")
    val_pairs = load_asqp_split(cfg.labeled_dir, "val")
    logger.info("Loaded %d train / %d val ASQP (review -> quad string) pairs", len(train_pairs), len(val_pairs))

    collator = ASQPCollator(tokenizer, cfg.max_source_length, cfg.max_target_length)
    train_loader = DataLoader(
        ASQPDataset(train_pairs), batch_size=cfg.train_batch_size, shuffle=True,
        collate_fn=collator, num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        ASQPDataset(val_pairs), batch_size=cfg.eval_batch_size, shuffle=False,
        collate_fn=collator, num_workers=cfg.num_workers,
    )

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * cfg.warmup_ratio), num_training_steps=total_steps
    )

    global_step, start_epoch = 0, 0
    if resume_ckpt is not None:
        state_path = os.path.join(resume_ckpt, _STATE_FILE)
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            global_step = state["global_step"]
            start_epoch = state["epoch"]
            logger.info("Resumed training state from %s (step %d, epoch %d)", state_path, global_step, start_epoch)

    model.train()
    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = compute_loss(model, batch, cfg.label_smoothing)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += float(loss)

            if global_step % cfg.logging_steps == 0:
                logger.info("epoch %d step %d | loss %.4f", epoch, step + 1, running_loss / cfg.logging_steps)
                running_loss = 0.0

            if cfg.save_steps and global_step % cfg.save_steps == 0:
                save_checkpoint(model, tokenizer, optimizer, scheduler, global_step, epoch,
                                 cfg.output_dir, cfg.save_total_limit)

        val_loss = evaluate(model, val_loader, device, cfg.label_smoothing)
        logger.info("epoch %d done in %.1fs | val_loss=%.4f", epoch, time.time() - epoch_start, val_loss)

        if cfg.preview_every_n_epochs and (epoch + 1) % cfg.preview_every_n_epochs == 0:
            preview_generation(model, tokenizer, val_pairs[:3], device, cfg)

    os.makedirs(cfg.final_dir, exist_ok=True)
    model.save_pretrained(cfg.final_dir, safe_serialization=True)
    tokenizer.save_pretrained(cfg.final_dir)
    logger.info("ASQP fine-tuning complete. Generator saved to '%s'.", cfg.final_dir)


if __name__ == "__main__":
    main()
