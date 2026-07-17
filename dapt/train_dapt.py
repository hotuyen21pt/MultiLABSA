"""Entry-point for Stage 0 — Multilingual Hotel Domain-Adaptive Pretraining.

Continues pretraining ``google/mt5-base`` with the denoising (span-corruption)
objective on the multilingual hotel-review corpus, then exports the adapted
backbone to ``hotel-mt5/``.

Example
-------
    python train_dapt.py \
        --data_dir ../data_final/unlabeled_data \
        --num_epochs 3 --batch_size 8 --gradient_accumulation_steps 4 \
        --precision auto --output_dir ../checkpoints/hotel-dapt \
        --final_dir ../hotel-mt5

    # resume from the latest checkpoint in --output_dir
    python train_dapt.py --resume ...

This script performs *only* domain-adaptive pretraining — there is no ASQP /
ACOS / teacher-student / pseudo-labelling logic here.
"""

from __future__ import annotations

import argparse
import dataclasses

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MT5ForConditionalGeneration

from collator import DataCollatorForSpanCorruption
from dataset import build_datasets
from masking import SpanCorruption
from trainer import DAPTTrainer, find_last_checkpoint
from utils import Config, get_device, set_seed, setup_logging

logger = setup_logging()


def parse_args() -> argparse.Namespace:
    """Define the CLI. Every field mirrors a :class:`Config` attribute."""
    p = argparse.ArgumentParser(description="Hotel-DAPT: continue-pretrain mT5 on hotel reviews")

    # data
    p.add_argument("--data_dir", default="data_final/unlabeled_data")
    p.add_argument("--text_column", default="review")
    p.add_argument("--language_column", default="language")
    p.add_argument("--file_glob", default="*_lang.csv")
    p.add_argument("--sampling_temperature", type=float, default=2.0,
                   help="Language sampling temperature; >1 up-weights low-resource langs")
    p.add_argument("--min_chars", type=int, default=10)
    p.add_argument("--max_samples_per_language", type=int, default=None)
    p.add_argument("--val_fraction", type=float, default=0.01)
    p.add_argument("--max_val_samples", type=int, default=5_000)
    p.add_argument("--epoch_size", type=int, default=None)

    # model / tokenizer
    p.add_argument("--model_name", default="google/mt5-base")
    p.add_argument("--max_seq_length", type=int, default=256)

    # span corruption
    p.add_argument("--noise_density", type=float, default=0.15)
    p.add_argument("--max_span_length", type=int, default=5)
    p.add_argument("--lexicon_boost", type=float, default=5.0)

    # optimisation
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=1_000)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    # runtime / IO
    p.add_argument("--output_dir", default="checkpoints/hotel-dapt")
    p.add_argument("--final_dir", default="hotel-mt5")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=1_000)
    p.add_argument("--save_steps", type=int, default=1_000)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest checkpoint in --output_dir")
    p.add_argument("--resume_from", default=None,
                   help="Explicit checkpoint dir to resume from (overrides --resume search)")
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a :class:`Config` from parsed args (fields share names)."""
    field_names = {f.name for f in dataclasses.fields(Config)}
    kwargs = {k: v for k, v in vars(args).items() if k in field_names}
    return Config(**kwargs)


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    set_seed(cfg.seed)
    device = get_device()
    logger.info("Device: %s", device)

    # ---- resolve resume checkpoint (if any) -------------------------------
    resume_ckpt = args.resume_from
    if resume_ckpt is None and args.resume:
        resume_ckpt = find_last_checkpoint(cfg.output_dir)
        if resume_ckpt is None:
            logger.warning("No checkpoint found in %s; starting from scratch.", cfg.output_dir)

    # Load model+tokenizer from the checkpoint when resuming, else the base model.
    load_from = resume_ckpt or cfg.model_name
    logger.info("Loading model & tokenizer from: %s", load_from)
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    model = MT5ForConditionalGeneration.from_pretrained(load_from)
    model.to(device)

    # ---- data -------------------------------------------------------------
    train_ds, val_ds = build_datasets(cfg)
    span_corruption = SpanCorruption(
        tokenizer=tokenizer,
        noise_density=cfg.noise_density,
        max_span_length=cfg.max_span_length,
        lexicon_boost=cfg.lexicon_boost,
        extra_lexicon_terms=cfg.extra_lexicon_terms,
    )
    collator = DataCollatorForSpanCorruption(
        tokenizer=tokenizer,
        span_corruption=span_corruption,
        max_seq_length=cfg.max_seq_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---- train ------------------------------------------------------------
    trainer = DAPTTrainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    if resume_ckpt is not None:
        trainer.load_training_state(resume_ckpt)

    trainer.train()
    logger.info("DAPT complete. Backbone saved to '%s'.", cfg.final_dir)


if __name__ == "__main__":
    main()
