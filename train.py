"""Entry point for the Dual Teacher ABSA pipeline.

Two stages:

    1. Train the Extractive Teacher (T_E)'s three heads (Span / Relation /
       Classification) on the gold ASQP annotations in ``--labeled_dir``.
       (The Generative Teacher, ``hotel-mt5``, is assumed already fine-tuned
       — see ``dapt/`` for how the backbone itself was domain-adapted.)

    2. Run BOTH teachers over the unlabeled corpus (``--unlabeled_csv``),
       reconcile their outputs with the Architectural Disagreement module,
       fuse each merged quad's confidence, and write every quad whose
       FinalScore clears ``--final_score_threshold`` to ``--pseudo_labels_out``
       as pseudo labels.

Example
-------
    python train.py \\
        --labeled_dir data_final/labeled_data/hamos26 \\
        --unlabeled_csv data_final/unlabeled_data/hotel_review_merged.csv \\
        --generative_model hotel-mt5 --extractive_backbone xlm-roberta-base \\
        --num_epochs 5 --output_dir checkpoints/dual-teacher

    # Skip extractive-teacher training and reuse a previously trained one:
    python train.py --skip_training --extractive_checkpoint checkpoints/dual-teacher/extractive_teacher.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import List, Optional

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from teacher.confidence_fusion import FusionWeights, fuse
from teacher.disagreement import DisagreementWeights, compute_agreement
from teacher.extractive_teacher import ExtractiveTeacher
from teacher.generative_teacher import GenerativeTeacher
from teacher.multiview import MultiViewGenerativeTeacher, MultiViewWeights
from teacher.translator import NLLBTranslator
from utils.common import Config, count_parameters, get_device, set_seed, setup_logging
from utils.data import ExtractiveCollator, ExtractiveDataset, load_split

logger = setup_logging()


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual Teacher ABSA: train T_E, fuse with T_G into pseudo labels")

    # data
    p.add_argument("--labeled_dir", default="data_final/labeled_data/hamos26")
    p.add_argument("--unlabeled_csv", default="data_final/unlabeled_data/hotel_review_merged.csv")
    p.add_argument("--text_column", default="review")
    p.add_argument("--max_unlabeled_samples", type=int, default=None)

    # generative teacher
    p.add_argument("--generative_model", default="hotel-mt5")
    p.add_argument("--gen_max_source_length", type=int, default=256)
    p.add_argument("--gen_max_target_length", type=int, default=160)
    p.add_argument("--gen_num_beams", type=int, default=4)

    # multi-view pseudo-label generation (native / translate-to-EN / back-translation)
    p.add_argument("--multiview", action="store_true",
                   help="Run T_G on 3 views per review (native, translated-to-English, "
                        "back-translated) and reconcile via self-consistency voting instead "
                        "of a single native-language prediction. Needs --lang_column in "
                        "--unlabeled_csv (fastText 'language' code) to pick translation directions.")
    p.add_argument("--translator_model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--lang_column", default="language")
    p.add_argument("--multiview_min_agreeing_views", type=int, default=2)
    p.add_argument("--multiview_confidence_boost", type=float, default=0.15)
    p.add_argument("--multiview_pivot_lang", default="fra_Latn",
                   help="Back-translation pivot language for reviews already in English")

    # extractive teacher
    p.add_argument("--extractive_backbone", default="xlm-roberta-base")
    p.add_argument("--relation_proj_size", type=int, default=256)
    p.add_argument("--classifier_proj_size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_seq_length", type=int, default=160)
    p.add_argument("--relation_threshold", type=float, default=0.5)

    # extractive teacher training
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--relation_loss_weight", type=float, default=1.0)
    p.add_argument("--classification_loss_weight", type=float, default=1.0)
    p.add_argument("--negative_relation_ratio", type=float, default=2.0)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--skip_training", action="store_true",
                   help="Skip T_E training and load --extractive_checkpoint instead")
    p.add_argument("--extractive_checkpoint", default=None,
                   help="Path to a previously saved T_E state_dict (required with --skip_training)")

    # disagreement
    p.add_argument("--overlap_aspect_weight", type=float, default=0.4)
    p.add_argument("--overlap_opinion_weight", type=float, default=0.3)
    p.add_argument("--category_match_weight", type=float, default=0.15)
    p.add_argument("--sentiment_match_weight", type=float, default=0.15)
    p.add_argument("--match_threshold", type=float, default=0.3)

    # confidence fusion
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--final_score_threshold", type=float, default=0.5)

    # runtime / IO
    p.add_argument("--output_dir", default="checkpoints/dual-teacher")
    p.add_argument("--pseudo_labels_out", default="checkpoints/dual-teacher/pseudo_labels.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--inference_batch_size", type=int, default=16)
    p.add_argument("--skip_pseudo_labeling", action="store_true",
                   help="Only train T_E; skip the T_G + fusion pseudo-labeling stage")
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    field_names = {f.name for f in dataclasses.fields(Config)}
    kwargs = {k: v for k, v in vars(args).items() if k in field_names}
    return Config(**kwargs)


# --------------------------------------------------------------------------- #
# Stage 1: train the Extractive Teacher's heads                                #
# --------------------------------------------------------------------------- #
def train_extractive_teacher(cfg: Config, tokenizer, device: torch.device) -> ExtractiveTeacher:
    train_examples = load_split(cfg.labeled_dir, "train")
    val_examples = load_split(cfg.labeled_dir, "val")
    logger.info("Loaded %d train / %d val labeled examples from %s",
                len(train_examples), len(val_examples), cfg.labeled_dir)

    collator = ExtractiveCollator(tokenizer=tokenizer, max_seq_length=cfg.max_seq_length)
    train_loader = DataLoader(
        ExtractiveDataset(train_examples), batch_size=cfg.train_batch_size, shuffle=True,
        collate_fn=collator, num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        ExtractiveDataset(val_examples), batch_size=cfg.eval_batch_size, shuffle=False,
        collate_fn=collator, num_workers=cfg.num_workers,
    )

    model = ExtractiveTeacher(
        backbone_name=cfg.extractive_backbone,
        relation_proj_size=cfg.relation_proj_size,
        classifier_proj_size=cfg.classifier_proj_size,
        dropout=cfg.dropout,
        max_seq_length=cfg.max_seq_length,
        negative_relation_ratio=cfg.negative_relation_ratio,
    ).to(device)
    logger.info("ExtractiveTeacher parameters: %.1fM", count_parameters(model) / 1e6)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * cfg.warmup_ratio), num_training_steps=total_steps
    )

    global_step = 0
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_start = time.time()
        running = {"span": 0.0, "relation": 0.0, "category": 0.0, "sentiment": 0.0}
        for step, batch in enumerate(train_loader):
            losses = model.compute_training_loss(batch, device)
            total_loss = (
                losses["span"]
                + cfg.relation_loss_weight * losses["relation"]
                + cfg.classification_loss_weight * (losses["category"] + losses["sentiment"])
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

            for k in running:
                running[k] += float(losses[k])
            if global_step % cfg.logging_steps == 0:
                n = cfg.logging_steps
                logger.info(
                    "epoch %d step %d | span %.4f rel %.4f cat %.4f sent %.4f",
                    epoch, step + 1, running["span"] / n, running["relation"] / n,
                    running["category"] / n, running["sentiment"] / n,
                )
                running = {k: 0.0 for k in running}

        val_loss = evaluate_extractive_teacher(model, val_loader, device, cfg)
        logger.info("epoch %d done in %.1fs | val_total_loss=%.4f", epoch, time.time() - epoch_start, val_loss)

    os.makedirs(cfg.output_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.output_dir, "extractive_teacher.pt")
    torch.save(model.state_dict(), ckpt_path)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("Saved Extractive Teacher checkpoint to %s", ckpt_path)
    return model


@torch.no_grad()
def evaluate_extractive_teacher(model: ExtractiveTeacher, val_loader: DataLoader, device, cfg: Config) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in val_loader:
        losses = model.compute_training_loss(batch, device)
        total += float(
            losses["span"]
            + cfg.relation_loss_weight * losses["relation"]
            + cfg.classification_loss_weight * (losses["category"] + losses["sentiment"])
        )
        count += 1
    return total / max(1, count)


# --------------------------------------------------------------------------- #
# Stage 2: dual-teacher inference -> disagreement -> confidence fusion         #
# --------------------------------------------------------------------------- #
def run_pseudo_labeling(cfg: Config, extractive_teacher: ExtractiveTeacher, tokenizer, device) -> None:
    try:
        generative_teacher = GenerativeTeacher(
            model_name_or_path=cfg.generative_model,
            device=device,
            max_source_length=cfg.gen_max_source_length,
            max_new_tokens=cfg.gen_max_target_length,
            num_beams=cfg.gen_num_beams,
        )
    except OSError as err:
        logger.warning(
            "Could not load Generative Teacher from '%s' (%s). "
            "Skipping the pseudo-labeling stage — T_E training already completed.",
            cfg.generative_model, err,
        )
        return

    multiview_teacher: Optional[MultiViewGenerativeTeacher] = None
    if cfg.multiview:
        translator = NLLBTranslator(model_name=cfg.translator_model, device=device)
        multiview_teacher = MultiViewGenerativeTeacher(
            generative_teacher, translator,
            weights=MultiViewWeights(
                min_agreeing_views=cfg.multiview_min_agreeing_views,
                confidence_boost_per_view=cfg.multiview_confidence_boost,
                pivot_lang_for_english=cfg.multiview_pivot_lang,
            ),
        )
        logger.info("Multi-view pseudo-labeling enabled (translator: %s)", cfg.translator_model)

    df = pd.read_csv(cfg.unlabeled_csv)
    df = df.dropna(subset=[cfg.text_column])
    texts: List[str] = df[cfg.text_column].astype(str).tolist()
    if cfg.multiview and cfg.lang_column in df.columns:
        langs: List[Optional[str]] = df[cfg.lang_column].tolist()
    else:
        if cfg.multiview:
            logger.warning(
                "--multiview set but '%s' column not found in %s; every review will be "
                "treated as English (View 2/3 translation becomes a no-op).",
                cfg.lang_column, cfg.unlabeled_csv,
            )
        langs = [None] * len(texts)
    if cfg.max_unlabeled_samples:
        texts = texts[: cfg.max_unlabeled_samples]
        langs = langs[: cfg.max_unlabeled_samples]
    logger.info("Running dual-teacher inference over %d unlabeled reviews", len(texts))

    disagreement_weights = DisagreementWeights(
        aspect_overlap=cfg.overlap_aspect_weight,
        opinion_overlap=cfg.overlap_opinion_weight,
        category_match=cfg.category_match_weight,
        sentiment_match=cfg.sentiment_match_weight,
        match_threshold=cfg.match_threshold,
    )
    fusion_weights = FusionWeights(
        alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma, threshold=cfg.final_score_threshold
    )

    pseudo_labeled_reviews: List[dict] = []
    total_kept, total_seen = 0, 0
    batch_size = cfg.inference_batch_size
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_langs = langs[start : start + batch_size]

        if multiview_teacher is not None:
            gen_predictions = multiview_teacher.predict(batch_texts, batch_langs)
        else:
            gen_predictions = generative_teacher.predict(batch_texts)
        ext_predictions = extractive_teacher.predict(
            batch_texts, tokenizer, device, relation_threshold=cfg.relation_threshold
        )

        for text, gen_quads, ext_quads in zip(batch_texts, gen_predictions, ext_predictions):
            merged = compute_agreement(gen_quads, ext_quads, text, disagreement_weights)
            fused = fuse(merged, fusion_weights)
            total_seen += len(merged)
            total_kept += len(fused)
            if fused:
                pseudo_labeled_reviews.append({"review": text, "quads": fused})

        if (start // batch_size) % 10 == 0:
            logger.info("  processed %d/%d reviews", min(start + batch_size, len(texts)), len(texts))

    os.makedirs(os.path.dirname(cfg.pseudo_labels_out) or ".", exist_ok=True)
    with open(cfg.pseudo_labels_out, "w", encoding="utf-8") as f:
        json.dump(pseudo_labeled_reviews, f, ensure_ascii=False, indent=2)

    logger.info(
        "Pseudo-labeling complete: kept %d/%d candidate quads (%.1f%%) across %d reviews -> %s",
        total_kept, total_seen, 100.0 * total_kept / max(1, total_seen),
        len(pseudo_labeled_reviews), cfg.pseudo_labels_out,
    )


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    set_seed(cfg.seed)
    device = get_device()
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.extractive_backbone)

    if args.skip_training:
        if not args.extractive_checkpoint:
            raise ValueError("--skip_training requires --extractive_checkpoint")
        extractive_teacher = ExtractiveTeacher(
            backbone_name=cfg.extractive_backbone,
            relation_proj_size=cfg.relation_proj_size,
            classifier_proj_size=cfg.classifier_proj_size,
            dropout=cfg.dropout,
            max_seq_length=cfg.max_seq_length,
            negative_relation_ratio=cfg.negative_relation_ratio,
        ).to(device)
        extractive_teacher.load_state_dict(torch.load(args.extractive_checkpoint, map_location=device))
        logger.info("Loaded Extractive Teacher from %s", args.extractive_checkpoint)
    else:
        extractive_teacher = train_extractive_teacher(cfg, tokenizer, device)

    if not args.skip_pseudo_labeling:
        run_pseudo_labeling(cfg, extractive_teacher, tokenizer, device)


if __name__ == "__main__":
    main()
