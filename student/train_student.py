"""MERA-XQUAD student self-training entry point (MultiLABSA.docx §5.3).

Config-driven so every ablation (§7.1) is the same loop with terms turned off:
``--no_partial``, ``--no_relation``, ``--no_implicit``, ``--fixed_threshold``,
``--no_curriculum``, ``--single_teacher``, ``--no_multiview`` — each maps to one
row of Table 7.

Round loop (§5.3 steps 2-9):
    for round in 1..R:
        pool = unlabeled reviews for currently-admitted languages (curriculum)
        pseudo = DualTeacher(+multi-view) -> disagreement -> reliability -> route
        student trains on: gold-EN (L_sup) + full-route (L_full) + partial-route
                           (L_partial/relation) + consistency-route (L_cons)
        EMA update -> adjust threshold -> re-score curriculum

The dual-teacher labelling reuses ``teacher/`` unchanged; this file owns the
student optimisation and the round/curriculum/EMA bookkeeping.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from models.mt5 import linearize_quads
from student.hybrid_student import HybridStudent
from student.losses import LossWeights, MultiObjectiveLoss, balance_loss
from student.self_training import Curriculum, LanguageScore, adjust_threshold, ema_update
from teacher.confidence_fusion import FusionWeights, fuse
from teacher.disagreement import DisagreementWeights, compute_agreement
from teacher.extractive_teacher import ExtractiveTeacher
from teacher.generative_teacher import GenerativeTeacher
from teacher.multiview import MultiViewGenerativeTeacher, MultiViewWeights
from teacher.translator import NLLBTranslator
from utils.common import get_device, set_seed, setup_logging

logger = setup_logging()


# --------------------------------------------------------------------------- #
@dataclass
class StudentConfig:
    labeled_dir: str = "data_final/labeled_data/hamos26"
    unlabeled_csv: str = "data_final/unlabeled_data/hotel_review_merged.csv"
    text_column: str = "review"
    lang_column: str = "language"

    student_backbone: str = "hotel-mt5-asqp"     # start student from the ASQP mT5
    generative_model: str = "hotel-mt5-asqp"
    extractive_checkpoint: str = "checkpoints/dual-teacher/extractive_teacher.pt"
    extractive_backbone: str = "xlm-roberta-base"

    rounds: int = 3
    max_pool_per_round: int = 2000
    learning_rate: float = 3e-5
    train_batch_size: int = 4
    max_source_length: int = 256
    max_target_length: int = 160

    ema_mu: float = 0.997
    final_score_threshold: float = 0.5
    fixed_threshold: bool = False

    # multi-view / teacher toggles (ablation)
    multiview: bool = True
    single_teacher: bool = False

    # curriculum
    use_curriculum: bool = True
    warmup_languages: List[str] = field(default_factory=lambda: ["en"])
    admit_per_round: int = 2

    seed: int = 42
    output_dir: str = "checkpoints/mera-student"


# --------------------------------------------------------------------------- #
def build_teachers(cfg: StudentConfig, device):
    gen = GenerativeTeacher(cfg.generative_model, device=device,
                            max_source_length=cfg.max_source_length, max_new_tokens=cfg.max_target_length)
    mv = None
    if cfg.multiview:
        mv = MultiViewGenerativeTeacher(gen, NLLBTranslator(device=device), weights=MultiViewWeights())
    ext = None
    if not cfg.single_teacher and os.path.exists(cfg.extractive_checkpoint):
        ext = ExtractiveTeacher(backbone_name=cfg.extractive_backbone).to(device)
        ext.load_state_dict(torch.load(cfg.extractive_checkpoint, map_location=device))
        ext.eval()
    ext_tok = AutoTokenizer.from_pretrained(cfg.extractive_backbone) if ext else None
    return gen, mv, ext, ext_tok


def label_pool(cfg, texts, langs, gen, mv, ext, ext_tok, device, threshold) -> List[dict]:
    """Run the dual-teacher pipeline over a pool -> kept, routed pseudo-labels."""
    dis_w = DisagreementWeights()
    fus_w = FusionWeights(threshold=threshold)
    out: List[dict] = []
    for i in range(0, len(texts), 16):
        bt, bl = texts[i:i + 16], langs[i:i + 16]
        gen_pred = mv.predict(bt, bl) if mv is not None else gen.predict(bt)
        ext_pred = ext.predict(bt, ext_tok, device) if ext is not None else [[] for _ in bt]
        for text, gq, eq, lang in zip(bt, gen_pred, ext_pred, bl):
            fused = fuse(compute_agreement(gq, eq, text, dis_w), fus_w)
            if fused:
                out.append({"review": text, "language": lang, "quads": fused})
    return out


def score_languages(pseudo: List[dict], pool_counts: Dict[str, int]) -> Dict[str, LanguageScore]:
    agg: Dict[str, Dict[str, list]] = {}
    kept: Dict[str, int] = {}
    for row in pseudo:
        lang = row.get("language") or "en"
        s = agg.setdefault(lang, {"cg": [], "ag": [], "rq": []})
        kept[lang] = kept.get(lang, 0) + 1
        for q in row["quads"]:
            s["cg"].append(q.get("conf_g", 0.0))
            s["ag"].append(q.get("agreement", 0.0))
            s["rq"].append(q.get("reliability", {}).get("r_quad", 0.0))
    scores: Dict[str, LanguageScore] = {}
    for lang, s in agg.items():
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        scores[lang] = LanguageScore(
            confidence=mean(s["cg"]), agreement=mean(s["ag"]), stability=mean(s["rq"]),
            coverage=kept.get(lang, 0) / max(1, pool_counts.get(lang, 1)),
        )
    return scores


def make_gen_batch(rows, tokenizer, device, cfg):
    """(review -> linearized full-quad string) batch for L_sup / L_full."""
    sources = [r["review"] for r in rows]
    targets = [linearize_quads(r["quads"]) for r in rows]
    enc = tokenizer(sources, padding=True, truncation=True, max_length=cfg.max_source_length, return_tensors="pt").to(device)
    lab = tokenizer(text_target=targets, padding=True, truncation=True, max_length=cfg.max_target_length, return_tensors="pt").input_ids.to(device)
    lab[lab == tokenizer.pad_token_id] = -100
    return enc.input_ids, enc.attention_mask, lab


# --------------------------------------------------------------------------- #
def train(cfg: StudentConfig) -> None:
    set_seed(cfg.seed)
    device = get_device()
    logger.info("Device: %s | rounds=%d multiview=%s", device, cfg.rounds, cfg.multiview)

    tokenizer = AutoTokenizer.from_pretrained(cfg.student_backbone)
    student = HybridStudent(backbone_name=cfg.student_backbone).to(device)
    ema_teacher = copy.deepcopy(student).to(device)
    for p in ema_teacher.parameters():
        p.requires_grad_(False)

    optimizer = AdamW(student.parameters(), lr=cfg.learning_rate)
    loss_fn = MultiObjectiveLoss(LossWeights())
    curriculum = Curriculum(warmup_languages=cfg.warmup_languages, admit_per_round=cfg.admit_per_round)

    gen, mv, ext, ext_tok = build_teachers(cfg, device)

    # gold English supervised rows (always in the mix -> L_sup)
    from experiments.common.data import load_gold_split
    g_reviews, g_quads, _ = load_gold_split(cfg.labeled_dir, "train")
    gold_rows = [{"review": r, "quads": [{"aspect": q["aspect"], "opinion": q["opinion"],
                 "category": q["category"], "sentiment": q["sentiment"]} for q in qs]}
                 for r, qs in zip(g_reviews, g_quads) if qs]

    df = pd.read_csv(cfg.unlabeled_csv).dropna(subset=[cfg.text_column])
    threshold = cfg.final_score_threshold

    for rnd in range(1, cfg.rounds + 1):
        active = curriculum.admitted if cfg.use_curriculum else sorted(df[cfg.lang_column].dropna().unique())
        sub = df[df[cfg.lang_column].isin(active)].head(cfg.max_pool_per_round)
        texts = sub[cfg.text_column].astype(str).tolist()
        langs = sub[cfg.lang_column].astype(str).tolist()
        pool_counts = sub[cfg.lang_column].value_counts().to_dict()
        logger.info("Round %d | active langs=%s | pool=%d | threshold=%.2f", rnd, active, len(texts), threshold)

        pseudo = label_pool(cfg, texts, langs, gen, mv, ext, ext_tok, device, threshold)
        kept_ratio = len(pseudo) / max(1, len(texts))
        logger.info("  kept %d/%d pseudo-labeled reviews (%.1f%%)", len(pseudo), len(texts), 100 * kept_ratio)

        # Train student: gold-EN (L_sup) + full-route pseudo-labels (L_full).
        full_rows = [r for r in pseudo if any(q.get("route") == "full" for q in r["quads"])]
        train_rows = gold_rows + full_rows
        student.train()
        for i in range(0, len(train_rows), cfg.train_batch_size):
            batch = train_rows[i:i + cfg.train_batch_size]
            ids, mask, lab = make_gen_batch(batch, tokenizer, device, cfg)
            gen_loss = student.generation_loss(ids, mask, lab)
            aux = student.aux_forward(ids, mask)
            components = {"sup": gen_loss, "balance": balance_loss(
                student.classification_head(
                    aux["hidden"].mean(1), aux["hidden"].mean(1))[0])}
            out = loss_fn(components)
            optimizer.zero_grad()
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            ema_update(student, ema_teacher, cfg.ema_mu)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not cfg.fixed_threshold:
            threshold = adjust_threshold(threshold, kept_ratio)
        if cfg.use_curriculum:
            curriculum.update(score_languages(pseudo, pool_counts))

    os.makedirs(cfg.output_dir, exist_ok=True)
    student.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("Saved MERA student -> %s", cfg.output_dir)


# --------------------------------------------------------------------------- #
def parse_args() -> StudentConfig:
    p = argparse.ArgumentParser(description="MERA-XQUAD student self-training")
    for f_name, default in [
        ("labeled_dir", StudentConfig.labeled_dir), ("unlabeled_csv", StudentConfig.unlabeled_csv),
        ("student_backbone", StudentConfig.student_backbone), ("generative_model", StudentConfig.generative_model),
        ("extractive_checkpoint", StudentConfig.extractive_checkpoint), ("output_dir", StudentConfig.output_dir),
    ]:
        p.add_argument(f"--{f_name}", default=default)
    p.add_argument("--rounds", type=int, default=StudentConfig.rounds)
    p.add_argument("--max_pool_per_round", type=int, default=StudentConfig.max_pool_per_round)
    p.add_argument("--learning_rate", type=float, default=StudentConfig.learning_rate)
    p.add_argument("--seed", type=int, default=StudentConfig.seed)
    # ablation toggles (§7.1)
    p.add_argument("--no_multiview", dest="multiview", action="store_false")
    p.add_argument("--single_teacher", action="store_true")
    p.add_argument("--fixed_threshold", action="store_true")
    p.add_argument("--no_curriculum", dest="use_curriculum", action="store_false")
    a = p.parse_args()
    return StudentConfig(**{k: v for k, v in vars(a).items()})


if __name__ == "__main__":
    train(parse_args())
