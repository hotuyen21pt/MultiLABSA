"""Shared runtime utilities + the single :class:`Config` for the Dual Teacher
pipeline (mirrors the style of ``dapt/utils.py``: one dataclass holding every
hyper-parameter, populated from CLI args in ``train.py``).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # heavy deps imported lazily at call sites
    import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for a reproducible run."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> "torch.device":
    """Return the CUDA device if available, else CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_precision(requested: str) -> str:
    """Resolve a requested precision to a concrete one usable on this machine.

    Args:
        requested: one of ``"auto"``, ``"bf16"``, ``"fp16"`` or ``"fp32"``.

    Returns:
        A concrete precision string. ``"auto"`` picks ``bf16`` when the GPU
        supports it (Ampere+), otherwise ``fp16`` on GPU, otherwise ``fp32``.
    """
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    return "fp32"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Create/return the package logger with a single stream handler."""
    logger = logging.getLogger("dual_teacher")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def count_parameters(model: "torch.nn.Module") -> int:
    """Count trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class Config:
    """All hyper-parameters for the Dual Teacher pipeline."""

    # ---- data -----------------------------------------------------------
    labeled_dir: str = "data_final/labeled_data/hamos26"
    unlabeled_csv: str = "data_final/unlabeled_data/hotel_review_merged.csv"
    text_column: str = "review"
    max_unlabeled_samples: Optional[int] = None

    # ---- Generative Teacher (T_G) ----------------------------------------
    generative_model: str = "hotel-mt5"     # already-fine-tuned ASQP mT5 checkpoint
    gen_max_source_length: int = 256
    gen_max_target_length: int = 160
    gen_num_beams: int = 4

    # ---- Extractive Teacher (T_E) -----------------------------------------
    extractive_backbone: str = "xlm-roberta-base"
    relation_proj_size: int = 256
    classifier_proj_size: int = 256
    dropout: float = 0.1
    max_seq_length: int = 160
    relation_threshold: float = 0.5         # min P(aspect<->opinion) to keep a pair

    # ---- Extractive Teacher training ---------------------------------------
    # learning_rate search space: {1e-5, 2e-5, 3e-5, 5e-5}; 2e-5 is the starting value.
    train_batch_size: int = 8               # 8-16 (raise via gradient accumulation if GPU memory is tight)
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 5
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    relation_loss_weight: float = 1.0
    classification_loss_weight: float = 1.0
    negative_relation_ratio: float = 2.0    # negative:positive pair sampling ratio
    logging_steps: int = 50

    # ---- Architectural Disagreement ----------------------------------------
    overlap_aspect_weight: float = 0.4
    overlap_opinion_weight: float = 0.3
    category_match_weight: float = 0.15
    sentiment_match_weight: float = 0.15
    match_threshold: float = 0.3            # min agreement to pair gen<->ext quads

    # ---- Confidence Fusion --------------------------------------------------
    alpha: float = 0.4                      # weight on Conf_G
    beta: float = 0.4                       # weight on Conf_E
    gamma: float = 0.2                      # weight on Agreement
    final_score_threshold: float = 0.5

    # ---- Self-training / EMA student (RESERVED — not wired into train.py yet) --
    # These describe the NEXT stage on top of the Dual Teacher pipeline: an
    # EMA-updated student trained iteratively on pseudo labels, with a
    # two-tier (high/medium) confidence gate instead of the single
    # `final_score_threshold` Confidence Fusion uses today. Kept here as the
    # agreed starting values + search space so the next implementation pass
    # doesn't have to re-derive them; `run_pseudo_labeling()` in train.py
    # does not read any of these fields yet.
    backbone_size: str = "base"             # {"base", "large"}; scale up only after base is validated
    ema_momentum: float = 0.997             # search: {0.995, 0.997, 0.999}
    high_confidence_threshold: float = 0.90  # search: 0.85-0.95 — pseudo label kept as-is
    medium_confidence_threshold: float = 0.75  # search: 0.65-0.85 — kept but down-weighted
    consistency_stability_threshold: float = 0.80  # search: 0.70-0.90
    self_training_rounds: int = 3           # search: 2-4
    temperature_sampling_alpha: float = 0.5  # search: 0.3-0.7 — pseudo-label resampling temperature
    pseudo_weight_exponent: float = 1.0     # search: 0.5-2.0 — student loss weight = confidence**gamma

    # ---- runtime / IO ---------------------------------------------------
    output_dir: str = "checkpoints/dual-teacher"
    pseudo_labels_out: str = "checkpoints/dual-teacher/pseudo_labels.json"
    seed: int = 42
    num_workers: int = 2
    inference_batch_size: int = 16
    extra: List[str] = field(default_factory=list)
