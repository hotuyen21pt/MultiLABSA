"""Shared utilities for Stage 0 — Multilingual Hotel Domain-Adaptive Pretraining.

Contains:
    * ``Config``           - a single dataclass holding every hyper-parameter.
    * ``set_seed``         - reproducible RNG seeding across random/numpy/torch.
    * ``get_device`` / ``resolve_precision`` - hardware & mixed-precision helpers.
    * ``setup_logging``    - a consistent logger for the whole pipeline.
    * ``LEXICON`` / ``build_lexicon`` - the multilingual biased-masking lexicon.
    * ``count_parameters`` - a small model-size reporter.

This module is intentionally free of any task-specific (ASQP/ACOS/…) logic — it
only supports the denoising Domain-Adaptive Pretraining (DAPT) objective.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

# NOTE: heavy deps (numpy/torch) are imported lazily inside the functions that
# need them, so lightweight consumers (e.g. build_lexicon.py, which only needs
# LEXICON / Config / setup_logging) can import this module without torch.
if TYPE_CHECKING:  # for type checkers only; no runtime import
    import torch


# --------------------------------------------------------------------------- #
# Reproducibility & hardware                                                    #
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs so a run is reproducible.

    The span-corruption masking draws from the *global* NumPy RNG, so seeding
    it here makes the whole masking stream deterministic for a given seed while
    still varying from one example to the next.
    """
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
        # bf16 is preferred: same range as fp32, so no loss scaling needed.
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    return "fp32"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Create/return the package logger with a single stream handler."""
    logger = logging.getLogger("hotel_dapt")
    if not logger.handlers:  # avoid duplicate handlers on re-import
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def count_parameters(model: "torch.nn.Module") -> int:
    """Count trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Multilingual biased-masking lexicon                                           #
# --------------------------------------------------------------------------- #
# Words the span-corruption process should mask *more often* than random tokens.
# Grouped only for readability/extensibility — the masker treats them as one set.
# The corpus is dominated by English + Vietnamese (~77%), so those two are the
# most complete; other languages carry a representative seed set. Extend freely.
LEXICON: Dict[str, List[str]] = {
    # ---- hotel terminology (nouns / noun-phrase heads) --------------------- #
    "hotel_terms": [
        # English
        "room", "rooms", "hotel", "breakfast", "staff", "reception", "bed",
        "bathroom", "shower", "pool", "location", "service", "wifi", "price",
        "view", "lobby", "checkin", "check-in", "checkout", "check-out",
        "restaurant", "food", "parking", "elevator", "air", "conditioning",
        "towel", "towels", "bar", "spa", "beach", "balcony", "kitchen",
        # Vietnamese
        "phòng", "khách", "sạn", "khách_sạn", "nhân", "viên", "nhân_viên",
        "lễ", "tân", "lễ_tân", "giường", "phòng_tắm", "vòi", "sen", "hồ",
        "bơi", "vị", "trí", "dịch", "vụ", "bữa", "sáng", "ăn", "đồ_ăn",
        "nhà", "hàng", "bãi", "biển", "ban", "công", "giá", "cả",
        # French / German / Spanish (seed)
        "chambre", "hôtel", "personnel", "petit-déjeuner", "piscine",
        "zimmer", "frühstück", "personal", "habitación", "desayuno",
        "personal", "recepción", "playa", "piscina",
    ],
    # ---- opinion / sentiment words ----------------------------------------- #
    "opinions": [
        # English
        "clean", "dirty", "excellent", "terrible", "great", "bad", "good",
        "amazing", "awful", "wonderful", "horrible", "comfortable",
        "uncomfortable", "friendly", "rude", "spacious", "cramped", "noisy",
        "quiet", "nice", "poor", "perfect", "disappointing", "lovely",
        "helpful", "beautiful", "old", "modern", "cheap", "expensive",
        # Vietnamese
        "sạch", "sẽ", "sạch_sẽ", "bẩn", "tốt", "tệ", "tuyệt", "vời",
        "tuyệt_vời", "đẹp", "xấu", "thoải", "mái", "thoải_mái", "ồn",
        "yên", "tĩnh", "thân", "thiện", "thân_thiện", "nhiệt", "tình",
        "rộng", "rãi", "chật", "cũ", "mới", "rẻ", "đắt", "ngon", "chu_đáo",
        # French / German / Spanish (seed)
        "propre", "sale", "excellent", "sauber", "schmutzig", "limpio",
        "sucio", "bueno", "malo",
    ],
    # ---- negation ---------------------------------------------------------- #
    "negations": [
        "not", "never", "no", "none", "nothing", "without", "cannot", "can't",
        "didn't", "wasn't", "isn't", "don't", "doesn't",
        # Vietnamese
        "không", "chẳng", "chưa", "đừng",
        # French / German / Spanish (seed)
        "ne", "pas", "nicht", "kein", "no", "nunca", "sin",
    ],
    # ---- intensifiers ------------------------------------------------------ #
    "intensifiers": [
        "very", "extremely", "really", "so", "too", "quite", "absolutely",
        "highly", "incredibly", "super",
        # Vietnamese
        "rất", "quá", "cực", "cực_kỳ", "vô_cùng", "hơi", "khá",
        # French / German / Spanish (seed)
        "très", "sehr", "muy", "demasiado",
    ],
}


def build_lexicon(extra_terms: Optional[List[str]] = None) -> Set[str]:
    """Flatten :data:`LEXICON` into a lowercase lookup set.

    Args:
        extra_terms: optional additional terms to merge in.

    Returns:
        A set of lowercased terms. Multi-word entries are stored both with an
        underscore (``khách_sạn``) so they can match a reconstructed
        SentencePiece word, and split into their parts so single tokens match.
    """
    terms: Set[str] = set()
    for group in LEXICON.values():
        for term in group:
            terms.add(term.lower())
            # also index the individual pieces of underscore/space compounds
            for piece in term.replace("-", "_").split("_"):
                if piece:
                    terms.add(piece.lower())
    if extra_terms:
        terms.update(t.lower() for t in extra_terms)
    return terms


# --------------------------------------------------------------------------- #
# Configuration                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """All hyper-parameters for a DAPT run (populated from CLI args)."""

    # ---- data ---------------------------------------------------------------
    data_dir: str = "data_final/unlabeled_data"
    text_column: str = "review"
    language_column: str = "language"
    file_glob: str = "*_lang.csv"
    sampling_temperature: float = 2.0     # higher => more language-balanced
    min_chars: int = 10                   # drop reviews shorter than this
    max_samples_per_language: Optional[int] = None  # cap for memory control
    val_fraction: float = 0.01            # per-language hold-out for validation
    max_val_samples: int = 5_000          # cap the (fixed) validation set
    epoch_size: Optional[int] = None      # #train draws per epoch (None => all)

    # ---- model / tokenizer --------------------------------------------------
    model_name: str = "google/mt5-base"
    max_seq_length: int = 256             # max raw tokens before corruption

    # ---- span corruption ----------------------------------------------------
    noise_density: float = 0.15           # ~15% of tokens masked
    max_span_length: int = 5              # spans are 1..5 tokens long
    lexicon_boost: float = 5.0            # bias strength: weight = 1 + boost*salience
    lexicon_file: Optional[str] = None    # data-driven lexicon from build_lexicon.py

    # ---- optimisation -------------------------------------------------------
    optimizer: str = "adamw"              # adamw|adafactor (adafactor: ~4x less optimizer-state memory)
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 1_000
    num_epochs: int = 3
    max_steps: Optional[int] = None       # if set, overrides num_epochs
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    precision: str = "auto"               # auto|bf16|fp16|fp32

    # ---- runtime / IO -------------------------------------------------------
    output_dir: str = "checkpoints/hotel-dapt"
    final_dir: str = "hotel-mt5"
    logging_steps: int = 50
    eval_steps: int = 1_000
    save_steps: int = 1_000
    save_total_limit: int = 3             # keep only the N most recent ckpts
    num_workers: int = 2
    seed: int = 42
    gradient_checkpointing: bool = False
    extra_lexicon_terms: List[str] = field(default_factory=list)
