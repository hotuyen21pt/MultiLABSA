"""Fixed label vocabularies shared by both teachers.

Both the Generative Teacher (free-text generation, parsed post-hoc) and the
Extractive Teacher (closed-set classification heads) must agree on the same
category / sentiment / BIO tag inventories so their outputs are directly
comparable in ``teacher/disagreement.py``.

The category set was derived empirically from ``data_final/labeled_data``
(6 categories cover >99.9% of gold quads; the rare casing variants like
``"Amenity"`` are folded in by :func:`normalize_category`).
"""

from __future__ import annotations

from typing import Dict, List

# --------------------------------------------------------------------------- #
# Aspect category vocabulary                                                    #
# --------------------------------------------------------------------------- #
CATEGORIES: List[str] = [
    "FACILITY",
    "AMENITY",
    "EXPERIENCE",
    "SERVICE",
    "LOYALTY",
    "BRANDING",
]
CATEGORY2ID: Dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}
ID2CATEGORY: Dict[int, str] = {i: c for c, i in CATEGORY2ID.items()}

# --------------------------------------------------------------------------- #
# Sentiment polarity vocabulary                                                 #
# --------------------------------------------------------------------------- #
SENTIMENTS: List[str] = ["positive", "negative", "neutral"]
SENTIMENT2ID: Dict[str, int] = {s: i for i, s in enumerate(SENTIMENTS)}
ID2SENTIMENT: Dict[int, str] = {i: s for s, i in SENTIMENT2ID.items()}

# --------------------------------------------------------------------------- #
# BIO tagging scheme for the Span Head (one 3-way head per span type)          #
# --------------------------------------------------------------------------- #
BIO_TAGS: List[str] = ["O", "B", "I"]
BIO2ID: Dict[str, int] = {t: i for i, t in enumerate(BIO_TAGS)}
ID2BIO: Dict[int, str] = {i: t for t, i in BIO2ID.items()}

# Subword positions that must not contribute to the tagging loss (special
# tokens, padding, and continuation sub-word pieces — see utils/data.py).
IGNORE_INDEX: int = -100


def normalize_category(raw: str) -> str:
    """Fold casing noise (e.g. ``"Amenity"``) onto the canonical vocabulary.

    Unknown categories fall back to ``"FACILITY"`` (the majority class) rather
    than raising, so a single malformed row never crashes data loading.
    """
    key = (raw or "").strip().upper()
    return key if key in CATEGORY2ID else "FACILITY"


def normalize_sentiment(raw: str) -> str:
    """Fold casing noise (e.g. ``"Neutral"``) onto the canonical vocabulary."""
    key = (raw or "").strip().lower()
    return key if key in SENTIMENT2ID else "neutral"
