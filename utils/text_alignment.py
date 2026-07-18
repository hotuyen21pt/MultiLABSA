"""Word/span/token alignment helpers shared by the extractive pipeline.

Gold annotations in ``data_final/labeled_data`` express Aspect_span /
Opinion_span as **word-level** ``[start, end)`` indices over
``review.split()``. XLM-R tokenizes into sub-words, so every span has to be
carried through three coordinate systems:

    word index  --(tokenizer word_ids)-->  sub-word token index

This module owns that bookkeeping plus the fuzzy text-overlap metrics used
by ``teacher/disagreement.py`` to compare a free-text generative quad
against a span-grounded extractive quad (which have no shared coordinate
system at all — only their surface text).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from utils.label_maps import BIO2ID, ID2BIO, IGNORE_INDEX

_PUNCT_RE = re.compile(r"[^\w]+", re.UNICODE)


# --------------------------------------------------------------------------- #
# Word tokenisation (must match how the gold spans were produced: split())     #
# --------------------------------------------------------------------------- #
def word_tokenize(text: str) -> List[str]:
    """Whitespace tokenizer — matches the convention used to build gold spans."""
    return text.split()


def span_text(words: Sequence[str], start: int, end: int) -> str:
    """Surface text of the word range ``[start, end)``."""
    return " ".join(words[start:end])


def normalize(text: str) -> str:
    """Lowercase + strip punctuation, for fuzzy comparisons only."""
    return _PUNCT_RE.sub(" ", text.lower()).strip()


def token_set(text: str) -> set:
    return {t for t in normalize(text).split() if t}


# --------------------------------------------------------------------------- #
# Fuzzy overlap metrics (generative text  <->  extractive span text)           #
# --------------------------------------------------------------------------- #
def token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity of the two strings' (normalized) word sets.

    Used to score how well a generative teacher's free-text aspect/opinion
    matches an extractive teacher's span-grounded one, since they do not
    share token boundaries.
    """
    sa, sb = token_set(a), token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def contains_phrase(haystack: str, needle: str) -> bool:
    """True if every content word of ``needle`` occurs somewhere in ``haystack``.

    This is the grounding check used to discard hallucinated aspects: a
    generative aspect term that the source review never actually mentions.
    Token-subset (rather than raw substring) containment tolerates the mT5
    decoder paraphrasing morphology/spacing while still catching genuine
    hallucinations.
    """
    needle_tokens = token_set(needle)
    if not needle_tokens:
        return False
    haystack_tokens = token_set(haystack)
    return needle_tokens.issubset(haystack_tokens)


# --------------------------------------------------------------------------- #
# BIO encode/decode at word level                                              #
# --------------------------------------------------------------------------- #
def spans_to_bio(num_words: int, spans: Sequence[Tuple[int, int]]) -> List[str]:
    """Encode a list of non-overlapping ``[start, end)`` word spans as BIO tags."""
    tags = ["O"] * num_words
    for start, end in spans:
        start = max(0, start)
        end = min(num_words, end)
        for i in range(start, end):
            tags[i] = "B" if i == start else "I"
    return tags


def bio_to_spans(tags: Sequence[str]) -> List[Tuple[int, int]]:
    """Decode a BIO tag sequence back into ``[start, end)`` word spans.

    Robust to a stray leading ``I`` (no preceding ``B``): it is treated as the
    start of a new span, since a model at inference time can emit an
    ill-formed sequence and we must not silently drop the evidence.
    """
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, tag in enumerate(list(tags) + ["O"]):  # sentinel flush
        if tag == "B" or (tag == "I" and start is None):
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag == "O":
            if start is not None:
                spans.append((start, i))
            start = None
        # tag == "I" with start set: span continues, nothing to do
    return spans


# --------------------------------------------------------------------------- #
# Word-index span  <->  sub-word token-index span (needs tokenizer word_ids)   #
# --------------------------------------------------------------------------- #
def word_span_to_token_indices(word_ids: Sequence[Optional[int]], start: int, end: int) -> List[int]:
    """All sub-word token positions covered by the word range ``[start, end)``.

    ``word_ids`` is the per-token word-index mapping returned by a fast
    tokenizer's ``BatchEncoding.word_ids(batch_index=i)`` when the input was
    built with ``is_split_into_words=True``. Special tokens map to ``None``
    and are always excluded.
    """
    return [i for i, w in enumerate(word_ids) if w is not None and start <= w < end]


def first_subword_positions(word_ids: Sequence[Optional[int]]) -> Dict[int, int]:
    """Map ``word_index -> the first sub-word token position representing it``.

    Standard NER-style convention: only the first sub-word piece of a word
    carries a supervised BIO label; the remaining pieces are masked out of
    the loss (see :data:`utils.label_maps.IGNORE_INDEX`) so the model is not
    penalised for the (arbitrary) tag of a continuation piece.
    """
    mapping: Dict[int, int] = {}
    for pos, w in enumerate(word_ids):
        if w is not None and w not in mapping:
            mapping[w] = pos
    return mapping


def word_tags_to_subword_labels(
    word_ids: Sequence[Optional[int]], word_tags: Sequence[str]
) -> List[int]:
    """Project word-level BIO tags onto sub-word positions for loss computation.

    Only the first sub-word of each word is labelled; every other position
    (special tokens, padding, continuation pieces) gets
    :data:`utils.label_maps.IGNORE_INDEX`.
    """
    first_pos = first_subword_positions(word_ids)
    labels = [IGNORE_INDEX] * len(word_ids)
    for word_idx, pos in first_pos.items():
        if word_idx < len(word_tags):
            labels[pos] = BIO2ID[word_tags[word_idx]]
    return labels


def decode_word_bio_from_subword_logits(
    word_ids: Sequence[Optional[int]], tag_ids: Sequence[int]
) -> List[str]:
    """Inverse of :func:`word_tags_to_subword_labels`: read the tag predicted
    at each word's first sub-word position back out into a word-level BIO
    sequence (mirrors training exactly, so train/inference stay consistent).
    """
    first_pos = first_subword_positions(word_ids)
    if not first_pos:
        return []
    num_words = max(first_pos.keys()) + 1
    tags = ["O"] * num_words
    for word_idx, pos in first_pos.items():
        tags[word_idx] = ID2BIO[int(tag_ids[pos])]
    return tags
