"""Span corruption for Multilingual Hotel-DAPT — mT5-style denoising.

Given a hotel review, we mask several random spans of 1-5 tokens (~15% of the
sequence) and replace each with a sentinel token ``<extra_id_i>``.  The target
sequence is the concatenation of ``<extra_id_i>`` followed by the tokens that
were masked, ending with a final sentinel — exactly the mT5 format.

On top of *random* span selection we **raise the probability** of masking spans
that contain domain-salient terms (hotel terminology, opinion words, negation,
intensifiers) via a lexicon.  When a review contains none of these terms the
per-token weights are uniform, so the process degrades gracefully to plain
random span corruption (the required fallback).

Note on the mT5 target format
------------------------------
The illustrative example in the task shows only ``<extra_id_0> room`` /
``<extra_id_1> breakfast``.  Real mT5 additionally appends a *final* sentinel
(here ``<extra_id_2>``) that marks the end of the reconstructed content; we
follow the real mT5 convention and also append the EOS token to both sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
from transformers import PreTrainedTokenizerBase

from utils import build_lexicon

# SentencePiece marks a word start with this "lower one-eighth block" character.
_SP_UNDERLINE = "▁"  # "▁"


def get_sentinel_token_ids(
    tokenizer: PreTrainedTokenizerBase, num_sentinels: int = 100
) -> List[int]:
    """Return the ids of ``<extra_id_0>`` .. ``<extra_id_{n-1}>`` for any T5/mT5.

    Different tokenizers store the sentinels differently:
        * standard T5 keeps them as added tokens named ``<extra_id_i>``;
        * mT5's SentencePiece stores them inside the vocab as ``▁<extra_id_i>``
          (with the word-start marker), descending from the top of the vocab.

    We look up both spellings in the vocab and only fall back to the T5
    "sentinels occupy the top ids" convention if neither is present — this
    avoids silently collapsing every sentinel onto the ``<unk>`` id.
    """
    vocab = tokenizer.get_vocab()
    ids: List[int] = []
    for i in range(num_sentinels):
        token_id = None
        for form in (f"<extra_id_{i}>", f"{_SP_UNDERLINE}<extra_id_{i}>"):
            if form in vocab:
                token_id = vocab[form]
                break
        if token_id is None:  # last-resort T5 convention
            token_id = tokenizer.vocab_size - 1 - i
        ids.append(token_id)
    return ids


@dataclass
class MaskedExample:
    """A single corrupted example (token ids plus decoded text for debugging)."""

    input_ids: List[int]     # corrupted input (sentinels replace masked spans)
    labels: List[int]        # mT5 target: <extra_id_i> + span tokens + final sentinel
    input_text: str          # decoded ``input_ids`` (illustration only)
    target_text: str         # decoded ``labels`` (illustration only)


class SpanCorruption:
    """Apply mT5-style span corruption with optional lexicon-biased masking.

    Args:
        tokenizer: an mT5 tokenizer (provides the ``<extra_id_i>`` sentinels).
        noise_density: fraction of tokens to mask (~0.15).
        max_span_length: maximum span length; spans are sampled uniformly in
            ``[1, max_span_length]``.
        lexicon_boost: multiplier applied to the start-weight of tokens that
            belong to a lexicon term (``1.0`` disables biasing).
        extra_lexicon_terms: additional domain terms to bias towards.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float = 0.15,
        max_span_length: int = 5,
        lexicon_boost: float = 5.0,
        extra_lexicon_terms: Optional[List[str]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.max_span_length = max_span_length
        self.lexicon_boost = lexicon_boost
        self.lexicon: Set[str] = build_lexicon(extra_lexicon_terms)

        # Cache the sentinel ids (<extra_id_0> .. <extra_id_99>).  mT5 provides
        # exactly 100 of them; this bounds the number of spans per example.
        self.sentinel_ids: List[int] = get_sentinel_token_ids(tokenizer, 100)
        self.max_sentinels = len(self.sentinel_ids)
        self.eos_id: int = tokenizer.eos_token_id

    # ------------------------------------------------------------------ #
    # Lexicon-based per-token weighting                                    #
    # ------------------------------------------------------------------ #
    def _token_weights(self, token_ids: Sequence[int]) -> np.ndarray:
        """Return a start-weight per token, boosting tokens inside lexicon terms.

        SentencePiece marks a word start with the ``▁`` prefix, so we can rebuild
        whole words from consecutive sub-word pieces without an offset map and
        check each reconstructed word against the lexicon.
        """
        pieces = self.tokenizer.convert_ids_to_tokens(list(token_ids))
        n = len(pieces)
        weights = np.ones(n, dtype=np.float64)

        i = 0
        while i < n:
            # a word runs from a word-start piece up to the next one
            j = i + 1
            while j < n and not pieces[j].startswith(_SP_UNDERLINE):
                j += 1
            word = "".join(pieces[i:j]).replace(_SP_UNDERLINE, "").lower()
            if word and word in self.lexicon:
                weights[i:j] = self.lexicon_boost
            i = j
        return weights

    # ------------------------------------------------------------------ #
    # Core masking                                                         #
    # ------------------------------------------------------------------ #
    def _sample_span_lengths(self, num_mask: int) -> List[int]:
        """Split ``num_mask`` masked tokens into spans of length 1..max."""
        lengths: List[int] = []
        remaining = num_mask
        while remaining > 0:
            high = min(self.max_span_length, remaining)
            length = int(np.random.randint(1, high + 1))  # high is inclusive
            lengths.append(length)
            remaining -= length
        return lengths

    def _place_spans(self, weights: np.ndarray, span_lengths: List[int]) -> np.ndarray:
        """Choose non-overlapping, non-adjacent span positions weighted by ``weights``.

        Spans must be separated by at least one un-masked token so that adjacent
        sentinels never merge (an mT5 requirement).  Longer spans are placed
        first as they are harder to fit.
        """
        n = len(weights)
        masked = np.zeros(n, dtype=bool)

        for length in sorted(span_lengths, reverse=True):
            if masked.sum() >= n - 1:  # always keep >=1 token unmasked
                break
            candidates: List[int] = []
            cand_weights: List[float] = []
            for start in range(0, n - length + 1):
                end = start + length
                if masked[start:end].any():
                    continue
                if start - 1 >= 0 and masked[start - 1]:  # left neighbour
                    continue
                if end < n and masked[end]:               # right neighbour
                    continue
                candidates.append(start)
                # region weight => spans covering lexicon terms are favoured
                cand_weights.append(float(weights[start:end].sum()))
            if not candidates:
                continue
            probs = np.asarray(cand_weights)
            probs = probs / probs.sum()
            chosen = candidates[int(np.random.choice(len(candidates), p=probs))]
            masked[chosen:chosen + length] = True

        # Fallback: guarantee at least one masked token for a usable target.
        if not masked.any() and n >= 1:
            masked[int(np.random.randint(0, n))] = True
        return masked

    def mask_tokens(self, token_ids: Sequence[int]) -> Tuple[List[int], List[int]]:
        """Corrupt a token sequence, returning ``(input_ids, labels)``.

        ``input_ids`` is the review with each masked span replaced by a sentinel;
        ``labels`` is the mT5 target (``<extra_id_i>`` + span tokens …) plus a
        final sentinel.  Both end with EOS.
        """
        token_ids = list(token_ids)
        n = len(token_ids)

        # Too short to corrupt meaningfully — return as-is with EOS.
        if n < 2:
            return token_ids + [self.eos_id], token_ids + [self.eos_id]

        num_mask = max(1, int(round(self.noise_density * n)))
        num_mask = min(num_mask, n - 1)  # keep at least one visible token

        span_lengths = self._sample_span_lengths(num_mask)
        # Never need more spans than we have sentinels.
        if len(span_lengths) > self.max_sentinels:
            span_lengths = span_lengths[: self.max_sentinels]

        weights = self._token_weights(token_ids)
        masked = self._place_spans(weights, span_lengths)

        # Walk the sequence, emitting sentinels for masked runs.
        input_ids: List[int] = []
        labels: List[int] = []
        sentinel = 0
        i = 0
        while i < n:
            if masked[i] and sentinel < self.max_sentinels:
                sid = self.sentinel_ids[sentinel]
                input_ids.append(sid)
                labels.append(sid)
                while i < n and masked[i]:
                    labels.append(int(token_ids[i]))
                    i += 1
                sentinel += 1
            else:
                input_ids.append(int(token_ids[i]))
                i += 1

        # Final trailing sentinel marks the end of reconstructed content (mT5).
        labels.append(self.sentinel_ids[min(sentinel, self.max_sentinels - 1)])

        # Append EOS to both sides, as in the reference mT5 data pipeline.
        input_ids.append(self.eos_id)
        labels.append(self.eos_id)
        return input_ids, labels

    # ------------------------------------------------------------------ #
    # Convenience / illustration                                           #
    # ------------------------------------------------------------------ #
    def generate_masked_example(self, text: str) -> MaskedExample:
        """Tokenize ``text``, corrupt it, and return ids + decoded strings.

        Handy for inspection/unit tests — the decoded ``input_text`` and
        ``target_text`` show the exact mT5 formatting.
        """
        token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        input_ids, labels = self.mask_tokens(token_ids)
        return MaskedExample(
            input_ids=input_ids,
            labels=labels,
            input_text=self.tokenizer.decode(input_ids),
            target_text=self.tokenizer.decode(labels),
        )


# Module-level convenience wrapper mirroring the class method, so callers can
# ``from masking import generate_masked_example`` per the task's file spec.
def generate_masked_example(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    noise_density: float = 0.15,
    max_span_length: int = 5,
    lexicon_boost: float = 5.0,
) -> MaskedExample:
    """Stateless helper that builds a :class:`SpanCorruption` and runs it once."""
    corrupter = SpanCorruption(
        tokenizer=tokenizer,
        noise_density=noise_density,
        max_span_length=max_span_length,
        lexicon_boost=lexicon_boost,
    )
    return corrupter.generate_masked_example(text)
