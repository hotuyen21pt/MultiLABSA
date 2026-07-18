"""The three prediction heads that sit on top of the XLM-R backbone for the
Extractive Teacher (T_E):

    1. SpanHead           - token-level BIO tagging for Aspect / Opinion spans
    2. RelationHead        - biaffine aspect<->opinion compatibility scoring
    3. ClassificationHead  - category + sentiment classification per pair

All three take the backbone's per-token hidden states as input; span pooling
(turning a token range into one vector) is provided here via
:func:`mean_pool` since it is the glue between SpanHead's output and the two
pair-level heads.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def mean_pool(hidden_states: torch.Tensor, token_spans: List[Tuple[int, int]]) -> torch.Tensor:
    """Mean-pool a single example's token hidden states over each span.

    Args:
        hidden_states: (seq_len, hidden_size) — ONE example (no batch dim).
        token_spans: list of (start_tok, end_tok) sub-word ranges, half-open.

    Returns:
        (num_spans, hidden_size). Averaging (rather than e.g. taking only the
        first sub-word) lets multi-token spans such as "a good size" or
        "very very clean" contribute every piece to the pooled representation.
    """
    if not token_spans:
        return hidden_states.new_zeros((0, hidden_states.size(-1)))
    pooled = [hidden_states[start:end].mean(dim=0) for start, end in token_spans]
    return torch.stack(pooled, dim=0)


# --------------------------------------------------------------------------- #
# 1. Span Head — Aspect / Opinion BIO tagging                                  #
# --------------------------------------------------------------------------- #
class SpanHead(nn.Module):
    """Two independent 3-way (O/B/I) token classifiers sharing the backbone.

    Aspect and Opinion spans are tagged by *separate* linear heads (rather
    than one 5-way O/B-ASP/I-ASP/B-OPN/I-OPN head) because a token can, in
    principle, belong to an opinion phrase that itself contains an aspect
    word (e.g. "room" appearing inside a longer descriptive opinion clause in
    another language's word order) — keeping the two tag spaces independent
    avoids forcing a token to choose only one role.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1, num_tags: int = 3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.aspect_classifier = nn.Linear(hidden_size, num_tags)
        self.opinion_classifier = nn.Linear(hidden_size, num_tags)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """hidden_states: (B, T, H) -> (aspect_logits, opinion_logits), each (B, T, 3)."""
        h = self.dropout(hidden_states)
        return self.aspect_classifier(h), self.opinion_classifier(h)


# --------------------------------------------------------------------------- #
# 2. Relation Head — associate each opinion with its aspect (biaffine)         #
# --------------------------------------------------------------------------- #
class RelationHead(nn.Module):
    """Biaffine scorer over every (aspect span, opinion span) pair in a sentence.

    Biaffine scoring (Dozat & Manning-style) is the standard architecture for
    span-pair relation extraction: it lets the aspect and opinion
    representations interact *multiplicatively* (via a learned bilinear
    term), not just additively, which is what a plain concatenation+MLP
    would give you — multiplicative interaction is what lets the head learn
    "does THIS opinion's semantics match THIS aspect's", not just "are both
    spans salient".

        score(a, o) = a^T U o + w_a^T a + w_o^T o + b

    ``U`` is the bilinear term, ``w_a``/``w_o`` are per-span bias terms that
    let a span be an intrinsically more/less "linkable" aspect or opinion
    regardless of its partner.
    """

    def __init__(self, hidden_size: int, proj_size: int = 256):
        super().__init__()
        self.aspect_proj = nn.Linear(hidden_size, proj_size)
        self.opinion_proj = nn.Linear(hidden_size, proj_size)
        self.activation = nn.Tanh()
        # Bilinear(P, P, 1): learns the pairwise interaction matrix U.
        self.bilinear = nn.Bilinear(proj_size, proj_size, 1)
        self.aspect_bias = nn.Linear(proj_size, 1, bias=False)
        self.opinion_bias = nn.Linear(proj_size, 1, bias=False)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, aspect_repr: torch.Tensor, opinion_repr: torch.Tensor) -> torch.Tensor:
        """Score every aspect-opinion pair for ONE example.

        Args:
            aspect_repr: (Na, hidden_size) pooled aspect-span vectors.
            opinion_repr: (No, hidden_size) pooled opinion-span vectors.

        Returns:
            (Na, No) raw logits — apply ``sigmoid`` for a link probability,
            or ``BCEWithLogitsLoss`` against a gold adjacency matrix to train.
        """
        num_aspects, num_opinions = aspect_repr.size(0), opinion_repr.size(0)
        if num_aspects == 0 or num_opinions == 0:
            return aspect_repr.new_zeros((num_aspects, num_opinions))

        a = self.activation(self.aspect_proj(aspect_repr))    # (Na, P)
        o = self.activation(self.opinion_proj(opinion_repr))  # (No, P)

        # Expand to every (aspect, opinion) pair so nn.Bilinear can score them
        # all in one batched call instead of a python double loop.
        proj_size = a.size(-1)
        a_exp = a.unsqueeze(1).expand(num_aspects, num_opinions, proj_size).reshape(-1, proj_size)
        o_exp = o.unsqueeze(0).expand(num_aspects, num_opinions, proj_size).reshape(-1, proj_size)
        biaffine = self.bilinear(a_exp, o_exp).view(num_aspects, num_opinions)

        # Broadcast per-span bias terms: (Na,1) + (1,No) -> (Na,No).
        scores = biaffine + self.aspect_bias(a) + self.opinion_bias(o).transpose(0, 1) + self.global_bias
        return scores


# --------------------------------------------------------------------------- #
# 3. Classification Head — category + sentiment per (aspect, opinion) pair     #
# --------------------------------------------------------------------------- #
class ClassificationHead(nn.Module):
    """Predicts aspect category and sentiment polarity for one linked pair.

    Input features concatenate the pooled aspect vector, the pooled opinion
    vector, and their elementwise (Hadamard) product — the product term gives
    the classifier a cheap, explicit "agreement" signal between the two spans
    on top of their raw content, which plain concatenation alone would make
    it re-derive from scratch.
    """

    def __init__(
        self,
        hidden_size: int,
        num_categories: int,
        num_sentiments: int,
        proj_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(hidden_size * 3, proj_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.category_out = nn.Linear(proj_size, num_categories)
        self.sentiment_out = nn.Linear(proj_size, num_sentiments)

    def forward(self, aspect_repr: torch.Tensor, opinion_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """aspect_repr / opinion_repr: (N, hidden_size), pair-aligned.

        Returns:
            (category_logits, sentiment_logits), each (N, num_classes).
        """
        features = torch.cat([aspect_repr, opinion_repr, aspect_repr * opinion_repr], dim=-1)
        hidden = self.dropout(self.activation(self.input_proj(features)))
        return self.category_out(hidden), self.sentiment_out(hidden)
