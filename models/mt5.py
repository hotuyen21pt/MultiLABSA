"""mT5 wrapper for the Generative Teacher (T_G): loads the already
domain/task fine-tuned ``hotel-mt5`` ASQP checkpoint and turns free-text
generation into (aspect, opinion, category, sentiment) quads with a
sequence-probability confidence score.

Target linearization
---------------------
The fine-tuned checkpoint is assumed to have been trained to emit one quad
per clause, formatted as::

    aspect | opinion | category | sentiment ; aspect | opinion | category | sentiment ; ...

e.g. ``room | clean | FACILITY | Positive ; breakfast | disappointing | FACILITY | Negative``

This is the standard ASQP "paraphrase" linearization (Zhang et al., 2021) —
a flat, deterministic string an encoder-decoder can learn to emit and that a
regex can parse back losslessly. :func:`parse_quads` is deliberately
tolerant of minor formatting drift (extra spaces, trailing separators,
missing fields) since a generative model's raw output is never perfectly
well-formed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration

QUAD_SEP = " ; "
FIELD_SEP = " | "


# --------------------------------------------------------------------------- #
# Linearization <-> structured quads                                           #
# --------------------------------------------------------------------------- #
def linearize_quads(quads: List[dict]) -> str:
    """Serialize structured quads into the flat target string the decoder emits."""
    parts = []
    for q in quads:
        parts.append(FIELD_SEP.join([q["aspect"], q["opinion"], q["category"], q["sentiment"]]))
    return QUAD_SEP.join(parts)


def parse_quads(text: str) -> List[dict]:
    """Parse the decoder's raw output back into structured quads.

    Tolerant of: extra whitespace, a trailing/leading separator, ``;`` OR
    newline used between quads, and ``|`` OR ``,`` used between fields — a
    generative model's output drifts from the exact training format often
    enough that a strict parser would silently discard usable predictions.
    Malformed groups (wrong field count) are skipped rather than raising.
    """
    quads: List[dict] = []
    if not text or not text.strip():
        return quads

    groups = re.split(r"\s*;\s*|\n+", text.strip())
    for group in groups:
        group = group.strip().strip(";").strip()
        if not group:
            continue
        fields = [f.strip() for f in re.split(r"\s*\|\s*", group)]
        if len(fields) != 4:
            # fall back to comma-splitting for outputs that drifted to ","
            fields = [f.strip() for f in group.split(",")]
        if len(fields) != 4 or not all(fields):
            continue
        aspect, opinion, category, sentiment = fields
        quads.append({
            "aspect": aspect,
            "opinion": opinion,
            "category": category.upper(),
            "sentiment": sentiment.capitalize(),
        })
    return quads


# --------------------------------------------------------------------------- #
# Generation + sequence-probability confidence                                 #
# --------------------------------------------------------------------------- #
class MT5Generator:
    """Loads a fine-tuned mT5 ASQP checkpoint and generates quads + Conf_G."""

    def __init__(
        self,
        model_name_or_path: str = "hotel-mt5",
        device: "torch.device" = None,
        max_source_length: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_source_length = max_source_length

    @torch.no_grad()
    def generate_with_confidence(
        self,
        texts: List[str],
        max_new_tokens: int = 160,
        num_beams: int = 4,
    ) -> List[Tuple[str, float]]:
        """Generate one decoded string + Conf_G (sequence probability) per input text.

        Conf_G is the length-normalized sequence probability: the geometric
        mean of per-token conditional probabilities
        ``exp(mean_t log P(token_t | token_<t, source)))``. Length-normalizing
        (rather than summing raw log-probs) is essential — otherwise longer
        (more informative, often *more* correct) quad lists would always look
        less confident than short/empty ones purely because they have more
        chances to lose probability mass.
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # `compute_transition_scores` gives the log-prob HF actually assigned
        # to every generated token (properly accounting for beam-search
        # re-ranking via beam_indices) — this is the correct, numerically
        # stable way to recover sequence log-probability, rather than
        # re-running a forward pass and re-deriving it by hand.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            outputs.get("beam_indices"),
            normalize_logits=True,
        )

        # For an encoder-decoder model, sequences[:, 0] is the decoder start
        # token (not a generation step); scores/transition_scores align with
        # sequences[:, 1:] one-to-one.
        generated_tokens = outputs.sequences[:, 1:]
        pad_id = self.tokenizer.pad_token_id
        valid_mask = (generated_tokens != pad_id).float()
        # transition_scores can be -inf at pad positions post-EOS; zero those
        # out explicitly before averaging so they don't poison the mean.
        safe_scores = torch.where(
            torch.isfinite(transition_scores), transition_scores, torch.zeros_like(transition_scores)
        )
        token_counts = valid_mask.sum(dim=1).clamp(min=1.0)
        mean_log_prob = (safe_scores * valid_mask).sum(dim=1) / token_counts
        confidences = mean_log_prob.exp().clamp(max=1.0)

        decoded = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        return list(zip(decoded, confidences.tolist()))
