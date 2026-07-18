"""Teacher 1 — Generative Teacher (T_G).

Wraps the already fine-tuned ``hotel-mt5`` ASQP checkpoint: given a review,
it free-text-generates every (aspect, opinion, category, sentiment) quad in
one shot, and reports a confidence score derived from the sequence's
generation log-probability (Conf_G).

This module does **not** filter hallucinations or cross-check against the
Extractive Teacher — that comparison is the explicit job of
``teacher/disagreement.py``. Keeping the two concerns separate means this
class can be tested/used standalone (e.g. for a purely generative baseline).
"""

from __future__ import annotations

from typing import List, Optional

import torch

from models.mt5 import MT5Generator, parse_quads
from utils.label_maps import normalize_category, normalize_sentiment
from utils.schema import QuadPrediction


class GenerativeTeacher:
    """High-level API: ``review text -> List[QuadPrediction]``."""

    def __init__(
        self,
        model_name_or_path: str = "hotel-mt5",
        device: Optional["torch.device"] = None,
        max_source_length: int = 256,
        max_new_tokens: int = 160,
        num_beams: int = 4,
    ):
        self.generator = MT5Generator(model_name_or_path, device=device, max_source_length=max_source_length)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    def predict(self, texts: List[str]) -> List[List[QuadPrediction]]:
        """Batch inference: one quad list per input review.

        Every quad in the list shares the *same* Conf_G — the log-probability
        is a property of the whole generated sequence (mT5 emits all quads in
        a single autoregressive decode), not of an individual quad.
        """
        generations = self.generator.generate_with_confidence(
            texts, max_new_tokens=self.max_new_tokens, num_beams=self.num_beams
        )

        results: List[List[QuadPrediction]] = []
        for decoded_text, seq_confidence in generations:
            raw_quads = parse_quads(decoded_text)
            preds = [
                QuadPrediction(
                    aspect=q["aspect"],
                    opinion=q["opinion"],
                    category=normalize_category(q["category"]),
                    sentiment=normalize_sentiment(q["sentiment"]),
                    confidence=seq_confidence,
                    source="generative",
                )
                for q in raw_quads
            ]
            results.append(preds)
        return results
