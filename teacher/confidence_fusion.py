"""Confidence Fusion.

Combines each merged (T_G, T_E) quad into one scalar:

    FinalScore = alpha * Conf_G + beta * Conf_E + gamma * Agreement

and keeps only quads with ``FinalScore > threshold`` as pseudo labels. This
is the final gate of the whole pipeline: everything upstream (generation,
extraction, disagreement) only exists to produce the three terms fused here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from utils.schema import MergedPrediction


@dataclass
class FusionWeights:
    alpha: float = 0.4   # weight on Conf_G (generative sequence probability)
    beta: float = 0.4    # weight on Conf_E (extractive classification confidence)
    gamma: float = 0.2   # weight on Agreement (cross-teacher consistency)
    threshold: float = 0.5

    def __post_init__(self) -> None:
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"alpha + beta + gamma must sum to 1.0, got {total}")


def final_score(m: MergedPrediction, weights: FusionWeights) -> float:
    return weights.alpha * m.conf_g + weights.beta * m.conf_e + weights.gamma * m.agreement


def fuse(merged_predictions: List[MergedPrediction], weights: FusionWeights = None) -> List[dict]:
    """Score every merged quad and keep those above ``weights.threshold``.

    Returns plain dicts (ready for ``json.dump``), sorted by descending
    FinalScore, each carrying the full audit trail (Conf_G, Conf_E,
    Agreement, FinalScore, contributing sources) so a human/downstream
    student model can inspect *why* a pseudo label was accepted.
    """
    weights = weights or FusionWeights()
    scored: List[dict] = []
    for m in merged_predictions:
        score = final_score(m, weights)
        if score > weights.threshold:
            record = m.to_dict()
            record["final_score"] = round(score, 4)
            scored.append(record)
    scored.sort(key=lambda r: r["final_score"], reverse=True)
    return scored
