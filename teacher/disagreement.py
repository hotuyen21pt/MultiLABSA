"""Architectural Disagreement module.

Compares every quad the Generative Teacher (T_G) proposed against every quad
the Extractive Teacher (T_E) proposed for the *same* review, and produces a
single merged list of :class:`~utils.schema.MergedPrediction` where each quad
carries both teachers' confidence plus an ``agreement`` score in ``[0, 1]``.

Two effects implement the "increase/decrease confidence on agree/disagree"
requirement:

1. A gen/ext quad pair that matches well gets a HIGH ``agreement`` score,
   which is a direct positive term in ``teacher/confidence_fusion.py``'s
   ``FinalScore``.
2. A quad proposed by only ONE teacher gets ``agreement = 0`` AND the other
   teacher's confidence term is 0 (it never vouched for this quad) — so an
   uncorroborated quad's FinalScore is capped at ``alpha*Conf_G`` (or
   ``beta*Conf_E``) alone, strictly lower than a corroborated quad with
   comparable individual confidences.

Hallucination filtering (generative aspects that do not exist anywhere in
the source review) happens here, BEFORE any matching — it is a hard
grounding check, not a soft disagreement signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from utils.schema import MergedPrediction, QuadPrediction
from utils.text_alignment import contains_phrase, token_jaccard


@dataclass
class DisagreementWeights:
    """Weights for the 4 agreement components; must sum to 1.0 to keep
    ``agreement`` itself bounded in ``[0, 1]`` (a precondition for it to be
    combined linearly with Conf_G/Conf_E, which are also in ``[0, 1]``)."""

    aspect_overlap: float = 0.4
    opinion_overlap: float = 0.3
    category_match: float = 0.15
    sentiment_match: float = 0.15
    match_threshold: float = 0.3   # min agreement score to accept a gen<->ext match


def _pair_agreement(g: QuadPrediction, e: QuadPrediction, weights: DisagreementWeights) -> float:
    """Weighted agreement between one generative quad and one extractive quad."""
    aspect_overlap = token_jaccard(g.aspect, e.aspect)
    opinion_overlap = token_jaccard(g.opinion, e.opinion)
    category_match = 1.0 if g.category.upper() == e.category.upper() else 0.0
    sentiment_match = 1.0 if g.sentiment.lower() == e.sentiment.lower() else 0.0
    return (
        weights.aspect_overlap * aspect_overlap
        + weights.opinion_overlap * opinion_overlap
        + weights.category_match * category_match
        + weights.sentiment_match * sentiment_match
    )


def filter_hallucinations(
    gen_quads: List[QuadPrediction], source_text: str
) -> Tuple[List[QuadPrediction], int]:
    """Drop generative quads whose aspect term is never actually mentioned
    in the source review. Returns ``(kept_quads, num_dropped)``.
    """
    kept = [q for q in gen_quads if contains_phrase(source_text, q.aspect)]
    return kept, len(gen_quads) - len(kept)


def compute_agreement(
    gen_quads: List[QuadPrediction],
    ext_quads: List[QuadPrediction],
    source_text: str,
    weights: DisagreementWeights = None,
) -> List[MergedPrediction]:
    """Merge one review's two teacher outputs into agreement-scored quads.

    Matching is greedy 1-to-1: each generative quad claims its best
    still-unclaimed extractive quad (if the match score clears
    ``weights.match_threshold``), so no extractive quad corroborates two
    different generative quads and vice versa.
    """
    weights = weights or DisagreementWeights()
    gen_quads, _num_hallucinated = filter_hallucinations(gen_quads, source_text)

    matched_ext_indices: set = set()
    merged: List[MergedPrediction] = []

    for g in gen_quads:
        best_idx, best_score = None, 0.0
        for j, e in enumerate(ext_quads):
            if j in matched_ext_indices:
                continue
            score = _pair_agreement(g, e, weights)
            if score > best_score:
                best_idx, best_score = j, score

        if best_idx is not None and best_score >= weights.match_threshold:
            e = ext_quads[best_idx]
            matched_ext_indices.add(best_idx)
            merged.append(
                MergedPrediction(
                    aspect=g.aspect,
                    opinion=g.opinion,
                    category=g.category,
                    sentiment=g.sentiment,
                    conf_g=g.confidence,
                    conf_e=e.confidence,
                    agreement=best_score,
                    sources=["generative", "extractive"],
                )
            )
        else:
            # No corroborating extractive quad: agreement=0, conf_e=0 — this
            # quad's FinalScore can only ever come from Conf_G alone.
            merged.append(
                MergedPrediction(
                    aspect=g.aspect,
                    opinion=g.opinion,
                    category=g.category,
                    sentiment=g.sentiment,
                    conf_g=g.confidence,
                    conf_e=0.0,
                    agreement=0.0,
                    sources=["generative"],
                )
            )

    for j, e in enumerate(ext_quads):
        if j in matched_ext_indices:
            continue
        # Extractive-only proposal: symmetric treatment, conf_g=0.
        merged.append(
            MergedPrediction(
                aspect=e.aspect,
                opinion=e.opinion,
                category=e.category,
                sentiment=e.sentiment,
                conf_g=0.0,
                conf_e=e.confidence,
                agreement=0.0,
                sources=["extractive"],
            )
        )

    return merged
