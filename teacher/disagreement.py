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


def _agreement_breakdown(
    g: QuadPrediction, e: QuadPrediction, weights: DisagreementWeights
) -> Tuple[float, dict]:
    """Weighted agreement between a gen and an ext quad, plus its component breakdown.

    The breakdown (per-component overlaps/matches) is carried onto the merged
    quad so ``teacher/reliability.py`` can derive element-wise reliability from
    the same signals, rather than recomputing them.
    """
    aspect_overlap = token_jaccard(g.aspect, e.aspect)
    opinion_overlap = token_jaccard(g.opinion, e.opinion)
    category_match = 1.0 if g.category.upper() == e.category.upper() else 0.0
    sentiment_match = 1.0 if g.sentiment.lower() == e.sentiment.lower() else 0.0
    score = (
        weights.aspect_overlap * aspect_overlap
        + weights.opinion_overlap * opinion_overlap
        + weights.category_match * category_match
        + weights.sentiment_match * sentiment_match
    )
    breakdown = {
        "aspect_overlap": aspect_overlap,
        "opinion_overlap": opinion_overlap,
        "category_match": category_match,
        "sentiment_match": sentiment_match,
    }
    return score, breakdown


def _pair_agreement(g: QuadPrediction, e: QuadPrediction, weights: DisagreementWeights) -> float:
    """Weighted agreement between one generative quad and one extractive quad."""
    return _agreement_breakdown(g, e, weights)[0]


def _optimal_assignment(score_matrix: List[List[float]], threshold: float) -> List[Tuple[int, int]]:
    """Globally-optimal 1-to-1 assignment maximising total agreement (MERA-XQUAD §4.4).

    Returns the list of ``(gen_idx, ext_idx)`` pairs whose score clears
    ``threshold``. Greedy matching (claim-your-best-first) can be globally
    sub-optimal — one quad grabbing a shared partner blocks a better overall
    pairing — so we solve the assignment problem instead:

        * ``scipy.optimize.linear_sum_assignment`` (Hungarian) when SciPy is
          installed — the intended implementation;
        * an exact brute force over the smaller dimension when the matrix is
          tiny (the normal case: a review has only a handful of quads), so the
          optimum is still reached with zero extra dependencies;
        * a greedy fallback only for the (unrealistic) large-matrix case.
    """
    n = len(score_matrix)
    m = len(score_matrix[0]) if n else 0
    if n == 0 or m == 0:
        return []

    try:  # preferred: Hungarian algorithm
        import numpy as _np
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-_np.asarray(score_matrix, dtype=float))
        return [(int(i), int(j)) for i, j in zip(rows, cols) if score_matrix[i][j] >= threshold]
    except Exception:
        pass

    if min(n, m) <= 7:  # exact optimum, dependency-free (quad counts are tiny)
        import itertools

        best_pairs: List[Tuple[int, int]] = []
        best_total = -1.0
        if n <= m:
            for cols in itertools.permutations(range(m), n):
                total = sum(score_matrix[i][cols[i]] for i in range(n))
                if total > best_total:
                    best_total, best_pairs = total, [(i, cols[i]) for i in range(n)]
        else:
            for rows in itertools.permutations(range(n), m):
                total = sum(score_matrix[rows[j]][j] for j in range(m))
                if total > best_total:
                    best_total, best_pairs = total, [(rows[j], j) for j in range(m)]
        return [(i, j) for i, j in best_pairs if score_matrix[i][j] >= threshold]

    # greedy fallback (large matrices only)
    ranked = sorted(
        ((score_matrix[i][j], i, j) for i in range(n) for j in range(m)), reverse=True
    )
    used_i: set = set()
    used_j: set = set()
    pairs: List[Tuple[int, int]] = []
    for score, i, j in ranked:
        if score < threshold:
            break
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j))
    return pairs


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

    merged: List[MergedPrediction] = []
    matched_gen: set = set()
    matched_ext: set = set()

    # Optimal 1-to-1 matching between the two teachers' quads (Hungarian /
    # exact assignment) instead of greedy best-first — see _optimal_assignment.
    if gen_quads and ext_quads:
        score_matrix: List[List[float]] = [[0.0] * len(ext_quads) for _ in gen_quads]
        breakdowns: List[List[dict]] = [[{} for _ in ext_quads] for _ in gen_quads]
        for i, g in enumerate(gen_quads):
            for j, e in enumerate(ext_quads):
                score_matrix[i][j], breakdowns[i][j] = _agreement_breakdown(g, e, weights)

        for i, j in _optimal_assignment(score_matrix, weights.match_threshold):
            g, e = gen_quads[i], ext_quads[j]
            matched_gen.add(i)
            matched_ext.add(j)
            merged.append(
                MergedPrediction(
                    aspect=g.aspect,
                    opinion=g.opinion,
                    category=g.category,
                    sentiment=g.sentiment,
                    conf_g=g.confidence,
                    conf_e=e.confidence,
                    agreement=score_matrix[i][j],
                    sources=["generative", "extractive"],
                    **breakdowns[i][j],
                )
            )

    # Generative-only quads: agreement=0, conf_e=0 — FinalScore from Conf_G alone.
    for i, g in enumerate(gen_quads):
        if i in matched_gen:
            continue
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

    # Extractive-only quads: symmetric treatment, conf_g=0.
    for j, e in enumerate(ext_quads):
        if j in matched_ext:
            continue
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
