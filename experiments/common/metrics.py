"""ASQP evaluation metrics (MultiLABSA.docx §6.2).

All metrics operate on aligned per-example quad lists:

    gold: List[List[quad]]      pred: List[List[quad]]
    quad = {"aspect","category","opinion","sentiment"}   (aspect/opinion may be "NULL")

Reported (mirrors §6.2 so the same function feeds every table):
    * Element F1        : AT-F1, AC-F1, OT-F1, SP-F1
    * Structure F1      : exact Quad-F1, partial/soft-span Quad-F1
    * Implicit          : EA-EO / IA-EO / EA-IO / IA-IO exact Quad-F1
    * Imbalance         : macro-F1 over categories, rare-/worst-category F1
    * Calibration (opt) : ECE from per-quad confidence vs correctness

Multilingual roll-ups (macro-avg, worst-language, LangGap) are computed by
``aggregate_by_language`` over per-language metric dicts.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from utils.text_alignment import token_jaccard

Quad = Dict[str, str]
NULL = "NULL"


# --------------------------------------------------------------------------- #
# Normalisation & keys                                                          #
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _quad_key(q: Quad) -> Tuple[str, str, str, str]:
    return (_norm(q.get("aspect", NULL)), _norm(q.get("category", "")),
            _norm(q.get("opinion", NULL)), _norm(q.get("sentiment", "")))


def _is_implicit(term: str) -> bool:
    return _norm(term) in ("", "null", "none")


def _bucket(q: Quad) -> str:
    a = "IA" if _is_implicit(q.get("aspect", "")) else "EA"
    o = "IO" if _is_implicit(q.get("opinion", "")) else "EO"
    return f"{a}-{o}"


# --------------------------------------------------------------------------- #
# Precision / recall / F1 from counts                                           #
# --------------------------------------------------------------------------- #
def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def _micro_f1(gold: List[List[Any]], pred: List[List[Any]], key_fn) -> Dict[str, float]:
    """Micro P/R/F1 over multiset keys, matched greedily within each example."""
    tp = fp = fn = 0
    for g_list, p_list in zip(gold, pred):
        g_keys = [key_fn(x) for x in g_list]
        p_keys = [key_fn(x) for x in p_list]
        remaining = defaultdict(int)
        for k in g_keys:
            remaining[k] += 1
        for k in p_keys:
            if remaining.get(k, 0) > 0:
                remaining[k] -= 1
                tp += 1
            else:
                fp += 1
        fn += sum(remaining.values())
    return _prf(tp, fp, fn)


# --------------------------------------------------------------------------- #
# Structure F1                                                                  #
# --------------------------------------------------------------------------- #
def exact_quad_f1(gold, pred) -> Dict[str, float]:
    return _micro_f1(gold, pred, _quad_key)


def partial_quad_f1(gold, pred, span_threshold: float = 0.5) -> Dict[str, float]:
    """Soft-span Quad-F1: category+sentiment exact, aspect/opinion token-overlap.

    Rewards near-miss spans ("comfortable" vs "extremely comfortable") that the
    strict exact match rejects — the §6.2 motivation for reporting both.
    """
    tp = fp = fn = 0
    for g_list, p_list in zip(gold, pred):
        used = set()
        for p in p_list:
            hit = False
            for i, g in enumerate(g_list):
                if i in used:
                    continue
                if (_norm(p.get("category", "")) == _norm(g.get("category", ""))
                        and _norm(p.get("sentiment", "")) == _norm(g.get("sentiment", ""))
                        and token_jaccard(p.get("aspect", ""), g.get("aspect", "")) >= span_threshold
                        and token_jaccard(p.get("opinion", ""), g.get("opinion", "")) >= span_threshold):
                    used.add(i)
                    hit = True
                    break
            tp += 1 if hit else 0
            fp += 0 if hit else 1
        fn += len(g_list) - len(used)
    return _prf(tp, fp, fn)


# --------------------------------------------------------------------------- #
# Element F1                                                                     #
# --------------------------------------------------------------------------- #
def element_f1(gold, pred) -> Dict[str, Dict[str, float]]:
    return {
        "AT": _micro_f1(gold, pred, lambda q: _norm(q.get("aspect", NULL))),
        "AC": _micro_f1(gold, pred, lambda q: _norm(q.get("category", ""))),
        "OT": _micro_f1(gold, pred, lambda q: _norm(q.get("opinion", NULL))),
        "SP": _micro_f1(gold, pred, lambda q: _norm(q.get("sentiment", ""))),
    }


# --------------------------------------------------------------------------- #
# Implicit buckets                                                              #
# --------------------------------------------------------------------------- #
def implicit_f1(gold, pred) -> Dict[str, Dict[str, float]]:
    """Exact Quad-F1 restricted to each explicit/implicit bucket."""
    out: Dict[str, Dict[str, float]] = {}
    for bucket in ("EA-EO", "IA-EO", "EA-IO", "IA-IO"):
        g = [[q for q in gl if _bucket(q) == bucket] for gl in gold]
        p = [[q for q in pl if _bucket(q) == bucket] for pl in pred]
        out[bucket] = _micro_f1(g, p, _quad_key)
    return out


# --------------------------------------------------------------------------- #
# Category imbalance (macro / rare / worst)                                      #
# --------------------------------------------------------------------------- #
def category_f1(gold, pred, rare_threshold: int = 50) -> Dict[str, Any]:
    cats = sorted({_norm(q.get("category", "")) for gl in gold for q in gl})
    per_cat: Dict[str, Dict[str, float]] = {}
    support: Dict[str, int] = {}
    for c in cats:
        g = [[q for q in gl if _norm(q.get("category", "")) == c] for gl in gold]
        p = [[q for q in pl if _norm(q.get("category", "")) == c] for pl in pred]
        per_cat[c] = _micro_f1(g, p, _quad_key)
        support[c] = sum(len(x) for x in g)
    f1s = [m["f1"] for m in per_cat.values()]
    rare = [per_cat[c]["f1"] for c in cats if support[c] < rare_threshold]
    return {
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "worst_category_f1": round(min(f1s), 4) if f1s else 0.0,
        "rare_category_f1": round(sum(rare) / len(rare), 4) if rare else None,
        "per_category": per_cat,
    }


# --------------------------------------------------------------------------- #
# Calibration (optional)                                                        #
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    pred, gold, confidences: List[List[float]], n_bins: int = 10
) -> float:
    """ECE: |accuracy - confidence| averaged over confidence bins."""
    gold_keys = [{_quad_key(q) for q in gl} for gl in gold]
    samples: List[Tuple[float, int]] = []
    for i, p_list in enumerate(pred):
        for q, c in zip(p_list, confidences[i]):
            samples.append((float(c), 1 if _quad_key(q) in gold_keys[i] else 0))
    if not samples:
        return 0.0
    total = len(samples)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bucket = [(c, ok) for c, ok in samples if (lo < c <= hi) or (b == 0 and c == 0)]
        if not bucket:
            continue
        conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(ok for _, ok in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(acc - conf)
    return round(ece, 4)


# --------------------------------------------------------------------------- #
# Top-level entry                                                               #
# --------------------------------------------------------------------------- #
def compute_quad_metrics(
    gold: List[List[Quad]],
    pred: List[List[Quad]],
    confidences: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """All §6.2 metrics for one (gold, pred) split. Feeds every comparison table."""
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred length mismatch: {len(gold)} vs {len(pred)}")
    metrics: Dict[str, Any] = {
        "exact_quad": exact_quad_f1(gold, pred),
        "partial_quad": partial_quad_f1(gold, pred),
        "element": element_f1(gold, pred),
        "implicit": implicit_f1(gold, pred),
        "category": category_f1(gold, pred),
        "num_examples": len(gold),
        "num_gold_quads": sum(len(x) for x in gold),
        "num_pred_quads": sum(len(x) for x in pred),
    }
    if confidences is not None:
        metrics["calibration_ece"] = expected_calibration_error(pred, gold, confidences)
    return metrics


# --------------------------------------------------------------------------- #
# Multilingual roll-up                                                          #
# --------------------------------------------------------------------------- #
def aggregate_by_language(per_language_f1: Dict[str, float]) -> Dict[str, float]:
    """Macro-avg, worst-language and LangGap over per-language exact Quad-F1."""
    if not per_language_f1:
        return {"macro_f1": 0.0, "worst_language_f1": 0.0, "lang_gap": 0.0}
    values = list(per_language_f1.values())
    return {
        "macro_f1": round(sum(values) / len(values), 4),
        "worst_language_f1": round(min(values), 4),
        "lang_gap": round(max(values) - min(values), 4),
    }
