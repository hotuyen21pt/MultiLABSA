"""Error analysis (MultiLABSA.docx §7.2).

Categorises every prediction error against the gold for a split, into the exact
buckets §7.2 lists, so the paper can report *where* a system fails rather than
only an aggregate F1:

    boundary          - AT/OT overlaps gold but span isn't exact (too short/long)
    category_confusion- AT+OT correct, category wrong
    sentiment_error   - AT+OT+category correct, sentiment wrong
    relation_error    - AT and OT each exist in gold but are paired wrongly
    implicit_error    - NULL predicted where explicit exists (or vice-versa)
    cross_lingual     - any error on a non-English review (tokenisation/morphology)
    spurious          - a predicted quad with no gold counterpart at all
    missed            - a gold quad with no predicted counterpart at all
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from utils.text_alignment import token_jaccard

Quad = Dict[str, str]


def _n(s: str) -> str:
    return (s or "").strip().lower()


def _implicit(t: str) -> bool:
    return _n(t) in ("", "null", "none")


def _exact(a: Quad, b: Quad) -> bool:
    return (_n(a["aspect"]), _n(a["category"]), _n(a["opinion"]), _n(a["sentiment"])) == \
           (_n(b["aspect"]), _n(b["category"]), _n(b["opinion"]), _n(b["sentiment"]))


def _classify(p: Quad, gold: List[Quad]) -> str:
    if any(_exact(p, g) for g in gold):
        return "correct"
    # AT+OT match (soft) -> narrow down what's wrong
    for g in gold:
        at_ok = token_jaccard(p["aspect"], g["aspect"]) >= 0.5 or _implicit(p["aspect"]) == _implicit(g["aspect"])
        ot_ok = token_jaccard(p["opinion"], g["opinion"]) >= 0.5
        if at_ok and ot_ok:
            if _n(p["category"]) != _n(g["category"]):
                return "category_confusion"
            if _n(p["sentiment"]) != _n(g["sentiment"]):
                return "sentiment_error"
            # spans overlap but not exact
            if _n(p["aspect"]) != _n(g["aspect"]) or _n(p["opinion"]) != _n(g["opinion"]):
                return "boundary"
    # implicit mismatch
    if any((_implicit(p["aspect"]) != _implicit(g["aspect"]) or _implicit(p["opinion"]) != _implicit(g["opinion"]))
           for g in gold):
        return "implicit_error"
    # AT and OT each exist in gold but not together -> mis-pairing
    at_in = any(_n(p["aspect"]) == _n(g["aspect"]) for g in gold)
    ot_in = any(_n(p["opinion"]) == _n(g["opinion"]) for g in gold)
    if at_in and ot_in:
        return "relation_error"
    return "spurious"


def analyze(gold: List[List[Quad]], pred: List[List[Quad]], langs: List[str] = None) -> Dict[str, Any]:
    langs = langs or ["en"] * len(gold)
    counts: Counter = Counter()
    cross_lingual = 0
    for gl, pl, lang in zip(gold, pred, langs):
        for p in pl:
            tag = _classify(p, gl)
            counts[tag] += 1
            if tag not in ("correct",) and lang != "en":
                cross_lingual += 1
        # missed gold
        for g in gl:
            if not any(_exact(g, p) for p in pl):
                counts["missed"] += 1
    counts["cross_lingual"] = cross_lingual
    total_err = sum(v for k, v in counts.items() if k not in ("correct", "cross_lingual"))
    return {
        "counts": dict(counts),
        "total_errors": total_err,
        "error_rate": round(total_err / max(1, total_err + counts.get("correct", 0)), 4),
    }
