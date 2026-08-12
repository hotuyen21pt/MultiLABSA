"""Pseudo-label audit (MultiLABSA.docx §6.2 "Pseudo-label", Table 5).

Given the pseudo-labels a labelling run kept (each quad carries its ``route`` and
``reliability`` from teacher/routing.py) and the gold for the same reviews,
measures **precision** (are kept pseudo-quads correct?) and **coverage** (what
fraction of gold quads did we recover?), broken down **per element** (AT/AC/OT/SP)
and **per route** (full/partial/verifier/consistency/deferred).

The returned dict is attached to a RunResult as ``metrics["pseudo_label"]`` so
``tables.table5_pseudo_label`` renders it. This closes the §6.2 gap where routing
existed but its precision/coverage was never quantified.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

Quad = Dict[str, Any]


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _elem_keys(q: Quad) -> Dict[str, str]:
    return {"AT": _norm(q.get("aspect", "")), "AC": _norm(q.get("category", "")),
            "OT": _norm(q.get("opinion", "")), "SP": _norm(q.get("sentiment", ""))}


def _prf(tp: int, total_pred: int, total_gold: int) -> Dict[str, float]:
    precision = tp / total_pred if total_pred else 0.0
    coverage = tp / total_gold if total_gold else 0.0
    return {"precision": round(precision, 4), "coverage": round(coverage, 4),
            "matched": tp, "kept": total_pred, "gold": total_gold}


def audit(pseudo: List[dict], gold_by_review: Dict[str, List[Quad]]) -> Dict[str, Any]:
    """pseudo: [{review, quads:[{...,route}]}]; gold_by_review: review -> gold quads."""
    # per-element counters
    el_tp = defaultdict(int)
    el_pred = defaultdict(int)
    el_gold = defaultdict(int)
    # per-route counters (quad-exact precision)
    rt_tp = defaultdict(int)
    rt_pred = defaultdict(int)

    total_gold_quads = 0
    for review, gold in gold_by_review.items():
        total_gold_quads += len(gold)
        for el in ("AT", "AC", "OT", "SP"):
            el_gold[el] += len({_elem_keys(g)[el] for g in gold})

    gold_quad_keys = {r: {(_norm(g["aspect"]), _norm(g["category"]), _norm(g["opinion"]), _norm(g["sentiment"]))
                          for g in gs} for r, gs in gold_by_review.items()}

    for row in pseudo:
        review = (row.get("review") or "").strip()
        gold = gold_by_review.get(review, [])
        gold_elems = {el: {_elem_keys(g)[el] for g in gold} for el in ("AT", "AC", "OT", "SP")}
        gkeys = gold_quad_keys.get(review, set())
        for q in row.get("quads", []):
            # element precision/coverage
            ek = _elem_keys(q)
            for el in ("AT", "AC", "OT", "SP"):
                el_pred[el] += 1
                if ek[el] in gold_elems[el]:
                    el_tp[el] += 1
            # route-level quad-exact precision
            route = q.get("route", "unknown")
            rt_pred[route] += 1
            if (ek["AT"], ek["AC"], ek["OT"], ek["SP"]) in gkeys:
                rt_tp[route] += 1

    by_element = {el: _prf(el_tp[el], el_pred[el], el_gold[el]) for el in ("AT", "AC", "OT", "SP")}
    by_route = {rt: {"precision": round(rt_tp[rt] / rt_pred[rt], 4) if rt_pred[rt] else 0.0,
                     "coverage": round(rt_tp[rt] / total_gold_quads, 4) if total_gold_quads else 0.0,
                     "kept": rt_pred[rt]}
                for rt in rt_pred}
    # Table 5 wants a flat {element_or_route: {precision, coverage}}
    flat = dict(by_element)
    for rt, v in by_route.items():
        flat[f"route:{rt}"] = v
    return flat


def audit_dict_list(pseudo: List[dict], gold: List[List[Quad]], reviews: List[str]) -> Dict[str, Any]:
    """Convenience: build the review->gold map from aligned lists, then audit."""
    return audit(pseudo, {r.strip(): g for r, g in zip(reviews, gold)})
