"""Element-wise & Relation-aware reliability (MERA-XQUAD §4.5–4.6).

Turns a merged (T_G, T_E) quad into per-element reliability scores plus a single
relation-aware ``r_quad``. Uses only signals already available after fusion
(calibrated teacher confidence + the per-component cross-teacher agreement
breakdown from ``teacher/disagreement.py``) — no extra model or training.

    r_e  = w_conf * C_e  +  w_agree * A_e            (per element AT/AC/OT/SP)
    r_rel = (r_AT–OT + r_AT–AC + r_OT–SP) / 3         (§4.6)
    r_quad = (r_AT · r_AC · r_OT · r_SP · r_rel)^(1/5)

The geometric mean for ``r_quad`` is deliberate: one very weak element or
relation must drag the whole quad's reliability down (a wrong pairing makes the
quad wrong even if every element is individually plausible).
"""

from __future__ import annotations

import math
from typing import Dict

from utils.schema import MergedPrediction


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _geomean(values) -> float:
    vals = [_clip01(v) for v in values]
    if not vals:
        return 0.0
    prod = 1.0
    for v in vals:
        prod *= max(v, 1e-9)  # avoid a single 0 collapsing everything to exactly 0
    return prod ** (1.0 / len(vals))


def calibrated_confidence(m: MergedPrediction) -> float:
    """Mean of the teacher confidences that actually vouched for this quad.

    A single-teacher quad (the other teacher's conf is 0 because it never
    proposed it) is scored on the vouching teacher alone, not penalised by
    averaging in a structural 0.
    """
    confs = [c for c in (m.conf_g, m.conf_e) if c > 0]
    return sum(confs) / len(confs) if confs else 0.0


def element_reliability(
    m: MergedPrediction, w_conf: float = 0.5, w_agree: float = 0.5
) -> Dict[str, float]:
    """Per-element reliability r_AT / r_AC / r_OT / r_SP in [0, 1]."""
    c = calibrated_confidence(m)
    return {
        "r_AT": _clip01(w_conf * c + w_agree * m.aspect_overlap),
        "r_OT": _clip01(w_conf * c + w_agree * m.opinion_overlap),
        "r_AC": _clip01(w_conf * c + w_agree * m.category_match),
        "r_SP": _clip01(w_conf * c + w_agree * m.sentiment_match),
    }


def relation_reliability(elem: Dict[str, float]) -> float:
    """r_rel from the three structural relations (§4.6).

    Each pairwise relation reliability is the geometric mean of its two element
    reliabilities (a relation can be no more reliable than its weaker endpoint).
    """
    r_at_ot = math.sqrt(elem["r_AT"] * elem["r_OT"])
    r_at_ac = math.sqrt(elem["r_AT"] * elem["r_AC"])
    r_ot_sp = math.sqrt(elem["r_OT"] * elem["r_SP"])
    return (r_at_ot + r_at_ac + r_ot_sp) / 3.0


def quad_reliability(m: MergedPrediction) -> Dict[str, float]:
    """Full reliability report: r_AT, r_AC, r_OT, r_SP, r_rel, r_quad."""
    elem = element_reliability(m)
    r_rel = relation_reliability(elem)
    r_quad = _geomean([elem["r_AT"], elem["r_AC"], elem["r_OT"], elem["r_SP"], r_rel])
    return {**elem, "r_rel": r_rel, "r_quad": r_quad}
