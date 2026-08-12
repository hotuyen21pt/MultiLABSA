"""Confidence–stability routing (MERA-XQUAD §4.8).

Assigns each merged quad to one of five routes describing how a self-training
loop should use it. Uses the reliability report from ``teacher/reliability.py``
plus (optionally) a multi-view stability signal.

    full          - every element & the relation clear the HIGH bar -> full supervision
    partial       - some (not all) elements are reliable            -> loss only on those
    verifier      - medium reliability OR the two teachers disagree  -> send to an LLM/cross-encoder
    consistency   - low reliability BUT stable across views          -> teacher-student consistency only
    deferred      - low and unstable                                 -> skip this round

``view_stability`` (fraction of views that agreed, in [0, 1]) is only available
when multi-view prediction is on; when it's ``None`` the "consistency" route is
simply not reachable and such quads fall through to "deferred", which is the
correct conservative behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from utils.schema import MergedPrediction

ROUTES = ["full", "partial", "verifier", "consistency", "deferred"]


@dataclass
class RoutingThresholds:
    high: float = 0.75           # "reliable" cutoff for an element / relation
    medium: float = 0.50         # lower bound for the verifier band
    stability_high: float = 0.75  # min view-agreement for the consistency route
    require_both_teachers_for_full: bool = True


def route_quad(
    m: MergedPrediction,
    reliability: Dict[str, float],
    thresholds: Optional[RoutingThresholds] = None,
    view_stability: Optional[float] = None,
) -> str:
    """Return one of :data:`ROUTES` for a single merged quad."""
    thr = thresholds or RoutingThresholds()
    elems = [reliability["r_AT"], reliability["r_AC"], reliability["r_OT"], reliability["r_SP"]]
    r_rel = reliability["r_rel"]
    r_quad = reliability["r_quad"]
    both_teachers = m.conf_g > 0 and m.conf_e > 0

    # Full: every element AND the relation clear the high bar (and, by default,
    # both teachers vouched for it).
    if (
        min(elems) >= thr.high
        and r_rel >= thr.high
        and (both_teachers or not thr.require_both_teachers_for_full)
    ):
        return "full"

    # Partial: at least one element is solid but not all of them are.
    if max(elems) >= thr.high and min(elems) < thr.high:
        return "partial"

    # Verifier: medium overall reliability, or the two teachers disagree
    # (single-teacher / uncorroborated quad) yet it isn't outright weak.
    if thr.medium <= r_quad < thr.high or (not both_teachers and r_quad >= thr.medium):
        return "verifier"

    # Consistency-only: weak reliability but stable across views (needs multi-view).
    if view_stability is not None and view_stability >= thr.stability_high:
        return "consistency"

    # Deferred: weak and unstable — don't use this round.
    return "deferred"
