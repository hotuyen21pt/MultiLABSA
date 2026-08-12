"""Implicit AT/OT acceptance rule (MultiLABSA.docx §4.7).

A NULL (implicit) aspect/opinion is only committed as a *hard* label when the
teachers agree, the prediction is stable across views, AND the implicit-head
probability is high — all three. If it's merely plausible, the example is routed
to a verifier / partial supervision instead of fabricating a hard NULL, which
would teach the student a wrong implicit whenever it was actually a missed span.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImplicitThresholds:
    agreement: float = 0.6      # cross-teacher agreement
    stability: float = 0.6      # multi-view stability (fraction of views agreeing / r_quad)
    implicit_prob: float = 0.6  # p(element = NULL) from the implicit head
    medium: float = 0.4         # below the bars but not clearly explicit -> verify


def decide_null(
    agreement: float, stability: float, implicit_prob: float,
    thr: ImplicitThresholds = None,
) -> str:
    """Return one of: 'accept_null', 'verifier', 'reject_null'."""
    thr = thr or ImplicitThresholds()
    if agreement >= thr.agreement and stability >= thr.stability and implicit_prob >= thr.implicit_prob:
        return "accept_null"
    if implicit_prob >= thr.medium:      # plausibly implicit but not certain
        return "verifier"
    return "reject_null"                  # likely a missed explicit span, not implicit
