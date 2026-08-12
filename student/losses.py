"""Multi-objective student loss (MultiLABSA.docx §4.9).

    L = L_sup + λ_f·L_full + λ_p·L_partial + λ_c·L_cons
          + λ_r·L_relation + λ_a·L_align + λ_b·L_balance

Each term is optional per batch: a routed batch supplies only the components its
route licenses (full route -> generation target; partial route -> the reliable
elements' head losses only; consistency route -> no hard label, only agreement
between two views). ``MultiObjectiveLoss`` sums whatever is present, so the
routing decision (§4.8) directly controls which losses fire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    lambda_full: float = 1.0        # L_full  : generation on full-route pseudo-labels
    lambda_partial: float = 1.0     # L_partial: head losses on reliable elements only
    lambda_cons: float = 0.5        # L_cons  : teacher-student / cross-view consistency
    lambda_relation: float = 1.0    # L_relation: AT<->OT linking
    lambda_align: float = 0.3       # L_align : cross-view span/representation alignment
    lambda_balance: float = 0.1     # L_balance: language/class balance regulariser


def span_loss(aspect_logits, opinion_logits, aspect_labels, opinion_labels) -> torch.Tensor:
    n = aspect_logits.size(-1)
    return F.cross_entropy(aspect_logits.reshape(-1, n), aspect_labels.reshape(-1), ignore_index=-100) + \
        F.cross_entropy(opinion_logits.reshape(-1, n), opinion_labels.reshape(-1), ignore_index=-100)


def implicit_loss(at_logit, ot_logit, at_is_null, ot_is_null) -> torch.Tensor:
    """BCE for the two NULL heads (§4.7)."""
    return F.binary_cross_entropy_with_logits(at_logit, at_is_null.float()) + \
        F.binary_cross_entropy_with_logits(ot_logit, ot_is_null.float())


def balance_loss(category_logits: torch.Tensor) -> torch.Tensor:
    """Encourage a non-degenerate category distribution (negative batch entropy).

    Counters the model collapsing onto the majority category on noisy pseudo-
    labels — the §4.10 "avoid forgetting rare classes/languages" concern.
    """
    p = category_logits.softmax(-1).mean(0)
    entropy = -(p * (p + 1e-9).log()).sum()
    return -entropy  # minimise -> maximise entropy -> flatter usage


class MultiObjectiveLoss:
    def __init__(self, weights: Optional[LossWeights] = None):
        self.w = weights or LossWeights()

    def __call__(self, components: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Combine whatever loss components a routed batch produced.

        Recognised keys (all optional): ``sup``, ``full``, ``partial``,
        ``cons``, ``relation``, ``align``, ``balance``. Returns the total plus
        the per-term breakdown for logging.
        """
        w = self.w
        zero = None
        for v in components.values():
            zero = torch.zeros((), device=v.device)
            break
        if zero is None:
            raise ValueError("empty loss components")

        total = components.get("sup", zero) \
            + w.lambda_full * components.get("full", zero) \
            + w.lambda_partial * components.get("partial", zero) \
            + w.lambda_cons * components.get("cons", zero) \
            + w.lambda_relation * components.get("relation", zero) \
            + w.lambda_align * components.get("align", zero) \
            + w.lambda_balance * components.get("balance", zero)

        breakdown = {k: v.detach() for k, v in components.items()}
        breakdown["total"] = total.detach()
        return {"total": total, "breakdown": breakdown}
