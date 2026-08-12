"""Self-training loop: EMA teacher + dynamic curriculum (MultiLABSA.docx §4.10, §5.3).

Each round (§5.3 steps 3-9):
    1. dual teachers (+multi-view) label the unlabeled multilingual pool;
    2. QUAD matching -> reliability -> routing (teacher/ modules already do this);
    3. the student trains on routed pseudo-labels with the multi-objective loss;
    4. the EMA teacher is updated:  θ_T <- μ·θ_T + (1-μ)·θ_S ;
    5. the language curriculum re-scores languages and admits reliable ones first.

This module owns the EMA update and the curriculum scheduler; the actual
labelling reuses ``teacher/`` and the routing tags from ``teacher/routing.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


# --------------------------------------------------------------------------- #
# EMA teacher (§4.10)                                                            #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ema_update(student: torch.nn.Module, teacher: torch.nn.Module, mu: float = 0.997) -> None:
    """θ_T ← μ·θ_T + (1-μ)·θ_S over all parameters and buffers."""
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(mu).add_(s.data, alpha=1.0 - mu)
    for t, s in zip(teacher.buffers(), student.buffers()):
        t.data.copy_(s.data)


# --------------------------------------------------------------------------- #
# Dynamic language curriculum (§4.10)                                            #
# --------------------------------------------------------------------------- #
@dataclass
class LanguageScore:
    confidence: float = 0.0     # mean pseudo-label confidence
    agreement: float = 0.0      # mean cross-teacher agreement
    stability: float = 0.0      # mean multi-view stability
    coverage: float = 0.0       # fraction of reviews that yielded a kept quad
    entropy: float = 0.0        # label-distribution entropy (lower = more peaked)

    def readiness(self) -> float:
        """A language is 'ready' when its pseudo-labels are confident, agreed,
        stable and cover enough reviews. Entropy is subtracted (over-peaked =
        possibly degenerate)."""
        return 0.35 * self.confidence + 0.30 * self.agreement + 0.20 * self.stability \
            + 0.15 * self.coverage - 0.10 * self.entropy


@dataclass
class Curriculum:
    """Admits languages into self-training in readiness order, but keeps a
    balanced sampler so already-admitted low-resource languages are not
    forgotten (§4.10)."""

    warmup_languages: List[str] = field(default_factory=lambda: ["en"])
    admit_per_round: int = 2
    admitted: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.admitted:
            self.admitted = list(self.warmup_languages)

    def update(self, scores: Dict[str, LanguageScore]) -> List[str]:
        """Admit the next most-ready not-yet-admitted languages; return the new
        active set."""
        candidates = sorted(
            (l for l in scores if l not in self.admitted),
            key=lambda l: scores[l].readiness(),
            reverse=True,
        )
        for lang in candidates[: self.admit_per_round]:
            self.admitted.append(lang)
        return list(self.admitted)

    def sampling_weights(self, counts: Dict[str, int], temperature: float = 2.0) -> Dict[str, float]:
        """Temperature-balanced weights over admitted languages (up-weight rare)."""
        active = {l: counts.get(l, 0) for l in self.admitted}
        adj = {l: (n ** (1.0 / temperature)) for l, n in active.items() if n > 0}
        z = sum(adj.values()) or 1.0
        return {l: w / z for l, w in adj.items()}


# --------------------------------------------------------------------------- #
# Adaptive threshold (§4.10 / §7.1 "fixed vs adaptive threshold")               #
# --------------------------------------------------------------------------- #
def adjust_threshold(current: float, kept_ratio: float, target_ratio: float = 0.3,
                     step: float = 0.02, lo: float = 0.5, hi: float = 0.95) -> float:
    """Nudge the acceptance threshold toward a target keep-ratio each round.

    Too many quads kept -> raise the bar; too few -> lower it. Bounded in
    [lo, hi] so it never collapses to accepting/ rejecting everything.
    """
    if kept_ratio > target_ratio:
        current += step
    elif kept_ratio < target_ratio:
        current -= step
    return max(lo, min(hi, current))
