"""Statistical significance testing (MultiLABSA.docx §7.3).

Paired bootstrap over test predictions to produce a 95% confidence interval on
the metric difference between two systems, so we never claim superiority on a
small gap whose CI straddles zero (§7.3).
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List

Quad = Dict[str, str]


def paired_bootstrap(
    gold: List[List[Quad]],
    pred_a: List[List[Quad]],
    pred_b: List[List[Quad]],
    metric_fn: Callable[[List[List[Quad]], List[List[Quad]]], Dict[str, float]],
    metric_key: str = "f1",
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap the (B - A) metric difference; return mean, 95% CI, p-value.

    metric_fn must return a dict containing ``metric_key`` (e.g. exact_quad_f1).
    """
    rng = random.Random(seed)
    n = len(gold)
    idx = list(range(n))
    diffs: List[float] = []
    for _ in range(n_resamples):
        sample = [rng.choice(idx) for _ in range(n)]
        g = [gold[i] for i in sample]
        a = metric_fn(g, [pred_a[i] for i in sample])[metric_key]
        b = metric_fn(g, [pred_b[i] for i in sample])[metric_key]
        diffs.append(b - a)
    diffs.sort()
    lo = diffs[int(0.025 * n_resamples)]
    hi = diffs[int(0.975 * n_resamples)]
    mean = sum(diffs) / len(diffs)
    # two-sided p-value: fraction of resamples on the opposite side of 0
    p = 2 * min(
        sum(1 for d in diffs if d <= 0) / n_resamples,
        sum(1 for d in diffs if d >= 0) / n_resamples,
    )
    return {
        "mean_diff": round(mean, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "p_value": round(min(p, 1.0), 4),
        "significant": lo > 0 or hi < 0,   # CI excludes 0
    }
