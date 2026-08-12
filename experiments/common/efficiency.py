"""Efficiency accounting (MultiLABSA.docx §6.2 "Efficiency", Table 8).

A tiny tracker for the cost columns the paper must report — training time,
inference time, GPU hours, LLM calls and token cost — so Table 8 is filled from
real measurements rather than left blank.

    eff = EfficiencyTracker()
    with eff.timer("train"):     ...            # -> train_hours
    with eff.timer("inference"): ...            # -> inference_seconds
    eff.add_llm_call(tokens=1234)               # verifier route usage
    metrics["efficiency"] = eff.summary(gpu_count=1)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict


class EfficiencyTracker:
    def __init__(self) -> None:
        self.seconds: Dict[str, float] = {}
        self.llm_calls = 0
        self.token_cost = 0

    @contextmanager
    def timer(self, name: str):
        start = time.time()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.time() - start)

    def add_llm_call(self, tokens: int = 0) -> None:
        self.llm_calls += 1
        self.token_cost += int(tokens)

    def summary(self, gpu_count: int = 1) -> Dict[str, float]:
        train_s = self.seconds.get("train", 0.0)
        infer_s = self.seconds.get("inference", 0.0)
        return {
            "train_hours": round(train_s / 3600.0, 4),
            "inference_seconds": round(infer_s, 2),
            "gpu_hours": round(gpu_count * train_s / 3600.0, 4),
            "llm_calls": self.llm_calls,
            "token_cost": self.token_cost,
        }
