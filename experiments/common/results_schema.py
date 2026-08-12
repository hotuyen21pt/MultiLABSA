"""Standardised result record — the contract that makes comparison tables work.

EVERY case (baseline, MERA-XQUAD, ablation) writes one ``results/<case_id>.json``
with this exact shape. ``experiments/common/tables.py`` reads the whole folder
and pivots into Table 1–8 (§6.3). Because the schema is fixed, adding a new case
never touches the table code — you just drop another json in.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# The 7 evaluation tracks of §6.1 (a case must name one so tables can group it).
TRACKS = [
    "english_supervised",
    "zero_shot",
    "dapt_zeroshot",
    "semi_supervised",
    "few_shot",
    "hotel_disjoint",
    "native_vs_translated",
]


@dataclass
class RunResult:
    """One evaluated run of one method, on one language split, at one seed."""

    case_id: str                      # unique, e.g. "mera_full__vi__seed42"
    method: str                       # "MERA-XQUAD", "M1_zeroshot", "paraphrase_mt5", ...
    track: str                        # one of TRACKS
    language: str                     # "en" / "vi" / ... / "all"
    seed: int
    metrics: Dict[str, Any]           # output of compute_quad_metrics (+ pseudo-label / efficiency)
    config: Dict[str, Any] = field(default_factory=dict)   # full run config (reproducibility, §6.4)
    ablation: Optional[str] = None    # ablation tag (§7.1), None for the full model
    predictions_path: Optional[str] = None                 # optional pointer to raw preds

    def __post_init__(self) -> None:
        if self.track not in TRACKS:
            raise ValueError(f"track must be one of {TRACKS}, got {self.track!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def save_result(result: RunResult, results_dir: str = "results") -> str:
    """Write ``results/<case_id>.json`` and return its path."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{result.case_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def load_results(results_dir: str = "results") -> List[RunResult]:
    """Load every ``*.json`` under ``results_dir`` into RunResult objects."""
    out: List[RunResult] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        out.append(RunResult(**data))
    return out
