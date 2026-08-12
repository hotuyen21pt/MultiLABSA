"""Ablation matrix (MultiLABSA.docx §7.1).

Defines the 10 mandated ablations as student-config deltas from the full model.
Each entry becomes one Table 7 row: run the student with the delta, evaluate,
and the ``ablation`` tag flows into the RunResult so ``tables.table7_ablation``
groups it automatically.

This module only *declares* the matrix (so it is importable without torch); the
Kaggle notebook / a driver script applies each delta to ``StudentConfig`` and
runs ``student.train_student.train`` then ``evaluation.run_eval``.
"""

from __future__ import annotations

from typing import Any, Dict, List

# tag -> (human hypothesis tested, StudentConfig overrides)
ABLATIONS: Dict[str, Dict[str, Any]] = {
    "full": {"_hypothesis": "full MERA-XQUAD", },
    "no_partial": {"_hypothesis": "is the correct part of a pseudo-quad useful?",
                   "lambda_partial": 0.0},
    "no_relation": {"_hypothesis": "does relation scoring cut mis-pairings?",
                    "lambda_relation": 0.0},
    "single_teacher": {"_hypothesis": "is architectural agreement useful?",
                       "single_teacher": True},
    "no_multiview": {"_hypothesis": "what does view stability contribute?",
                     "multiview": False},
    "no_dapt": {"_hypothesis": "does the gain come only from domain adaptation?",
                "student_backbone": "google/mt5-base"},
    "fixed_threshold": {"_hypothesis": "are rare classes/languages protected?",
                        "fixed_threshold": True},
    "no_consistency": {"_hypothesis": "is low-confidence stable data useful?",
                       "lambda_cons": 0.0},
    "no_deferred": {"_hypothesis": "does using every sample amplify confirmation bias?",
                    "use_all_routes": True},
    "no_implicit": {"_hypothesis": "what does implicit modelling contribute?",
                    "no_implicit": True},
}


def ablation_configs(base: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge each ablation delta onto a base config dict -> list of run configs."""
    runs = []
    for tag, delta in ABLATIONS.items():
        cfg = dict(base)
        cfg.update({k: v for k, v in delta.items() if not k.startswith("_")})
        cfg["ablation"] = tag
        runs.append(cfg)
    return runs


if __name__ == "__main__":
    for tag, d in ABLATIONS.items():
        print(f"{tag:16s} {d['_hypothesis']}")
