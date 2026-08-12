"""Score one case's predictions into a standardised RunResult (MultiLABSA.docx §6).

Input: a predictions json (list of {"review","language","quads":[...]}) — exactly
what ``infer.py`` / a baseline / the student emit — plus the gold split.
Output: ``results/<case_id>.json`` (per-language + overall metrics), the row
material every comparison table reads.

Usage:
    python -m evaluation.run_eval \
        --predictions preds/mera_full.json --labeled_dir data_final/labeled_data/hamos26 \
        --split test --method MERA-XQUAD --track semi_supervised --seed 42
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from experiments.common.data import group_by_language, load_gold_split
from experiments.common.metrics import aggregate_by_language, compute_quad_metrics
from experiments.common.results_schema import RunResult, save_result


def _pred_by_review(predictions: List[dict]) -> Dict[str, List[dict]]:
    return {(p.get("review") or "").strip(): p.get("quads", []) for p in predictions}


def align(gold_reviews, gold_quads, predictions):
    """Align predictions to gold by review text (empty quad list if missing)."""
    lut = _pred_by_review(predictions)
    return [lut.get(r.strip(), []) for r in gold_reviews]


def evaluate(predictions_path, labeled_dir, split, method, track, seed, ablation=None, results_dir="results"):
    reviews, gold_quads, langs = load_gold_split(labeled_dir, split)
    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)
    pred_quads = align(reviews, gold_quads, predictions)

    metrics = compute_quad_metrics(gold_quads, pred_quads)

    # per-language exact Quad-F1 -> multilingual roll-up (macro / worst / LangGap)
    per_lang_f1: Dict[str, float] = {}
    per_lang_metrics: Dict[str, dict] = {}
    for lang, (g, p) in group_by_language(gold_quads, pred_quads, langs).items():
        lm = compute_quad_metrics(g, p)
        per_lang_f1[lang] = lm["exact_quad"]["f1"]
        per_lang_metrics[lang] = lm
    metrics["multilingual"] = aggregate_by_language(per_lang_f1)
    metrics["per_language"] = per_lang_metrics

    result = RunResult(
        case_id=f"{method}__{split}__seed{seed}" + (f"__{ablation}" if ablation else ""),
        method=method, track=track, language="all", seed=seed, ablation=ablation,
        metrics=metrics, predictions_path=predictions_path,
    )
    path = save_result(result, results_dir)
    print(f"[run_eval] exact Quad-F1={metrics['exact_quad']['f1']:.4f} "
          f"| macro={metrics['multilingual']['macro_f1']:.4f} "
          f"| worst-lang={metrics['multilingual']['worst_language_f1']:.4f} -> {path}")
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate predictions -> RunResult")
    p.add_argument("--predictions", required=True)
    p.add_argument("--labeled_dir", default="data_final/labeled_data/hamos26")
    p.add_argument("--split", default="test")
    p.add_argument("--method", required=True)
    p.add_argument("--track", default="semi_supervised")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablation", default=None)
    p.add_argument("--results_dir", default="results")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    evaluate(a.predictions, a.labeled_dir, a.split, a.method, a.track, a.seed, a.ablation, a.results_dir)
