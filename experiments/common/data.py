"""Evaluation-facing data loaders (no torch dependency).

Reads the gold ASQP json (``data_final/labeled_data/*/{split}.json``) into
plain quad lists for the metrics module, and exposes the multilingual splits
used by the §6.1 tracks. Training-time ``Dataset`` objects live with each model
(``utils/asqp_data.py`` for mT5); this file is only for reading gold/predictions
so it stays importable in a metrics-only environment.

Gold record shape (as in utils/asqp_data.py):
    {"review": str, "language": str,
     "extraction": [{"aspect_term","opinion_term","aspect_category","sentiment"}...]}
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Tuple

Quad = Dict[str, str]
NULL = "NULL"


def _record_to_quads(record: dict) -> List[Quad]:
    quads: List[Quad] = []
    for ext in record.get("extraction", []):
        quads.append({
            "aspect": (ext.get("aspect_term") or NULL).strip() or NULL,
            "opinion": (ext.get("opinion_term") or NULL).strip() or NULL,
            "category": (ext.get("aspect_category") or ext.get("Category") or "").strip().upper(),
            "sentiment": (ext.get("sentiment") or ext.get("Polarity") or "").strip().lower(),
        })
    return quads


def load_gold(path: str) -> Tuple[List[str], List[List[Quad]], List[str]]:
    """Load one gold json file -> (reviews, quad_lists, languages)."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    reviews = [(r.get("review") or "").strip() for r in records]
    quads = [_record_to_quads(r) for r in records]
    langs = [(r.get("language") or "en") for r in records]
    return reviews, quads, langs


def load_gold_split(labeled_dir: str, split: str) -> Tuple[List[str], List[List[Quad]], List[str]]:
    """Load a named split, single-file or ``*/{split}.json`` glob (matches load_asqp_split)."""
    direct = os.path.join(labeled_dir, f"{split}.json")
    if os.path.isfile(direct):
        return load_gold(direct)
    reviews: List[str] = []
    quads: List[List[Quad]] = []
    langs: List[str] = []
    for p in sorted(glob.glob(os.path.join(labeled_dir, "*", f"{split}.json"))):
        r, q, l = load_gold(p)
        reviews += r
        quads += q
        langs += l
    if not reviews:
        raise FileNotFoundError(f"No '{split}.json' under {labeled_dir}")
    return reviews, quads, langs


def group_by_language(
    quads_gold: List[List[Quad]], quads_pred: List[List[Quad]], langs: List[str]
) -> Dict[str, Tuple[List[List[Quad]], List[List[Quad]]]]:
    """Split aligned (gold, pred) into per-language buckets for LangGap / worst-lang."""
    buckets: Dict[str, Tuple[list, list]] = {}
    for g, p, lang in zip(quads_gold, quads_pred, langs):
        buckets.setdefault(lang, ([], []))
        buckets[lang][0].append(g)
        buckets[lang][1].append(p)
    return buckets
