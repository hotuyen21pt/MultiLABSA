"""Data audit — Phase P0 (MultiLABSA.docx §5.1) and Table 1 material.

Reports, per split: record/quad counts, duplicate reviews, category & sentiment
distribution, implicit ratio, language distribution, and — crucially — any
train/test review LEAKAGE (a review appearing in more than one split), which
would silently inflate every downstream number.

    python -m experiments.data_prep.audit --labeled_dir data_final/labeled_data/hamos26
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Dict, List

from experiments.common.data import load_gold_split

SPLITS = ["train", "val", "test"]


def _split_stats(reviews: List[str], quads, langs) -> Dict[str, Any]:
    n_quads = sum(len(q) for q in quads)
    cats = Counter(qi["category"].upper() for q in quads for qi in q)
    sents = Counter(qi["sentiment"].lower() for q in quads for qi in q)
    implicit_at = sum(1 for q in quads for qi in q if (qi["aspect"] or "NULL").upper() == "NULL")
    implicit_ot = sum(1 for q in quads for qi in q if (qi["opinion"] or "NULL").upper() == "NULL")
    dup = len(reviews) - len(set(r.strip() for r in reviews))
    return {
        "records": len(reviews),
        "quads": n_quads,
        "duplicate_reviews": dup,
        "avg_quads_per_review": round(n_quads / max(1, len(reviews)), 3),
        "implicit_ratio": {"AT": round(implicit_at / max(1, n_quads), 4),
                           "OT": round(implicit_ot / max(1, n_quads), 4)},
        "category_dist": dict(cats.most_common()),
        "sentiment_dist": dict(sents.most_common()),
        "language_dist": dict(Counter(langs).most_common()),
    }


def audit(labeled_dir: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    review_sets: Dict[str, set] = {}
    for split in SPLITS:
        try:
            reviews, quads, langs = load_gold_split(labeled_dir, split)
        except FileNotFoundError:
            continue
        report[split] = _split_stats(reviews, quads, langs)
        review_sets[split] = {r.strip() for r in reviews}

    # cross-split leakage
    leakage = {}
    splits_present = list(review_sets)
    for i in range(len(splits_present)):
        for j in range(i + 1, len(splits_present)):
            a, b = splits_present[i], splits_present[j]
            overlap = review_sets[a] & review_sets[b]
            if overlap:
                leakage[f"{a}∩{b}"] = len(overlap)
    report["leakage"] = leakage or "none"
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Gold data audit (P0)")
    p.add_argument("--labeled_dir", default="data_final/labeled_data/hamos26")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    rep = audit(a.labeled_dir)
    text = json.dumps(rep, ensure_ascii=False, indent=2)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
        print(f"[audit] wrote {a.out}")
    else:
        print(text)
