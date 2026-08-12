"""Few-shot target-label builder (MultiLABSA.docx §6.1 "Few-shot", baseline M6).

Samples k target-language gold labels (10/25/50/100 per §3-M6) to add to the
English train set, for the label-efficiency learning curve. Sampling is seeded
and stratified per language so each language contributes its share.

    python -m experiments.data_prep.few_shot_builder \
        --input target_gold.json --k 25 --out fewshot_k25.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List


def build_few_shot(records: List[dict], k: int, lang_field: str = "language", seed: int = 42) -> List[dict]:
    """Return up to k labels *per language* (stratified), deterministically sampled."""
    import random
    rng = random.Random(seed)
    by_lang: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_lang[r.get(lang_field, "unk")].append(r)
    out: List[dict] = []
    for lang, rows in sorted(by_lang.items()):
        rng.shuffle(rows)
        out.extend(rows[:k])
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Few-shot builder (M6)")
    p.add_argument("--input", required=True)
    p.add_argument("--k", type=int, default=25, help="labels per language (10/25/50/100)")
    p.add_argument("--lang_field", default="language")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    records = json.load(open(a.input, encoding="utf-8"))
    fs = build_few_shot(records, a.k, a.lang_field, a.seed)
    json.dump(fs, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[few_shot] {len(fs)} labels (k={a.k}/lang) -> {a.out}")
