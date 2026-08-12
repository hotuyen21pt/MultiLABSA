"""Hotel-disjoint split (MultiLABSA.docx §6.1, "Hotel-disjoint" track).

Guarantees no hotel in the test set appears in train — the realistic-deployment
scenario where the model meets unseen properties. Splits by the ``hotel`` field
(id/name); every review of a hotel goes entirely to one side.

    python -m experiments.data_prep.hotel_disjoint_split \
        --input reviews.json --hotel_field hotel --test_ratio 0.2 --out_dir splits/
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List


def hotel_disjoint(records: List[dict], hotel_field: str, test_ratio: float, seed: int = 42):
    """Partition records so hotels are disjoint between train and test."""
    if not records or hotel_field not in records[0]:
        raise KeyError(
            f"records need a '{hotel_field}' field for a hotel-disjoint split; "
            f"available keys: {sorted(records[0].keys()) if records else '[]'}"
        )
    import random
    by_hotel: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_hotel[str(r[hotel_field])].append(r)

    hotels = sorted(by_hotel)                       # deterministic before shuffle
    random.Random(seed).shuffle(hotels)
    n_test_hotels = max(1, int(round(len(hotels) * test_ratio)))
    test_hotels = set(hotels[:n_test_hotels])

    train = [r for h in hotels if h not in test_hotels for r in by_hotel[h]]
    test = [r for h in test_hotels for r in by_hotel[h]]
    assert not (set(str(r[hotel_field]) for r in train) & test_hotels), "hotel leaked into train"
    return train, test


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hotel-disjoint split")
    p.add_argument("--input", required=True)
    p.add_argument("--hotel_field", default="hotel")
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="splits")
    a = p.parse_args()
    records = json.load(open(a.input, encoding="utf-8"))
    train, test = hotel_disjoint(records, a.hotel_field, a.test_ratio, a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(train, open(os.path.join(a.out_dir, "train.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(test, open(os.path.join(a.out_dir, "test.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[hotel_disjoint] train={len(train)} test={len(test)} (hotels disjoint) -> {a.out_dir}")
