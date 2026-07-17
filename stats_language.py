"""Count samples per language for the labeled and unlabeled datasets.

Writes two report files to the project root:
  - language_stats.csv  : dataset, file, language, count, percentage (machine-readable)
  - language_stats.txt  : human-readable breakdown
"""

import csv
import json
import os
from collections import Counter

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELED_DIR = os.path.join(BASE_DIR, "data_final", "labeled_data", "hamos26")
UNLABELED_DIR = os.path.join(BASE_DIR, "data_final", "unlabeled_data")

LABELED_FILES = ["train_lang.json", "val_lang.json", "test_lang.json"]
UNLABELED_FILES = ["hotel_review1_lang.csv", "hotel_review2_lang.csv", "hotel_review3_lang.csv"]
CHUNK_SIZE = 100_000

CSV_OUT = os.path.join(BASE_DIR, "language_stats.csv")
TXT_OUT = os.path.join(BASE_DIR, "language_stats.txt")

csv_rows = []   # (dataset, file, language, count, percentage)
txt_lines = []


def record(dataset, file_label, counter):
    total = sum(counter.values())
    txt_lines.append(f"\n{dataset} | {file_label}  (total = {total:,})")
    for lang, n in counter.most_common():
        pct = n / total * 100 if total else 0.0
        lang_disp = lang or "<empty>"
        txt_lines.append(f"  {lang_disp:<8} {n:>10,}  ({pct:5.2f}%)")
        csv_rows.append((dataset, file_label, lang_disp, n, round(pct, 4)))


def count_labeled():
    grand = Counter()
    for name in LABELED_FILES:
        with open(os.path.join(LABELED_DIR, name), "r", encoding="utf-8") as f:
            data = json.load(f)
        c = Counter(row.get("language", "") for row in data)
        record("labeled", name, c)
        grand.update(c)
    record("labeled", "ALL", grand)


def count_unlabeled():
    grand = Counter()
    for name in UNLABELED_FILES:
        c = Counter()
        # Proper CSV parsing (handles embedded newlines/commas); only load the column we need.
        for chunk in pd.read_csv(os.path.join(UNLABELED_DIR, name), usecols=["language"], chunksize=CHUNK_SIZE):
            c.update(chunk["language"].fillna("").astype(str))
        record("unlabeled", name, c)
        grand.update(c)
    record("unlabeled", "ALL", grand)


if __name__ == "__main__":
    count_labeled()
    count_unlabeled()

    report = "\n".join(txt_lines).lstrip("\n")
    print(report)

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "file", "language", "count", "percentage"])
        w.writerows(csv_rows)

    print(f"\nWrote: {os.path.basename(CSV_OUT)}  and  {os.path.basename(TXT_OUT)}")
