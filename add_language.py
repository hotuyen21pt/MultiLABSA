"""Add `language` + `language_score` to labeled_data (JSON) and unlabeled_data
(CSV) using fastText language identification (lid.176.bin).

Originals are left untouched; results are written to new *_lang.json / *_lang.csv
files next to them.

Model download (place lid.176.bin next to this script):
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

Usage:
    python add_language.py
"""

import json
import os

import fasttext
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lid.176.bin")

LABELED_DIR = os.path.join(BASE_DIR, "data_final", "labeled_data", "hamos26")
LABELED_FILES = ["train.json", "val.json", "test.json"]

UNLABELED_DIR = os.path.join(BASE_DIR, "data_final", "unlabeled_data")
UNLABELED_FILES = ["hotel_review1.csv", "hotel_review2.csv", "hotel_review3.csv"]

TEXT_FIELD = "review"          # column/field holding the text
LANG_FIELD = "language"        # ISO code, e.g. 'vi', 'en'
SCORE_FIELD = "language_score" # fastText confidence
CHUNK_SIZE = 50_000            # rows per chunk for the large CSVs

model = fasttext.load_model(MODEL_PATH)


def out_path(path: str) -> str:
    """foo.json -> foo_lang.json, foo.csv -> foo_lang.csv"""
    root, ext = os.path.splitext(path)
    return f"{root}_lang{ext}"


def detect_language(text):
    """Return (iso_code, confidence). fastText fails on newlines, so the text
    is flattened to a single line first."""
    if not isinstance(text, str) or not text.strip():
        return "", 0.0
    clean = text.replace("\n", " ").replace("\r", " ").strip()
    label, score = model.predict(clean)
    return label[0].replace("__label__", ""), float(score[0])


def process_labeled():
    for name in LABELED_FILES:
        path = os.path.join(LABELED_DIR, name)
        if not os.path.exists(path):
            print(f"  skip (missing): {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            lang, score = detect_language(row.get(TEXT_FIELD, ""))
            row[LANG_FIELD] = lang
            row[SCORE_FIELD] = score
        dst = out_path(path)
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  {name}: {len(data)} rows -> {os.path.basename(dst)}")


def process_unlabeled():
    for name in UNLABELED_FILES:
        path = os.path.join(UNLABELED_DIR, name)
        if not os.path.exists(path):
            print(f"  skip (missing): {path}")
            continue
        dst = out_path(path)
        total = 0
        # Chunked read/write so the 150-300 MB files stay memory-friendly.
        with pd.read_csv(path, chunksize=CHUNK_SIZE) as reader:
            for i, chunk in enumerate(reader):
                detected = chunk[TEXT_FIELD].apply(detect_language)
                chunk[LANG_FIELD] = [d[0] for d in detected]
                chunk[SCORE_FIELD] = [d[1] for d in detected]
                chunk.to_csv(
                    dst,
                    mode="w" if i == 0 else "a",
                    header=(i == 0),
                    index=False,
                )
                total += len(chunk)
                print(f"    {name}: {total} rows processed", end="\r")
        print(f"\n  {name}: {total} rows -> {os.path.basename(dst)}")


if __name__ == "__main__":
    print("Labeled data (JSON):")
    process_labeled()
    print("Unlabeled data (CSV):")
    process_unlabeled()
    print("Done.")
