"""Merge hotel_review1_lang.csv, hotel_review2_lang.csv, hotel_review3_lang.csv
into a single CSV.

The stray "Unnamed: 0" index column from the source files is dropped and a
fresh index is written for the merged file. Files are streamed in chunks so
memory usage stays low regardless of total size.

Usage:
    python merge_hotel_reviews.py
"""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNLABELED_DIR = os.path.join(BASE_DIR, "data_final", "unlabeled_data")

SOURCE_FILES = [
    "hotel_review1_lang.csv",
    "hotel_review2_lang.csv",
    "hotel_review3_lang.csv",
]
DEST_FILE = "hotel_review_merged.csv"

CHUNK_SIZE = 100_000


def main():
    dst = os.path.join(UNLABELED_DIR, DEST_FILE)
    row_offset = 0
    wrote_header = False

    for name in SOURCE_FILES:
        path = os.path.join(UNLABELED_DIR, name)
        if not os.path.exists(path):
            print(f"  skip (missing): {path}")
            continue

        file_rows = 0
        with pd.read_csv(path, chunksize=CHUNK_SIZE) as reader:
            for chunk in reader:
                chunk = chunk.drop(columns=[c for c in chunk.columns if c.startswith("Unnamed")])
                chunk.index = range(row_offset, row_offset + len(chunk))
                chunk.to_csv(
                    dst,
                    mode="w" if not wrote_header else "a",
                    header=not wrote_header,
                    index=True,
                )
                wrote_header = True
                row_offset += len(chunk)
                file_rows += len(chunk)
                print(f"    {name}: {file_rows} rows processed", end="\r")
        print(f"\n  {name}: {file_rows} rows merged")

    print(f"Done. Total rows: {row_offset} -> {os.path.basename(dst)}")


if __name__ == "__main__":
    main()
