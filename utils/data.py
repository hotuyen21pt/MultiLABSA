"""Loads ``data_final/labeled_data/*/{train,val,test}.json`` (the ASQP-style
quad annotations) into the structures the Extractive Teacher's three heads
are trained against, and collates them into padded batches.

Gold schema per review (see ``data_final/labeled_data/hamos26/train.json``)::

    {
      "review": "The pool is beautiful, and a good size, and very very clean",
      "extraction": [
        {"aspect_term": "pool", "aspect_category": "AMENITY",
         "opinion_term": "beautiful", "sentiment": "positive",
         "Aspect_span": [1, 2], "Opinion_span": [3, 4]},
        ...
      ]
    }

Multiple ``extraction`` rows commonly repeat the same aspect span paired with
different opinion spans (one aspect, several opinions) — :func:`build_example`
de-duplicates spans into two flat lists (``aspect_spans``, ``opinion_spans``)
and expresses each gold quad as a ``(aspect_idx, opinion_idx, category, sentiment)``
tuple that indexes into them. This is exactly the structure the Relation Head
and Classification Head are trained on.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from utils.label_maps import (
    BIO2ID,
    CATEGORY2ID,
    SENTIMENT2ID,
    normalize_category,
    normalize_sentiment,
)
from utils.text_alignment import (
    spans_to_bio,
    word_span_to_token_indices,
    word_tags_to_subword_labels,
    word_tokenize,
)


# --------------------------------------------------------------------------- #
# Raw JSON -> structured example                                                #
# --------------------------------------------------------------------------- #
@dataclass
class ExtractiveExample:
    words: List[str]
    aspect_spans: List[Tuple[int, int]]     # unique word-index spans
    opinion_spans: List[Tuple[int, int]]    # unique word-index spans
    pairs: List[Tuple[int, int, int, int]]  # (aspect_idx, opinion_idx, category_id, sentiment_id)


def _dedup_span(spans: List[Tuple[int, int]], span: Tuple[int, int]) -> int:
    """Return the index of ``span`` in ``spans``, appending it if new."""
    span = (int(span[0]), int(span[1]))
    for i, s in enumerate(spans):
        if s == span:
            return i
    spans.append(span)
    return len(spans) - 1


def build_example(record: dict) -> Optional[ExtractiveExample]:
    """Convert one raw JSON review record into an :class:`ExtractiveExample`."""
    text = record.get("review", "")
    words = word_tokenize(text)
    if not words:
        return None

    aspect_spans: List[Tuple[int, int]] = []
    opinion_spans: List[Tuple[int, int]] = []
    pairs: List[Tuple[int, int, int, int]] = []

    for ext in record.get("extraction", []):
        a_span = ext.get("Aspect_span")
        o_span = ext.get("Opinion_span")
        if not a_span or not o_span:
            continue
        # Clip to sentence bounds defensively (annotation off-by-ones happen).
        a_span = (max(0, a_span[0]), min(len(words), a_span[1]))
        o_span = (max(0, o_span[0]), min(len(words), o_span[1]))
        if a_span[0] >= a_span[1] or o_span[0] >= o_span[1]:
            continue

        category = normalize_category(ext.get("aspect_category") or ext.get("Category", ""))
        sentiment = normalize_sentiment(ext.get("sentiment") or ext.get("Polarity", ""))

        a_idx = _dedup_span(aspect_spans, a_span)
        o_idx = _dedup_span(opinion_spans, o_span)
        pairs.append((a_idx, o_idx, CATEGORY2ID[category], SENTIMENT2ID[sentiment]))

    return ExtractiveExample(words=words, aspect_spans=aspect_spans, opinion_spans=opinion_spans, pairs=pairs)


def load_examples(json_path: str) -> List[ExtractiveExample]:
    """Load and convert every review record in one ``*.json`` split file."""
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    examples: List[ExtractiveExample] = []
    for record in records:
        ex = build_example(record)
        if ex is not None:
            examples.append(ex)
    return examples


def load_split(labeled_dir: str, split: str) -> List[ExtractiveExample]:
    """Load a named split (``train``/``val``/``test``) from a labeled-data dir.

    Falls back to globbing ``*/{split}.json`` when ``labeled_dir`` is a parent
    directory containing multiple annotated sub-corpora (mirrors how
    ``dapt/dataset.py`` globs multiple source files).
    """
    direct = os.path.join(labeled_dir, f"{split}.json")
    if os.path.isfile(direct):
        return load_examples(direct)

    examples: List[ExtractiveExample] = []
    for path in sorted(glob.glob(os.path.join(labeled_dir, "*", f"{split}.json"))):
        examples.extend(load_examples(path))
    if not examples:
        raise FileNotFoundError(f"No '{split}.json' found under {labeled_dir}")
    return examples


class ExtractiveDataset(Dataset):
    """Thin ``Dataset`` wrapper — tokenization/label-projection happens in the
    collator because it needs the tokenizer's fast ``word_ids()`` mapping,
    which is only available once a whole batch is encoded together.
    """

    def __init__(self, examples: List[ExtractiveExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ExtractiveExample:
        return self.examples[idx]


# --------------------------------------------------------------------------- #
# Collator: word-level examples -> padded sub-word tensors + aligned labels    #
# --------------------------------------------------------------------------- #
@dataclass
class ExtractiveCollator:
    tokenizer: PreTrainedTokenizerBase
    max_seq_length: int = 160

    def __call__(self, batch: List[ExtractiveExample]) -> Dict[str, object]:
        word_lists = [ex.words for ex in batch]

        # Batched fast-tokenizer call over pre-split words: gives one
        # `word_ids(batch_index=i)` mapping per example, which is how word-level
        # gold spans get projected onto sub-word positions (see text_alignment.py).
        encoding = self.tokenizer(
            word_lists,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        batch_size, seq_len = encoding["input_ids"].shape
        aspect_labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        opinion_labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        word_ids_list: List[List[Optional[int]]] = []

        for i, ex in enumerate(batch):
            word_ids = encoding.word_ids(batch_index=i)
            word_ids_list.append(word_ids)

            aspect_word_tags = spans_to_bio(len(ex.words), ex.aspect_spans)
            opinion_word_tags = spans_to_bio(len(ex.words), ex.opinion_spans)
            aspect_labels[i] = torch.tensor(word_tags_to_subword_labels(word_ids, aspect_word_tags))
            opinion_labels[i] = torch.tensor(word_tags_to_subword_labels(word_ids, opinion_word_tags))

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "aspect_labels": aspect_labels,
            "opinion_labels": opinion_labels,
            "word_ids": word_ids_list,     # List[List[Optional[int]]], one per example
            "examples": batch,             # raw gold spans/pairs, needed for relation+cls loss
        }


def gold_token_spans(example: ExtractiveExample, word_ids: List[Optional[int]]) -> Tuple[
    List[Optional[Tuple[int, int]]], List[Optional[Tuple[int, int]]]
]:
    """Project every gold word-index span onto a (start_tok, end_tok) sub-word
    range for one example, using that example's ``word_ids`` mapping.

    A span can project to ``None`` if truncation cut off every word it
    covers — callers must skip those spans (see ``teacher/extractive_teacher.py``).
    """

    def _project(span: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        indices = word_span_to_token_indices(word_ids, span[0], span[1])
        if not indices:
            return None
        return (min(indices), max(indices) + 1)

    aspect_token_spans = [_project(s) for s in example.aspect_spans]
    opinion_token_spans = [_project(s) for s in example.opinion_spans]
    return aspect_token_spans, opinion_token_spans
