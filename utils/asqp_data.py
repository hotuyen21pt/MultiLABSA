"""Loads ``data_final/labeled_data/*/{train,val,test}.json`` into
(review, linearized-quad-string) pairs for **supervised seq2seq fine-tuning**
of mT5 on the ASQP quad-generation task.

This is what turns the plain DAPT backbone (``hotel-mt5/`` — domain-adapted
via span-corruption denoising only, see ``dapt/``) into a checkpoint that
actually knows how to emit ``models/mt5.py``'s
``aspect | opinion | category | sentiment ; ...`` format, i.e. the
"already fine-tuned" ``hotel-mt5`` that ``teacher/generative_teacher.py``
assumes it's loading. See ``train_asqp_mt5.py`` for the training loop that
consumes this module.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import List, Set, Tuple

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from models.mt5 import linearize_quads
from utils.label_maps import normalize_category, normalize_sentiment


def load_asqp_pairs(json_path: str) -> List[Tuple[str, str]]:
    """Convert one raw ``*.json`` split file into (review, target_string) pairs.

    Duplicate quads within a review (repeated ``extraction`` rows differing
    only in casing/whitespace) are collapsed — the decoder should not be
    taught to emit the exact same quad twice.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    pairs: List[Tuple[str, str]] = []
    for record in records:
        review = (record.get("review") or "").strip()
        extraction = record.get("extraction", [])
        if not review or not extraction:
            continue

        quads: List[dict] = []
        seen: Set[tuple] = set()
        for ext in extraction:
            aspect = (ext.get("aspect_term") or "").strip()
            opinion = (ext.get("opinion_term") or "").strip()
            if not aspect or not opinion:
                continue
            category = normalize_category(ext.get("aspect_category") or ext.get("Category", ""))
            sentiment = normalize_sentiment(ext.get("sentiment") or ext.get("Polarity", "")).capitalize()

            key = (aspect.lower(), opinion.lower(), category, sentiment)
            if key in seen:
                continue
            seen.add(key)
            quads.append({"aspect": aspect, "opinion": opinion, "category": category, "sentiment": sentiment})

        if not quads:
            continue
        pairs.append((review, linearize_quads(quads)))
    return pairs


def load_asqp_split(labeled_dir: str, split: str) -> List[Tuple[str, str]]:
    """Load a named split (``train``/``val``/``test``), same layout convention
    as ``utils.data.load_split`` (single file or ``*/{split}.json`` glob)."""
    direct = os.path.join(labeled_dir, f"{split}.json")
    if os.path.isfile(direct):
        return load_asqp_pairs(direct)

    pairs: List[Tuple[str, str]] = []
    for path in sorted(glob.glob(os.path.join(labeled_dir, "*", f"{split}.json"))):
        pairs.extend(load_asqp_pairs(path))
    if not pairs:
        raise FileNotFoundError(f"No '{split}.json' found under {labeled_dir}")
    return pairs


class ASQPDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.pairs[idx]


@dataclass
class ASQPCollator:
    """Tokenizes (source, target) text pairs into a padded seq2seq batch."""

    tokenizer: PreTrainedTokenizerBase
    max_source_length: int = 256
    max_target_length: int = 160

    def __call__(self, batch: List[Tuple[str, str]]) -> dict:
        sources = [s for s, _ in batch]
        targets = [t for _, t in batch]

        model_inputs = self.tokenizer(
            sources, padding=True, truncation=True, max_length=self.max_source_length, return_tensors="pt",
        )
        # `text_target=` tokenizes with the decoder-side special-token
        # convention (no encoder BOS quirks); this is the modern replacement
        # for the deprecated `as_target_tokenizer()` context manager.
        labels = self.tokenizer(
            text_target=targets, padding=True, truncation=True, max_length=self.max_target_length,
            return_tensors="pt",
        )
        label_ids = labels["input_ids"]
        # Padded target positions must not contribute to the loss.
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = label_ids
        return model_inputs
