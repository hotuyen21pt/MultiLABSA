"""Data collator that turns raw review strings into a padded DAPT batch.

For each review the collator:
    1. tokenizes to at most ``max_seq_length`` sub-word tokens (no special
       tokens — span corruption adds its own sentinels + EOS),
    2. applies :class:`masking.SpanCorruption` to get ``input_ids`` / ``labels``,
    3. dynamically pads the batch (``pad_token`` for inputs, ``-100`` for labels
       so padded positions are ignored by the cross-entropy loss).

The returned dict (``input_ids``, ``attention_mask``, ``labels``) is exactly
what :class:`transformers.MT5ForConditionalGeneration` expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase

from masking import SpanCorruption


@dataclass
class DataCollatorForSpanCorruption:
    """Collate raw texts into a span-corruption training batch.

    Args:
        tokenizer: the mT5 tokenizer (for tokenizing + pad id).
        span_corruption: the configured span-corruption operator.
        max_seq_length: truncation length for the *raw* token sequence.
    """

    tokenizer: PreTrainedTokenizerBase
    span_corruption: SpanCorruption
    max_seq_length: int = 256

    def __post_init__(self) -> None:
        self.pad_token_id: int = self.tokenizer.pad_token_id
        # T5/mT5 use -100 to mask out positions in the loss.
        self.label_pad_token_id: int = -100

    def __call__(self, batch: List[str]) -> Dict[str, torch.Tensor]:
        input_seqs: List[List[int]] = []
        label_seqs: List[List[int]] = []

        for text in batch:
            token_ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_length,
            ).input_ids
            if not token_ids:  # skip degenerate rows defensively
                continue
            input_ids, labels = self.span_corruption.mask_tokens(token_ids)
            input_seqs.append(input_ids)
            label_seqs.append(labels)

        # Guard against an entirely empty batch (all rows degenerate).
        if not input_seqs:
            eos = self.tokenizer.eos_token_id
            input_seqs, label_seqs = [[eos]], [[eos]]

        # ---- dynamic padding to the longest sequence in the batch ---------
        max_in = max(len(s) for s in input_seqs)
        max_lab = max(len(s) for s in label_seqs)

        input_ids = torch.full((len(input_seqs), max_in), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(input_seqs), max_in), dtype=torch.long)
        labels = torch.full((len(label_seqs), max_lab), self.label_pad_token_id, dtype=torch.long)

        for i, (inp, lab) in enumerate(zip(input_seqs, label_seqs)):
            input_ids[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
            attention_mask[i, : len(inp)] = 1
            labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
