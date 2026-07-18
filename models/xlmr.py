"""XLM-R backbone for the Extractive Teacher (T_E).

A thin wrapper around ``transformers.AutoModel`` — kept separate from
``models/heads.py`` so the shared encoder and the task-specific heads can be
frozen/unfrozen or swapped independently (e.g. discriminative fine-tuning,
or later replacing XLM-R with another multilingual encoder without touching
head code).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class XLMREncoder(nn.Module):
    """Wraps ``xlm-roberta-{base,large}`` and exposes per-token hidden states.

    The Span/Relation/Classification heads all consume
    ``last_hidden_state`` (B, T, H); no pooler is used since every downstream
    task needs token- or span-level (not sentence-level) representations.
    """

    def __init__(self, model_name: str = "xlm-roberta-base", dropout: float = 0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        # Encoder-internal dropout — separate from the task-head dropout
        # applied on top of its output (see models/heads.py).
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size: int = self.backbone.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return ``last_hidden_state``: (batch, seq_len, hidden_size)."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
