"""Hybrid student model (MultiLABSA.docx §4.7 + §4.9).

A pure seq2seq decoder cannot learn from *partial* pseudo-labels (§4.9: "student
should not be only a seq2seq decoder"). So the student couples:

    * a generative QUAD backbone  -> mT5 encoder-decoder (full-quad generation);
    * auxiliary heads on the encoder hidden states, reusing the Extractive
      Teacher's heads so partial supervision can train individual elements:
          - SpanHead            : AT / OT BIO tagging
          - RelationHead        : AT<->OT biaffine linking
          - ClassificationHead  : AC + SP per pair
    * two implicit heads (§4.7): p(AT=NULL | x), p(OT=NULL | x), so an implicit
      aspect/opinion is a first-class prediction rather than an empty string.

The generative and extractive views share ONE encoder, so the auxiliary losses
regularise the same representation the decoder generates from.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration

from models.heads import ClassificationHead, RelationHead, SpanHead
from utils.label_maps import CATEGORIES, SENTIMENTS


class ImplicitHead(nn.Module):
    """p(element = NULL | sentence) from the mean-pooled encoder state (§4.7)."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, sentence_repr: torch.Tensor) -> torch.Tensor:
        """sentence_repr: (B, H) -> (B,) logit for 'this element is implicit'."""
        return self.proj(self.dropout(sentence_repr)).squeeze(-1)


class HybridStudent(nn.Module):
    def __init__(
        self,
        backbone_name: str = "hotel-mt5-asqp",
        relation_proj_size: int = 256,
        classifier_proj_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Generative backbone (full-quad decoding + shared multilingual encoder).
        self.backbone = MT5ForConditionalGeneration.from_pretrained(backbone_name)
        hidden = self.backbone.config.d_model

        # Auxiliary heads on the encoder (enable element-wise / partial supervision).
        self.span_head = SpanHead(hidden, dropout=dropout)
        self.relation_head = RelationHead(hidden, proj_size=relation_proj_size)
        self.classification_head = ClassificationHead(
            hidden, len(CATEGORIES), len(SENTIMENTS), proj_size=classifier_proj_size, dropout=dropout
        )
        self.implicit_at = ImplicitHead(hidden, dropout)
        self.implicit_ot = ImplicitHead(hidden, dropout)

    # ------------------------------------------------------------------ #
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encoder hidden states shared by every auxiliary head. (B, T, H)"""
        return self.backbone.get_encoder()(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool over valid tokens -> (B, H)."""
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)

    def aux_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """All auxiliary-head outputs for a batch (for partial/relation losses)."""
        hidden = self.encode(input_ids, attention_mask)
        aspect_logits, opinion_logits = self.span_head(hidden)
        sent_repr = self._masked_mean(hidden, attention_mask)
        return {
            "hidden": hidden,
            "aspect_logits": aspect_logits,
            "opinion_logits": opinion_logits,
            "implicit_at_logit": self.implicit_at(sent_repr),
            "implicit_ot_logit": self.implicit_ot(sent_repr),
        }

    # ------------------------------------------------------------------ #
    def generation_loss(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Standard seq2seq CE for full-quad generation (L_sup / L_full)."""
        return self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens: int = 160, num_beams: int = 4):
        return self.backbone.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, num_beams=num_beams,
        )

    def save_pretrained(self, path: str) -> None:
        self.backbone.save_pretrained(path)
        torch.save(
            {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")},
            f"{path}/aux_heads.pt",
        )
