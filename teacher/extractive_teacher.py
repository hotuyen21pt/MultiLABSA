"""Teacher 2 — Extractive Teacher (T_E).

XLM-R backbone + three heads (Span / Relation / Classification), trained
end-to-end on the gold ASQP annotations. Produces the *same* quad schema as
the Generative Teacher so the two can be compared quad-for-quad in
``teacher/disagreement.py``.

Pipeline (both train and inference share the same span<->token alignment
machinery in ``utils/text_alignment.py``, so the two paths stay consistent):

    review -> XLM-R hidden states
           -> SpanHead: BIO tags -> decode Aspect/Opinion word spans
           -> mean-pool each span's sub-word hidden states -> span vectors
           -> RelationHead: score every (aspect, opinion) pair
           -> keep pairs above `relation_threshold`
           -> ClassificationHead: category + sentiment per kept pair
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from models.heads import ClassificationHead, RelationHead, SpanHead, mean_pool
from models.xlmr import XLMREncoder
from utils.data import ExtractiveExample, gold_token_spans
from utils.label_maps import (
    CATEGORIES,
    ID2CATEGORY,
    ID2SENTIMENT,
    SENTIMENTS,
)
from utils.schema import QuadPrediction
from utils.text_alignment import (
    bio_to_spans,
    decode_word_bio_from_subword_logits,
    span_text,
    word_span_to_token_indices,
    word_tokenize,
)


class ExtractiveTeacher(nn.Module):
    def __init__(
        self,
        backbone_name: str = "xlm-roberta-base",
        relation_proj_size: int = 256,
        classifier_proj_size: int = 256,
        dropout: float = 0.1,
        max_seq_length: int = 160,
        negative_relation_ratio: float = 2.0,
    ):
        super().__init__()
        self.backbone = XLMREncoder(backbone_name, dropout=dropout)
        hidden_size = self.backbone.hidden_size
        self.span_head = SpanHead(hidden_size, dropout=dropout)
        self.relation_head = RelationHead(hidden_size, proj_size=relation_proj_size)
        self.classification_head = ClassificationHead(
            hidden_size, len(CATEGORIES), len(SENTIMENTS), proj_size=classifier_proj_size, dropout=dropout
        )
        self.max_seq_length = max_seq_length
        # Weights the positive (linked) class in the relation BCE loss to
        # counter the natural sparsity of the gold adjacency matrix (most
        # aspect x opinion pairs in a sentence are NOT linked).
        self.negative_relation_ratio = negative_relation_ratio

    # ------------------------------------------------------------------ #
    # Shared backbone + span-tagging forward pass                          #
    # ------------------------------------------------------------------ #
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.backbone(input_ids, attention_mask)
        aspect_logits, opinion_logits = self.span_head(hidden_states)
        return hidden_states, aspect_logits, opinion_logits

    # ------------------------------------------------------------------ #
    # Training: teacher-forced on gold spans                               #
    # ------------------------------------------------------------------ #
    def compute_training_loss(self, batch: Dict[str, object], device: torch.device) -> Dict[str, torch.Tensor]:
        """One training step's loss, combining all three heads.

        The Relation and Classification heads are trained with *gold* spans
        (teacher forcing) rather than the Span Head's own (still noisy,
        early in training) predictions — standard practice for pipelined
        span-pair extraction models, since bootstrapping relation/
        classification training off unstable predicted spans would make
        their gradients chase a moving target.
        """
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        aspect_labels = batch["aspect_labels"].to(device)
        opinion_labels = batch["opinion_labels"].to(device)
        word_ids_list: List[List[Optional[int]]] = batch["word_ids"]
        examples: List[ExtractiveExample] = batch["examples"]

        hidden_states, aspect_logits, opinion_logits = self.forward(input_ids, attention_mask)

        # ---- 1. Span Head loss: token-level BIO cross-entropy -------------
        num_tags = aspect_logits.size(-1)
        span_loss = F.cross_entropy(
            aspect_logits.reshape(-1, num_tags), aspect_labels.reshape(-1), ignore_index=-100
        ) + F.cross_entropy(
            opinion_logits.reshape(-1, num_tags), opinion_labels.reshape(-1), ignore_index=-100
        )

        # ---- 2 & 3. Relation Head + Classification Head loss ---------------
        # Every example has a different number of gold aspect/opinion spans,
        # so pair construction is inherently per-example (ragged) — looped
        # explicitly rather than forced into a dense batch tensor.
        relation_losses: List[torch.Tensor] = []
        category_losses: List[torch.Tensor] = []
        sentiment_losses: List[torch.Tensor] = []
        pos_weight = torch.tensor(self.negative_relation_ratio, device=device)

        for i, example in enumerate(examples):
            word_ids = word_ids_list[i]
            aspect_token_spans, opinion_token_spans = gold_token_spans(example, word_ids)

            # Drop spans that truncation cut away entirely; remap the
            # remaining ones to a contiguous 0..N-1 index space.
            valid_aspect = [(j, s) for j, s in enumerate(aspect_token_spans) if s is not None]
            valid_opinion = [(j, s) for j, s in enumerate(opinion_token_spans) if s is not None]
            if not valid_aspect or not valid_opinion:
                continue
            aspect_remap = {orig: new for new, (orig, _) in enumerate(valid_aspect)}
            opinion_remap = {orig: new for new, (orig, _) in enumerate(valid_opinion)}

            aspect_repr = mean_pool(hidden_states[i], [s for _, s in valid_aspect])
            opinion_repr = mean_pool(hidden_states[i], [s for _, s in valid_opinion])

            relation_logits = self.relation_head(aspect_repr, opinion_repr)  # (Na, No)
            gold_matrix = torch.zeros_like(relation_logits)

            pair_aspect_reprs, pair_opinion_reprs, pair_cat_ids, pair_sent_ids = [], [], [], []
            for a_idx, o_idx, cat_id, sent_id in example.pairs:
                if a_idx not in aspect_remap or o_idx not in opinion_remap:
                    continue
                na, no = aspect_remap[a_idx], opinion_remap[o_idx]
                gold_matrix[na, no] = 1.0
                pair_aspect_reprs.append(aspect_repr[na])
                pair_opinion_reprs.append(opinion_repr[no])
                pair_cat_ids.append(cat_id)
                pair_sent_ids.append(sent_id)

            if gold_matrix.numel() > 0:
                relation_losses.append(
                    F.binary_cross_entropy_with_logits(relation_logits, gold_matrix, pos_weight=pos_weight)
                )

            if pair_cat_ids:
                cat_logits, sent_logits = self.classification_head(
                    torch.stack(pair_aspect_reprs), torch.stack(pair_opinion_reprs)
                )
                category_losses.append(
                    F.cross_entropy(cat_logits, torch.tensor(pair_cat_ids, device=device, dtype=torch.long))
                )
                sentiment_losses.append(
                    F.cross_entropy(sent_logits, torch.tensor(pair_sent_ids, device=device, dtype=torch.long))
                )

        zero = span_loss.new_zeros(())
        relation_loss = torch.stack(relation_losses).mean() if relation_losses else zero
        category_loss = torch.stack(category_losses).mean() if category_losses else zero
        sentiment_loss = torch.stack(sentiment_losses).mean() if sentiment_losses else zero

        return {
            "span": span_loss,
            "relation": relation_loss,
            "category": category_loss,
            "sentiment": sentiment_loss,
        }

    # ------------------------------------------------------------------ #
    # Inference: predicted spans -> predicted relations -> classification   #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        relation_threshold: float = 0.5,
    ) -> List[List[QuadPrediction]]:
        self.eval()
        word_lists = [word_tokenize(t) for t in texts]
        encoding = tokenizer(
            word_lists,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(device)

        hidden_states, aspect_logits, opinion_logits = self.forward(
            encoding["input_ids"], encoding["attention_mask"]
        )
        aspect_probs = F.softmax(aspect_logits, dim=-1)   # (B, T, 3)
        opinion_probs = F.softmax(opinion_logits, dim=-1)
        aspect_tag_ids = aspect_probs.argmax(dim=-1)       # (B, T)
        opinion_tag_ids = opinion_probs.argmax(dim=-1)

        results: List[List[QuadPrediction]] = []
        for i, words in enumerate(word_lists):
            word_ids = encoding.word_ids(batch_index=i)

            aspect_word_tags = decode_word_bio_from_subword_logits(word_ids, aspect_tag_ids[i].tolist())
            opinion_word_tags = decode_word_bio_from_subword_logits(word_ids, opinion_tag_ids[i].tolist())
            aspect_word_spans = bio_to_spans(aspect_word_tags)
            opinion_word_spans = bio_to_spans(opinion_word_tags)

            preds = self._decode_example(
                words=words,
                word_ids=word_ids,
                aspect_word_spans=aspect_word_spans,
                opinion_word_spans=opinion_word_spans,
                hidden_states_i=hidden_states[i],
                aspect_probs_i=aspect_probs[i],
                opinion_probs_i=opinion_probs[i],
                aspect_tag_ids_i=aspect_tag_ids[i],
                opinion_tag_ids_i=opinion_tag_ids[i],
                relation_threshold=relation_threshold,
            )
            results.append(preds)
        return results

    def _decode_example(
        self,
        words: List[str],
        word_ids: List[Optional[int]],
        aspect_word_spans,
        opinion_word_spans,
        hidden_states_i: torch.Tensor,
        aspect_probs_i: torch.Tensor,
        opinion_probs_i: torch.Tensor,
        aspect_tag_ids_i: torch.Tensor,
        opinion_tag_ids_i: torch.Tensor,
        relation_threshold: float,
    ) -> List[QuadPrediction]:
        if not aspect_word_spans or not opinion_word_spans:
            return []

        # Project decoded word spans onto sub-word token ranges for pooling;
        # drop any span truncation happened to swallow entirely.
        aspect_pairs = [(s, word_span_to_token_indices(word_ids, s[0], s[1])) for s in aspect_word_spans]
        aspect_pairs = [(s, (min(idx), max(idx) + 1)) for s, idx in aspect_pairs if idx]
        opinion_pairs = [(s, word_span_to_token_indices(word_ids, s[0], s[1])) for s in opinion_word_spans]
        opinion_pairs = [(s, (min(idx), max(idx) + 1)) for s, idx in opinion_pairs if idx]
        if not aspect_pairs or not opinion_pairs:
            return []

        aspect_word_spans = [p[0] for p in aspect_pairs]
        aspect_token_spans = [p[1] for p in aspect_pairs]
        opinion_word_spans = [p[0] for p in opinion_pairs]
        opinion_token_spans = [p[1] for p in opinion_pairs]

        aspect_repr = mean_pool(hidden_states_i, aspect_token_spans)
        opinion_repr = mean_pool(hidden_states_i, opinion_token_spans)
        relation_probs = torch.sigmoid(self.relation_head(aspect_repr, opinion_repr))  # (Na, No)

        # Keep EVERY pair above threshold, not just the argmax opinion per
        # aspect — the gold data routinely links one aspect to several
        # opinions (e.g. "pool" <- "beautiful", "a good size", "very clean").
        kept = [
            (a, o, float(relation_probs[a, o]))
            for a in range(relation_probs.size(0))
            for o in range(relation_probs.size(1))
            if relation_probs[a, o] > relation_threshold
        ]
        if not kept:
            return []

        a_idx = torch.tensor([p[0] for p in kept], device=hidden_states_i.device)
        o_idx = torch.tensor([p[1] for p in kept], device=hidden_states_i.device)
        cat_logits, sent_logits = self.classification_head(aspect_repr[a_idx], opinion_repr[o_idx])
        cat_probs = F.softmax(cat_logits, dim=-1)
        sent_probs = F.softmax(sent_logits, dim=-1)
        cat_ids = cat_probs.argmax(dim=-1)
        sent_ids = sent_probs.argmax(dim=-1)

        preds: List[QuadPrediction] = []
        for k, (a, o, rel_prob) in enumerate(kept):
            asp_conf = self._span_confidence(aspect_probs_i, aspect_tag_ids_i, aspect_token_spans[a])
            opn_conf = self._span_confidence(opinion_probs_i, opinion_tag_ids_i, opinion_token_spans[o])
            cat_id = int(cat_ids[k])
            sent_id = int(sent_ids[k])
            cat_conf = float(cat_probs[k, cat_id])
            sent_conf = float(sent_probs[k, sent_id])

            # Conf_E: geometric mean over every head's confidence in this
            # quad — a single weak link (e.g. an uncertain relation pairing)
            # pulls the whole quad's confidence down, which is the desired
            # behaviour for a pipelined (span -> relation -> class) prediction.
            conf_e = (asp_conf * opn_conf * rel_prob * cat_conf * sent_conf) ** 0.2

            preds.append(
                QuadPrediction(
                    aspect=span_text(words, *aspect_word_spans[a]),
                    opinion=span_text(words, *opinion_word_spans[o]),
                    category=ID2CATEGORY[cat_id],
                    sentiment=ID2SENTIMENT[sent_id],
                    confidence=conf_e,
                    source="extractive",
                    aspect_span=aspect_word_spans[a],
                    opinion_span=opinion_word_spans[o],
                )
            )
        return preds

    @staticmethod
    def _span_confidence(probs: torch.Tensor, tag_ids: torch.Tensor, token_span) -> float:
        """Mean predicted-tag probability across a span's sub-word tokens."""
        start, end = token_span
        if end <= start:
            return 0.0
        span_probs = probs[start:end]
        span_tags = tag_ids[start:end]
        gathered = span_probs.gather(1, span_tags.unsqueeze(-1)).squeeze(-1)
        return float(gathered.mean())
