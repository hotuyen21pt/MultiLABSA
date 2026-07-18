"""NLLB-200 translator used to build the Multi-view Pseudo-label Generation
inputs for the Generative Teacher (T_G).

One self-hosted many-to-many model (``facebook/nllb-200-distilled-600M``)
covers every language pair the multi-view stage needs (native->English for
View 2, English->native for View 3's back-translation) with a single
checkpoint -- unlike MarianMT, which would need a separate checkpoint per
language pair to match the unlabeled corpus's long language tail (see
``language_stats.txt``: en/vi/fr/de/es/nl/it/ko/ru/ja/zh plus a long tail).
"""

from __future__ import annotations

from typing import List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# fastText language codes (as written by add_language.py's "language" column)
# -> NLLB-200 FLORES-200 codes. Covers the languages that actually appear in
# data_final/unlabeled_data (per language_stats.txt); anything else falls
# back to ENGLISH_CODE in nllb_code() below.
FASTTEXT_TO_NLLB = {
    "en": "eng_Latn",
    "vi": "vie_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "nl": "nld_Latn",
    "it": "ita_Latn",
    "ko": "kor_Hang",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
}

ENGLISH_CODE = "eng_Latn"


def nllb_code(fasttext_lang: Optional[str]) -> str:
    """Map a fastText language code to its NLLB-200 FLORES-200 code.

    Unknown or missing codes fall back to English: treating unidentified
    text as already-English simply skips translation for that review
    (View 2 becomes a no-op) rather than risking mistranslation by guessing
    a source language for a code we have no mapping for.
    """
    if not fasttext_lang:
        return ENGLISH_CODE
    return FASTTEXT_TO_NLLB.get(fasttext_lang, ENGLISH_CODE)


class NLLBTranslator:
    """Thin wrapper around a single NLLB-200 checkpoint for many-to-many MT."""

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: Optional["torch.device"] = None,
        max_length: int = 256,
        num_beams: int = 4,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.num_beams = num_beams

    @torch.no_grad()
    def translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """Translate a batch of same-source-language texts to ``tgt_lang``."""
        if not texts:
            return []
        if src_lang == tgt_lang:
            return list(texts)

        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=self.max_length,
            num_beams=self.num_beams,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
