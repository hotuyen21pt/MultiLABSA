"""Multilingual transfer baselines (MultiLABSA.docx §3, M1 & M3).

Both are inference-only (no training) so they run directly against the shared
``hotel-mt5-asqp`` generative teacher — the natural transfer floor to compare
MERA-XQUAD against.

    M1 Zero-shot       : predict directly on the native-language review.
    M3 Translate-test  : translate the review to English (T_G's strong language)
                         with NLLB, predict there. Aspect/opinion surface stays
                         English — the known translate-test limitation, reported
                         honestly rather than projected back.
"""

from __future__ import annotations

from typing import List, Optional

from teacher.generative_teacher import GenerativeTeacher
from teacher.translator import ENGLISH_CODE, NLLBTranslator, nllb_code
from utils.schema import QuadPrediction

Quad = dict


def _to_dicts(quads: List[QuadPrediction]) -> List[Quad]:
    return [{"aspect": q.aspect, "opinion": q.opinion, "category": q.category,
             "sentiment": q.sentiment, "conf_g": round(q.confidence, 4)} for q in quads]


class ZeroShotBaseline:
    """M1 — T_G predicts directly on the native review."""

    def __init__(self, teacher: GenerativeTeacher):
        self.teacher = teacher

    def predict(self, texts: List[str], langs: Optional[List] = None) -> List[List[Quad]]:
        return [_to_dicts(qs) for qs in self.teacher.predict(texts)]


class TranslateTestBaseline:
    """M3 — translate native -> English, then T_G predicts."""

    def __init__(self, teacher: GenerativeTeacher, translator: NLLBTranslator):
        self.teacher = teacher
        self.translator = translator

    def predict(self, texts: List[str], langs: Optional[List] = None) -> List[List[Quad]]:
        langs = langs or [None] * len(texts)
        english: List[str] = []
        for text, lang in zip(texts, langs):
            src = nllb_code(lang)
            english.append(text if src == ENGLISH_CODE else self.translator.translate([text], src, ENGLISH_CODE)[0])
        return [_to_dicts(qs) for qs in self.teacher.predict(english)]


BASELINES = {"zero_shot": ZeroShotBaseline, "translate_test": TranslateTestBaseline}
