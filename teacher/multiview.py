"""Multi-view Pseudo-label Generation front-end for the Generative Teacher (T_G).

The unlabeled corpus is heavily multilingual (see ``language_stats.txt``:
~57% en, ~20% vi, plus a long tail of fr/de/es/nl/it/ko/ru/ja/zh/...) while
T_G was fine-tuned almost exclusively on English gold quads
(``utils/asqp_data.py``). A single forward pass on a non-English review asks
T_G to generalize both across domain *and* language at once. This module
gives T_G three independent chances per review instead:

    View 1 - Native          : the review as-is.
    View 2 - Translate to EN : machine-translated into English (T_G's
                                strongest language) before prediction.
    View 3 - Back-translation: View 2 translated back into the native
                                language (a paraphrase of the original),
                                predicted again.

and reconciles the three predictions via self-consistency voting: a quad
survives only if enough of the 3 views independently proposed it, and its
confidence is boosted by how many views agreed.

Output shape is identical to ``GenerativeTeacher.predict`` --
``List[List[QuadPrediction]]`` -- so ``MultiViewGenerativeTeacher`` is a
drop-in substitute for it in ``train.py``'s pseudo-labeling loop; nothing in
``teacher/disagreement.py`` or ``teacher/confidence_fusion.py`` needs to
change.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from teacher.generative_teacher import GenerativeTeacher
from teacher.translator import ENGLISH_CODE, NLLBTranslator, nllb_code
from utils.schema import QuadPrediction


@dataclass
class MultiViewWeights:
    """Self-consistency voting knobs.

    Views vote on ``(category, sentiment)`` rather than aspect/opinion
    surface text: category/sentiment come from a fixed, language-invariant
    label vocabulary (``utils/label_maps.py``), while View 2's aspect/opinion
    text is in English and Views 1/3's is in the native language -- they
    have no shared coordinate system to fuzzy-match against directly.
    """

    min_agreeing_views: int = 2                # of 3; below this the quad is dropped
    confidence_boost_per_view: float = 0.15    # extra multiplier per corroborating view beyond the first
    pivot_lang_for_english: str = "fra_Latn"   # back-translation pivot when the review is already English


def _grouped_translate(
    translator: NLLBTranslator, texts: List[str], pairs: List[Tuple[str, str]]
) -> List[str]:
    """Translate ``texts``, each with its own ``(src_code, tgt_code)`` pair.

    Groups items sharing the same pair into a single batched ``translate()``
    call (instead of one call per review) since a mini-batch from
    ``train.py`` is typically a handful of reviews spread across only a few
    of the ~11 mapped languages.
    """
    results: List[Optional[str]] = [None] * len(texts)
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, pair in enumerate(pairs):
        groups[pair].append(i)

    for (src, tgt), indices in groups.items():
        group_texts = [texts[i] for i in indices]
        translated = translator.translate(group_texts, src, tgt)
        for i, t in zip(indices, translated):
            results[i] = t
    return results  # type: ignore[return-value]


class MultiViewGenerativeTeacher:
    """Runs T_G on 3 views of each review and reconciles them via
    self-consistency voting into a single ``List[QuadPrediction]`` per review.
    """

    def __init__(
        self,
        generative_teacher: GenerativeTeacher,
        translator: NLLBTranslator,
        weights: Optional[MultiViewWeights] = None,
    ):
        self.teacher = generative_teacher
        self.translator = translator
        self.weights = weights or MultiViewWeights()

    def predict(
        self, texts: List[str], langs: Optional[List[Optional[str]]] = None
    ) -> List[List[QuadPrediction]]:
        langs = langs if langs is not None else [None] * len(texts)
        src_codes = [nllb_code(l) for l in langs]

        # View 2: translate every review to English (native -> EN; a no-op
        # per grouped pair where the review is already English).
        translated = _grouped_translate(
            self.translator, texts, [(src, ENGLISH_CODE) for src in src_codes]
        )

        # View 3: back-translate the English version to the native language
        # (EN -> native, a genuine paraphrase). Already-English reviews have
        # no distinct native language to round-trip through, so they pivot
        # through pivot_lang_for_english instead (EN -> FR -> EN) to still
        # get a real paraphrase rather than an identical no-op copy of
        # View 1/2.
        pivot = self.weights.pivot_lang_for_english
        bt_targets = [pivot if src == ENGLISH_CODE else src for src in src_codes]
        backtranslated = _grouped_translate(
            self.translator, translated, [(ENGLISH_CODE, tgt) for tgt in bt_targets]
        )
        pivot_indices = [i for i, src in enumerate(src_codes) if src == ENGLISH_CODE]
        if pivot_indices:
            pivot_texts = [backtranslated[i] for i in pivot_indices]
            pivoted_back = self.translator.translate(pivot_texts, pivot, ENGLISH_CODE)
            for i, t in zip(pivot_indices, pivoted_back):
                backtranslated[i] = t

        quads_native = self.teacher.predict(texts)
        quads_translated = self.teacher.predict(translated)
        quads_backtranslated = self.teacher.predict(backtranslated)

        return [
            self._vote(native, en, bt)
            for native, en, bt in zip(quads_native, quads_translated, quads_backtranslated)
        ]

    def _vote(
        self,
        native: List[QuadPrediction],
        translated: List[QuadPrediction],
        backtranslated: List[QuadPrediction],
    ) -> List[QuadPrediction]:
        """Self-consistency vote across the 3 views for one review.

        A ``(category, sentiment)`` key needs >= ``min_agreeing_views``
        distinct views proposing it to survive. The surviving quad keeps the
        *native* view's surface aspect/opinion text so it stays grounded in
        the original source review for ``disagreement.py``'s hallucination
        filter, falling back to the back-translated (still native-language)
        view and finally the English-translated view when the native view
        itself didn't propose that key.
        """
        views = [native, translated, backtranslated]
        keys_per_view = [{(q.category, q.sentiment) for q in v} for v in views]
        all_keys = keys_per_view[0] | keys_per_view[1] | keys_per_view[2]

        voted: List[QuadPrediction] = []
        for key in all_keys:
            agreeing = sum(1 for ks in keys_per_view if key in ks)
            if agreeing < self.weights.min_agreeing_views:
                continue

            representative = self._pick_representative(key, native, backtranslated, translated)
            if representative is None:
                continue

            contributing_confidences = [
                v[0].confidence for v, ks in zip(views, keys_per_view) if key in ks
            ]
            base_conf = sum(contributing_confidences) / len(contributing_confidences)
            boost = 1.0 + self.weights.confidence_boost_per_view * (agreeing - 1)
            boosted_conf = min(1.0, base_conf * boost)

            voted.append(
                QuadPrediction(
                    aspect=representative.aspect,
                    opinion=representative.opinion,
                    category=representative.category,
                    sentiment=representative.sentiment,
                    confidence=boosted_conf,
                    source="generative",
                )
            )
        return voted

    @staticmethod
    def _pick_representative(
        key: Tuple[str, str], *views_in_priority_order: List[QuadPrediction]
    ) -> Optional[QuadPrediction]:
        for view in views_in_priority_order:
            for q in view:
                if (q.category, q.sentiment) == key:
                    return q
        return None
