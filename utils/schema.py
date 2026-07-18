"""Shared data contracts passed between the two teachers and the fusion stage.

Keeping these as plain dataclasses (rather than dicts) means every stage of
the pipeline — generation, extraction, disagreement, fusion — is type-checked
against the same quad schema described in the task spec::

    {"aspect": ..., "opinion": ..., "category": ..., "sentiment": ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class QuadPrediction:
    """One (aspect, opinion, category, sentiment) prediction from ONE teacher."""

    aspect: str
    opinion: str
    category: str
    sentiment: str
    confidence: float               # Conf_G (generative) or Conf_E (extractive)
    source: str                     # "generative" | "extractive"
    # Extractive-only provenance (word-index spans); unset for the generative
    # teacher, which never grounds its text in explicit source positions.
    aspect_span: Optional[Tuple[int, int]] = None
    opinion_span: Optional[Tuple[int, int]] = None

    def to_dict(self) -> dict:
        return {
            "aspect": self.aspect,
            "opinion": self.opinion,
            "category": self.category,
            "sentiment": self.sentiment,
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
        }


@dataclass
class MergedPrediction:
    """Output of the disagreement module: one quad + both teachers' evidence.

    ``conf_g`` / ``conf_e`` are 0.0 when the corresponding teacher did not
    propose (a matching version of) this quad — that absence is itself
    informative and flows straight into confidence fusion.
    """

    aspect: str
    opinion: str
    category: str
    sentiment: str
    conf_g: float
    conf_e: float
    agreement: float
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "aspect": self.aspect,
            "opinion": self.opinion,
            "category": self.category,
            "sentiment": self.sentiment,
            "conf_g": round(float(self.conf_g), 4),
            "conf_e": round(float(self.conf_e), 4),
            "agreement": round(float(self.agreement), 4),
            "sources": list(self.sources),
        }
