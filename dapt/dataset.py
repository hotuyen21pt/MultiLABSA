"""Corpus loading + temperature-balanced language sampling for Hotel-DAPT.

The training corpus is the *unlabeled* hotel reviews produced earlier
(``data_final/unlabeled_data/hotel_review*_lang.csv``), each row carrying a
``review`` text and a detected ``language`` code.

Because the language distribution is heavily skewed (English + Vietnamese make
up ~77%), we apply **temperature sampling** (as in mT5/XLM) so lower-resource
languages are seen more often than their raw frequency:

    q_l ∝ n_l ** (1 / T)

``T = 1`` reproduces the natural distribution; larger ``T`` flattens it towards
uniform, up-weighting rare languages.  Each training draw first samples a
language from ``q`` and then a review uniformly within that language.

Two modes:
    * ``split="train"`` — a *sampling* dataset of fixed ``epoch_size`` whose
      draws follow the temperature distribution (reproducible per epoch).
    * ``split="val"``   — a *fixed* held-out set for a stable validation loss.
"""

from __future__ import annotations

import glob
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils import Config, setup_logging

logger = setup_logging()


def _load_reviews_by_language(cfg: Config) -> Dict[str, List[str]]:
    """Read every corpus CSV and group cleaned review texts by language code.

    Reads only the two needed columns, in chunks, so the 150-300 MB files stay
    memory-friendly.  Rows that are empty or shorter than ``cfg.min_chars`` are
    dropped, and an optional per-language cap bounds memory.
    """
    pattern = os.path.join(cfg.data_dir, cfg.file_glob)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No corpus files matched: {pattern}")

    by_lang: Dict[str, List[str]] = defaultdict(list)
    cap = cfg.max_samples_per_language

    for path in files:
        logger.info("Loading corpus file: %s", os.path.basename(path))
        reader = pd.read_csv(
            path,
            usecols=[cfg.text_column, cfg.language_column],
            chunksize=100_000,
        )
        for chunk in reader:
            texts = chunk[cfg.text_column].astype(str)
            langs = chunk[cfg.language_column].fillna("unknown").astype(str)
            for text, lang in zip(texts, langs):
                text = text.strip()
                if len(text) < cfg.min_chars:
                    continue
                if cap is not None and len(by_lang[lang]) >= cap:
                    continue
                by_lang[lang].append(text)

    total = sum(len(v) for v in by_lang.values())
    logger.info("Loaded %d reviews across %d languages", total, len(by_lang))
    return dict(by_lang)


class HotelReviewDataset(Dataset):
    """Map-style dataset yielding raw review strings (corruption happens later).

    Args:
        cfg: the run configuration.
        split: ``"train"`` (temperature-sampled) or ``"val"`` (fixed hold-out).
        by_lang: optional pre-loaded ``{language: [texts]}`` so train and val can
            share one disk read; if ``None`` the corpus is read from disk.
    """

    def __init__(
        self,
        cfg: Config,
        split: str = "train",
        by_lang: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        assert split in {"train", "val"}
        self.cfg = cfg
        self.split = split

        if by_lang is None:
            by_lang = _load_reviews_by_language(cfg)

        # Deterministic per-language train/val split.
        rng = random.Random(cfg.seed)
        train_by_lang: Dict[str, List[str]] = {}
        val_texts: List[str] = []
        for lang, texts in by_lang.items():
            texts = list(texts)
            rng.shuffle(texts)
            n_val = max(1, int(len(texts) * cfg.val_fraction)) if len(texts) > 1 else 0
            val_texts.extend(texts[:n_val])
            train_part = texts[n_val:]
            if train_part:
                train_by_lang[lang] = train_part

        if split == "val":
            # Fixed, capped validation set (shuffled once for language mixing).
            rng.shuffle(val_texts)
            self.examples: List[str] = val_texts[: cfg.max_val_samples]
            logger.info("Validation set: %d examples", len(self.examples))
            return

        # ---- training (sampling) mode -------------------------------------
        self.languages: List[str] = sorted(train_by_lang.keys())
        self.texts_by_lang: Dict[str, List[str]] = train_by_lang
        counts = np.array([len(train_by_lang[l]) for l in self.languages], dtype=np.float64)

        # Temperature-adjusted language probabilities: q_l ∝ n_l ** (1/T).
        exponent = 1.0 / cfg.sampling_temperature
        weighted = counts ** exponent
        self.lang_probs: np.ndarray = weighted / weighted.sum()

        n_train_total = int(counts.sum())
        self.epoch_size: int = cfg.epoch_size or n_train_total
        self._epoch = 0

        logger.info(
            "Train set: %d reviews, %d languages, epoch_size=%d, T=%.2f",
            n_train_total, len(self.languages), self.epoch_size, cfg.sampling_temperature,
        )
        self._log_top_sampling_probs()

    def _log_top_sampling_probs(self, top: int = 8) -> None:
        """Log the languages the sampler will draw most often."""
        order = np.argsort(-self.lang_probs)[:top]
        summary = ", ".join(
            f"{self.languages[i]}={self.lang_probs[i]*100:.1f}%" for i in order
        )
        logger.info("Top sampling probabilities: %s", summary)

    def set_epoch(self, epoch: int) -> None:
        """Update the epoch so the sampling stream varies between epochs.

        The trainer calls this at the start of every epoch; combined with the
        per-index seeding below it keeps sampling reproducible *and* resumable
        while still showing different reviews each epoch.
        """
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.examples) if self.split == "val" else self.epoch_size

    def __getitem__(self, idx: int) -> str:
        if self.split == "val":
            return self.examples[idx]

        # Reproducible draw keyed by (epoch, idx): same seed => same sample.
        seed = (self.cfg.seed * 1_000_003 + self._epoch * self.epoch_size + idx) % (2**32)
        rng = np.random.default_rng(seed)
        lang = self.languages[int(rng.choice(len(self.languages), p=self.lang_probs))]
        pool = self.texts_by_lang[lang]
        return pool[int(rng.integers(0, len(pool)))]


def build_datasets(cfg: Config) -> "tuple[HotelReviewDataset, HotelReviewDataset]":
    """Read the corpus once and build the train + validation datasets."""
    by_lang = _load_reviews_by_language(cfg)
    train_ds = HotelReviewDataset(cfg, split="train", by_lang=by_lang)
    val_ds = HotelReviewDataset(cfg, split="val", by_lang=by_lang)
    return train_ds, val_ds
