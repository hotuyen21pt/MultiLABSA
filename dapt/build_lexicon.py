"""Data-driven construction of the biased-masking lexicon for Hotel-DAPT.

Instead of hand-listing words, we *derive* the lexicon from the unlabeled hotel
corpus with corpus-linguistics statistics, per language:

  1. Terminology (open class) — ranked by domain salience:
       * unigrams by **weirdness** = f_hotel(w) / f_general(w)
         (f_general comes from the `wordfreq` background frequencies);
       * bigrams by **PMI** (pointwise mutual information) = collocation
         strength inside the corpus (e.g. "swimming pool", "check in").
     No POS tagger needed -> language-agnostic and cheap.

  2. Opinion words (seed + expansion) — a closed seed set intersected with the
     corpus, optionally expanded via **fastText nearest neighbours** trained on
     the corpus itself.

  3. Negation / Intensifier (closed class) — curated finite lists (these are
     grammatical function words; a data-driven method adds nothing here).

  4. Weighting — every kept term gets a **continuous salience score in [0, 1]**
     (min-max of its ranking metric). `SpanCorruption` turns this into a mask
     weight of ``1 + boost * salience`` (smoother than a flat on/off boost).

Outputs:
    <out_dir>/lexicon_<lang>.json   per-language, with per-category scores
    <out_dir>/lexicon.json          merged {term: weight} used by masking.py

Run (see requirements-lexicon note in the repo Dockerfile.lexicon):
    python build_lexicon.py --data_dir ../data_final/unlabeled_data --expand
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

from utils import LEXICON, setup_logging

logger = setup_logging()

# Languages with enough volume to build a meaningful lexicon (all supported by
# wordfreq). Override with --languages.
DEFAULT_LANGUAGES = ["en", "vi", "fr", "de", "es", "it", "nl", "ru", "ja", "ko", "zh"]

# A light regex tokenizer used as a fallback when wordfreq cannot tokenize a
# language (keeps unicode letters, drops punctuation/digits).
_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)

# Closed-class seed sets reused from the hand-written lexicon in utils.py.
_NEGATIONS = set(LEXICON["negations"])
_INTENSIFIERS = set(LEXICON["intensifiers"])
_OPINION_SEEDS = set(LEXICON["opinions"])


# --------------------------------------------------------------------------- #
# Corpus loading                                                                #
# --------------------------------------------------------------------------- #
def load_reviews(
    data_dir: str,
    languages: List[str],
    file_glob: str = "*_lang.csv",
    text_column: str = "review",
    language_column: str = "language",
    max_per_lang: Optional[int] = 200_000,
) -> Dict[str, List[str]]:
    """Collect up to ``max_per_lang`` reviews per target language from the CSVs."""
    import glob

    files = sorted(glob.glob(os.path.join(data_dir, file_glob)))
    if not files:
        raise FileNotFoundError(f"No corpus files under {data_dir}/{file_glob}")

    want = set(languages)
    by_lang: Dict[str, List[str]] = {l: [] for l in languages}
    for path in files:
        for chunk in pd.read_csv(path, usecols=[text_column, language_column], chunksize=100_000):
            langs = chunk[language_column].astype(str)
            texts = chunk[text_column].astype(str)
            for text, lang in zip(texts, langs):
                if lang not in want:
                    continue
                if max_per_lang is not None and len(by_lang[lang]) >= max_per_lang:
                    continue
                by_lang[lang].append(text)
        if all(max_per_lang is not None and len(by_lang[l]) >= max_per_lang for l in languages):
            break  # every language already capped

    for lang in languages:
        logger.info("Loaded %d reviews for '%s'", len(by_lang[lang]), lang)
    return by_lang


# --------------------------------------------------------------------------- #
# Tokenisation                                                                  #
# --------------------------------------------------------------------------- #
def tokenize(text: str, lang: str) -> List[str]:
    """Language-aware lowercase tokenisation via wordfreq, with a regex fallback."""
    from wordfreq import tokenize as wf_tokenize

    try:
        return wf_tokenize(text.lower(), lang)
    except Exception:  # language needs an unavailable segmenter (e.g. mecab)
        return _WORD_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# Statistics                                                                    #
# --------------------------------------------------------------------------- #
def _minmax(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a score dict to [0, 1] (constant -> all 1.0)."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def terminology_unigrams(
    unigram_counts: Counter,
    total_tokens: int,
    lang: str,
    min_count: int,
    top_k: int,
    min_general_freq: float,
) -> Dict[str, float]:
    """Rank unigrams by weirdness (domain freq / general freq) -> salience.

    A ``min_general_freq`` floor is essential: pure weirdness rewards *rarity*,
    so without it the top terms become proper nouns (place/brand names) and
    typos, which are rare in general text and thus score highest. Requiring the
    word to exist in the general vocabulary keeps genuine domain vocabulary
    (``reception``, ``breakfast``, ``homestay``) and drops the noise.
    """
    from wordfreq import word_frequency

    log_weirdness: Dict[str, float] = {}
    for word, count in unigram_counts.items():
        if count < min_count or len(word) < 2:
            continue
        f_general = word_frequency(word, lang, minimum=0.0)
        if f_general < min_general_freq:  # unknown -> proper noun / typo / foreign
            continue
        f_domain = count / total_tokens
        log_weirdness[word] = math.log(f_domain / f_general)

    # keep the most domain-specific terms
    top = dict(sorted(log_weirdness.items(), key=lambda kv: kv[1], reverse=True)[:top_k])
    return _minmax(top)


def terminology_bigrams(
    bigram_counts: Counter,
    unigram_counts: Counter,
    total_tokens: int,
    lang: str,
    min_count: int,
    top_k: int,
    min_general_freq: float,
) -> Dict[str, float]:
    """Rank bigrams by PMI (collocation strength inside the corpus) -> salience.

    A bigram is kept only if at least one of its words is a real general word,
    which drops proper-noun collocations (e.g. place names) while keeping
    domain collocations like ``front desk`` / ``swimming pool``.
    """
    from wordfreq import word_frequency

    pmi: Dict[str, float] = {}
    for (w1, w2), count in bigram_counts.items():
        if count < min_count:
            continue
        if (word_frequency(w1, lang, minimum=0.0) < min_general_freq
                and word_frequency(w2, lang, minimum=0.0) < min_general_freq):
            continue  # both words unknown -> proper-noun pair, skip
        p_bigram = count / total_tokens
        p_w1 = unigram_counts[w1] / total_tokens
        p_w2 = unigram_counts[w2] / total_tokens
        if p_w1 <= 0 or p_w2 <= 0:
            continue
        pmi[f"{w1} {w2}"] = math.log(p_bigram / (p_w1 * p_w2))

    top = dict(sorted(pmi.items(), key=lambda kv: kv[1], reverse=True)[:top_k])
    return _minmax(top)


# --------------------------------------------------------------------------- #
# fastText expansion (optional)                                                 #
# --------------------------------------------------------------------------- #
def expand_with_fasttext(
    reviews: List[str],
    seeds: List[str],
    lang: str,
    epochs: int,
    neighbours: int,
    min_similarity: float,
    tmp_dir: str = "/tmp",
) -> Dict[str, float]:
    """Train fastText on the corpus and expand ``seeds`` via nearest neighbours.

    Returns ``{neighbour_word: similarity}`` for neighbours above the similarity
    threshold. Kept separate so the whole step can be skipped with a flag.
    """
    import fasttext

    train_path = os.path.join(tmp_dir, f"ft_{lang}.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        for text in reviews:
            f.write(" ".join(tokenize(text, lang)) + "\n")

    logger.info("Training fastText (skipgram) for '%s' on %d reviews", lang, len(reviews))
    model = fasttext.train_unsupervised(
        train_path, model="skipgram", dim=100, epoch=epochs, minCount=5
    )

    expanded: Dict[str, float] = {}
    for seed in seeds:
        try:
            for sim, word in model.get_nearest_neighbors(seed, k=neighbours):
                word = word.lower()
                if sim >= min_similarity and len(word) >= 2:
                    expanded[word] = max(expanded.get(word, 0.0), float(sim))
        except Exception:
            continue
    os.remove(train_path)
    return expanded


# --------------------------------------------------------------------------- #
# Per-language build                                                            #
# --------------------------------------------------------------------------- #
def build_for_language(
    lang: str, reviews: List[str], args: argparse.Namespace
) -> Dict[str, Dict[str, float]]:
    """Build the four category maps (term -> salience) for one language."""
    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    corpus_vocab: set = set()
    total_tokens = 0

    for text in reviews:
        tokens = tokenize(text, lang)
        total_tokens += len(tokens)
        unigram_counts.update(tokens)
        corpus_vocab.update(tokens)
        bigram_counts.update(zip(tokens, tokens[1:]))

    if total_tokens == 0:
        return {"hotel_terms": {}, "opinions": {}, "negations": {}, "intensifiers": {}}

    # 1. terminology (data-driven)
    uni = terminology_unigrams(
        unigram_counts, total_tokens, lang, args.min_count, args.top_k, args.min_general_freq
    )
    bi = terminology_bigrams(
        bigram_counts, unigram_counts, total_tokens, lang,
        args.min_count, args.top_k_bigrams, args.min_general_freq,
    )
    hotel_terms = {**uni, **bi}

    # 2. opinion (seed ∩ corpus, optionally fastText-expanded)
    opinions = {w: 1.0 for w in _OPINION_SEEDS if w in corpus_vocab}
    if args.expand:
        present_seeds = [w for w in _OPINION_SEEDS if w in corpus_vocab]
        expanded = expand_with_fasttext(
            reviews, present_seeds, lang,
            epochs=args.fasttext_epochs,
            neighbours=args.fasttext_neighbours,
            min_similarity=args.min_similarity,
        )
        for w, sim in expanded.items():
            opinions.setdefault(w, sim)  # keep seed's 1.0 if already present

    # 3. negation / intensifier (closed class; always max salience)
    negations = {w: 1.0 for w in _NEGATIONS if w in corpus_vocab}
    intensifiers = {w: 1.0 for w in _INTENSIFIERS if w in corpus_vocab}

    logger.info(
        "[%s] terms=%d (uni=%d, bi=%d) opinions=%d neg=%d intens=%d",
        lang, len(hotel_terms), len(uni), len(bi), len(opinions), len(negations), len(intensifiers),
    )
    return {
        "hotel_terms": hotel_terms,
        "opinions": opinions,
        "negations": negations,
        "intensifiers": intensifiers,
    }


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a data-driven biased-masking lexicon")
    p.add_argument("--data_dir", default="../data_final/unlabeled_data")
    p.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    p.add_argument("--max_per_lang", type=int, default=200_000)
    p.add_argument("--min_count", type=int, default=20, help="min corpus count to keep a term")
    p.add_argument("--min_general_freq", type=float, default=1e-6,
                   help="drop terms below this wordfreq general frequency (proper nouns/typos)")
    p.add_argument("--top_k", type=int, default=300, help="top unigram terms per language")
    p.add_argument("--top_k_bigrams", type=int, default=100, help="top bigram collocations per language")
    p.add_argument("--expand", action="store_true", help="expand opinion seeds via fastText NN")
    p.add_argument("--fasttext_epochs", type=int, default=5)
    p.add_argument("--fasttext_neighbours", type=int, default=5)
    p.add_argument("--min_similarity", type=float, default=0.5)
    p.add_argument("--out_dir", default="lexicon")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    by_lang = load_reviews(args.data_dir, args.languages, max_per_lang=args.max_per_lang)

    # merged {term: weight} used at masking time (max salience across languages)
    merged: Dict[str, float] = {}

    for lang in args.languages:
        reviews = by_lang.get(lang, [])
        if not reviews:
            logger.warning("No reviews for '%s'; skipping.", lang)
            continue
        try:
            categories = build_for_language(lang, reviews, args)
        except Exception:
            logger.exception("Failed to build lexicon for '%s'; skipping.", lang)
            continue

        # per-language file (with category breakdown + scores)
        with open(os.path.join(args.out_dir, f"lexicon_{lang}.json"), "w", encoding="utf-8") as f:
            json.dump({"language": lang, "categories": categories}, f, ensure_ascii=False, indent=2)

        for cat_scores in categories.values():
            for term, score in cat_scores.items():
                merged[term] = max(merged.get(term, 0.0), float(score))

    merged_path = os.path.join(args.out_dir, "lexicon.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump({"weights": merged}, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d merged terms to %s", len(merged), merged_path)


if __name__ == "__main__":
    main()
