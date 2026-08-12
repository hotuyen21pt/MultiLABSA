"""Run an inference baseline over reviews -> predictions json (MultiLABSA.docx §3).

The predictions json is the exact input ``evaluation/run_eval.py`` expects, so a
baseline becomes a comparison-table row with:

    python -m baselines.run_baseline --baseline zero_shot \
        --input data_final/labeled_data/hamos26/test.json --out preds/m1_zeroshot.json
    python -m evaluation.run_eval --predictions preds/m1_zeroshot.json \
        --method M1_zeroshot --track zero_shot --split test
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

from teacher.generative_teacher import GenerativeTeacher
from utils.common import get_device, setup_logging

logger = setup_logging()


def load_reviews(path: str, text_column: str, lang_column: str):
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        texts = [(r.get("review") or "").strip() for r in rows]
        langs = [r.get(lang_column, "en") for r in rows]
    else:
        import pandas as pd
        df = pd.read_csv(path).dropna(subset=[text_column])
        texts = df[text_column].astype(str).tolist()
        langs = df[lang_column].tolist() if lang_column in df.columns else [None] * len(texts)
    return texts, langs


def main():
    p = argparse.ArgumentParser(description="Run an inference baseline")
    p.add_argument("--baseline", choices=["zero_shot", "translate_test"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--generative_model", default="hotel-mt5-asqp")
    p.add_argument("--translator_model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--text_column", default="review")
    p.add_argument("--lang_column", default="language")
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    device = get_device()
    teacher = GenerativeTeacher(args.generative_model, device=device)

    from baselines.transfer import ZeroShotBaseline, TranslateTestBaseline
    if args.baseline == "zero_shot":
        model = ZeroShotBaseline(teacher)
    else:
        from teacher.translator import NLLBTranslator
        model = TranslateTestBaseline(teacher, NLLBTranslator(model_name=args.translator_model, device=device))

    texts, langs = load_reviews(args.input, args.text_column, args.lang_column)
    logger.info("Baseline %s over %d reviews", args.baseline, len(texts))

    predictions = []
    for i in range(0, len(texts), args.batch_size):
        bt, bl = texts[i:i + args.batch_size], langs[i:i + args.batch_size]
        for text, lang, quads in zip(bt, bl, model.predict(bt, bl)):
            predictions.append({"review": text, "language": lang, "quads": quads})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d predictions -> %s", len(predictions), args.out)


if __name__ == "__main__":
    main()
