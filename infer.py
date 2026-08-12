"""Inference cho Dual-Teacher ASQP (T_G = hotel-mt5-asqp, T_E = extractive_teacher.pt).

Chạy hai teacher trên câu review bất kỳ -> Architectural Disagreement -> Confidence
Fusion -> in ra các quad (aspect, opinion, category, sentiment) kèm điểm tin cậy.

Bố cục thư mục mong đợi (đặt cạnh repo, xem README phần "Đặt model"):

    MultiLABSA/
    ├── hotel-mt5-asqp/                     # T_G: model ASQP hoàn chỉnh (HF)
    │   ├── config.json  generation_config.json  model.safetensors
    │   └── tokenizer.json  tokenizer_config.json  (spiece.model nếu có)
    ├── checkpoints/
    │   └── dual-teacher/
    │       ├── extractive_teacher.pt       # T_E: state_dict (backbone + 3 head)
    │       └── tokenizer.json ...          # tokenizer XLM-R đã lưu (tuỳ chọn)
    └── infer.py

Ví dụ
-----
    # Một câu, có bật multi-view (cần NLLB + internet/dataset)
    python infer.py --text "Phòng rất sạch nhưng bữa sáng thì tệ." --lang vi --multiview

    # Nhiều câu từ file (mỗi dòng 1 review) hoặc CSV có cột review[,language]
    python infer.py --input_file reviews.txt --out predictions.json

    # Chỉ dùng Generative Teacher (bỏ qua T_E), nhanh & không cần xlm-roberta-base
    python infer.py --text "The room was spotless." --generative_only
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import torch

from teacher.confidence_fusion import FusionWeights, fuse
from teacher.disagreement import DisagreementWeights, compute_agreement
from teacher.extractive_teacher import ExtractiveTeacher
from teacher.generative_teacher import GenerativeTeacher
from teacher.multiview import MultiViewGenerativeTeacher, MultiViewWeights
from teacher.translator import NLLBTranslator
from utils.common import get_device, setup_logging
from utils.schema import MergedPrediction

logger = setup_logging()


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual-Teacher ASQP inference")

    # input
    p.add_argument("--text", default=None, help="Một review để suy luận nhanh")
    p.add_argument("--input_file", default=None,
                   help=".txt (mỗi dòng 1 review) hoặc .csv có cột review[,language]")
    p.add_argument("--text_column", default="review")
    p.add_argument("--lang_column", default="language")
    p.add_argument("--lang", default=None, help="Mã ngôn ngữ fastText cho --text (vd: vi, en, de)")
    p.add_argument("--out", default=None, help="Ghi kết quả ra file JSON (mặc định: chỉ in)")

    # generative teacher (T_G)
    p.add_argument("--generative_model", default="hotel-mt5-asqp",
                   help="Thư mục model ASQP hoàn chỉnh (mặc định: hotel-mt5-asqp/)")
    p.add_argument("--gen_max_source_length", type=int, default=256)
    p.add_argument("--gen_max_target_length", type=int, default=160)
    p.add_argument("--gen_num_beams", type=int, default=4)
    p.add_argument("--generative_only", action="store_true",
                   help="Chỉ dùng T_G; bỏ qua T_E (không cần xlm-roberta-base)")

    # multi-view (chỉ áp cho T_G)
    p.add_argument("--multiview", action="store_true",
                   help="Chạy T_G trên 3 view (native / dịch-EN / back-translation) rồi vote")
    p.add_argument("--translator_model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--multiview_min_agreeing_views", type=int, default=2)
    p.add_argument("--multiview_confidence_boost", type=float, default=0.15)
    p.add_argument("--multiview_pivot_lang", default="fra_Latn")

    # extractive teacher (T_E)
    p.add_argument("--extractive_checkpoint", default="checkpoints/dual-teacher/extractive_teacher.pt",
                   help="File state_dict của T_E (.pt)")
    p.add_argument("--extractive_backbone", default="xlm-roberta-base",
                   help="Backbone HF cho T_E; đổi sang đường dẫn local nếu chạy offline")
    p.add_argument("--extractive_tokenizer", default=None,
                   help="Tokenizer cho T_E (mặc định = --extractive_backbone; có thể trỏ vào "
                        "checkpoints/dual-teacher nếu đã lưu tokenizer ở đó)")
    p.add_argument("--relation_proj_size", type=int, default=256)
    p.add_argument("--classifier_proj_size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_seq_length", type=int, default=160)
    p.add_argument("--relation_threshold", type=float, default=0.5)

    # disagreement + fusion
    p.add_argument("--overlap_aspect_weight", type=float, default=0.4)
    p.add_argument("--overlap_opinion_weight", type=float, default=0.3)
    p.add_argument("--category_match_weight", type=float, default=0.15)
    p.add_argument("--sentiment_match_weight", type=float, default=0.15)
    p.add_argument("--match_threshold", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--final_score_threshold", type=float, default=0.2)

    p.add_argument("--inference_batch_size", type=int, default=16)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Input loading                                                                #
# --------------------------------------------------------------------------- #
def load_inputs(args: argparse.Namespace) -> tuple[List[str], List[Optional[str]]]:
    if args.text is not None:
        return [args.text], [args.lang]

    if args.input_file is None:
        raise SystemExit("Cần --text hoặc --input_file")

    if args.input_file.lower().endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(args.input_file).dropna(subset=[args.text_column])
        texts = df[args.text_column].astype(str).tolist()
        if args.lang_column in df.columns:
            langs = df[args.lang_column].tolist()
        else:
            langs = [None] * len(texts)
        return texts, langs

    with open(args.input_file, encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts, [args.lang] * len(texts)


# --------------------------------------------------------------------------- #
# Model loading                                                                #
# --------------------------------------------------------------------------- #
def load_extractive_teacher(args: argparse.Namespace, device: torch.device):
    from transformers import AutoTokenizer

    if not os.path.exists(args.extractive_checkpoint):
        raise FileNotFoundError(
            f"Không thấy checkpoint T_E: {args.extractive_checkpoint}. "
            "Kiểm tra lại đường dẫn (đã giải nén dual-teacher-output.zip chưa?)."
        )
    tokenizer_src = args.extractive_tokenizer or args.extractive_backbone
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

    model = ExtractiveTeacher(
        backbone_name=args.extractive_backbone,
        relation_proj_size=args.relation_proj_size,
        classifier_proj_size=args.classifier_proj_size,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
    ).to(device)
    state = torch.load(args.extractive_checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("Đã nạp Extractive Teacher từ %s", args.extractive_checkpoint)
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    device = get_device()
    logger.info("Device: %s", device)

    texts, langs = load_inputs(args)
    logger.info("Suy luận trên %d review", len(texts))

    # ---- Generative Teacher (T_G) -------------------------------------------
    generative_teacher = GenerativeTeacher(
        model_name_or_path=args.generative_model,
        device=device,
        max_source_length=args.gen_max_source_length,
        max_new_tokens=args.gen_max_target_length,
        num_beams=args.gen_num_beams,
    )
    logger.info("Đã nạp Generative Teacher từ %s", args.generative_model)

    multiview_teacher: Optional[MultiViewGenerativeTeacher] = None
    if args.multiview:
        translator = NLLBTranslator(model_name=args.translator_model, device=device)
        multiview_teacher = MultiViewGenerativeTeacher(
            generative_teacher, translator,
            weights=MultiViewWeights(
                min_agreeing_views=args.multiview_min_agreeing_views,
                confidence_boost_per_view=args.multiview_confidence_boost,
                pivot_lang_for_english=args.multiview_pivot_lang,
            ),
        )
        logger.info("Bật multi-view (translator: %s)", args.translator_model)

    # ---- Extractive Teacher (T_E) -------------------------------------------
    extractive_teacher = tokenizer = None
    if not args.generative_only:
        extractive_teacher, tokenizer = load_extractive_teacher(args, device)

    disagreement_weights = DisagreementWeights(
        aspect_overlap=args.overlap_aspect_weight,
        opinion_overlap=args.overlap_opinion_weight,
        category_match=args.category_match_weight,
        sentiment_match=args.sentiment_match_weight,
        match_threshold=args.match_threshold,
    )
    fusion_weights = FusionWeights(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma, threshold=args.final_score_threshold
    )

    # ---- Loop over batches ---------------------------------------------------
    results: List[dict] = []
    bs = args.inference_batch_size
    for start in range(0, len(texts), bs):
        batch_texts = texts[start : start + bs]
        batch_langs = langs[start : start + bs]

        if multiview_teacher is not None:
            gen_predictions = multiview_teacher.predict(batch_texts, batch_langs)
        else:
            gen_predictions = generative_teacher.predict(batch_texts)

        if extractive_teacher is not None:
            ext_predictions = extractive_teacher.predict(
                batch_texts, tokenizer, device, relation_threshold=args.relation_threshold
            )
        else:
            ext_predictions = [[] for _ in batch_texts]

        for text, gen_quads, ext_quads in zip(batch_texts, gen_predictions, ext_predictions):
            if args.generative_only:
                # Không có T_E: đưa thẳng quad của T_G qua fusion (conf_e = 0, agreement = 0)
                merged = [
                    MergedPrediction(
                        aspect=q.aspect, opinion=q.opinion, category=q.category, sentiment=q.sentiment,
                        conf_g=q.confidence, conf_e=0.0, agreement=0.0, sources=["generative"],
                    )
                    for q in gen_quads
                ]
            else:
                merged = compute_agreement(gen_quads, ext_quads, text, disagreement_weights)
            fused = fuse(merged, fusion_weights)
            results.append({"review": text, "quads": fused})

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Output --------------------------------------------------------------
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Đã ghi %d kết quả -> %s", len(results), args.out)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
