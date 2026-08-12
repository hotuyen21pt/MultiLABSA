"""UI đơn giản (Gradio) cho Dual-Teacher ASQP: nhập 1 câu review -> ra các quad.

Chạy:
    pip install gradio
    python app.py                 # mở http://127.0.0.1:7860
    # trên Kaggle/Colab: app.launch(share=True) đã bật sẵn qua env SHARE=1

Đường dẫn model lấy mặc định giống infer.py, có thể override bằng biến môi trường:
    GEN_MODEL   (mặc định: hotel-mt5-asqp)
    EXT_CKPT    (mặc định: checkpoints/dual-teacher/extractive_teacher.pt)
    EXT_BACKBONE(mặc định: xlm-roberta-base)
    EXT_TOKENIZER (mặc định: = thư mục chứa EXT_CKPT nếu có tokenizer, else EXT_BACKBONE)
"""

from __future__ import annotations

import os
from typing import List, Optional

import gradio as gr
import torch

from teacher.confidence_fusion import FusionWeights, fuse
from teacher.disagreement import DisagreementWeights, compute_agreement
from teacher.extractive_teacher import ExtractiveTeacher
from teacher.generative_teacher import GenerativeTeacher
from teacher.multiview import MultiViewGenerativeTeacher, MultiViewWeights
from teacher.translator import FASTTEXT_TO_NLLB, NLLBTranslator
from utils.common import get_device
from utils.schema import MergedPrediction

# --------------------------------------------------------------------------- #
# Cấu hình (override bằng env var)                                             #
# --------------------------------------------------------------------------- #
GEN_MODEL = os.environ.get("GEN_MODEL", "hotel-mt5-asqp")
EXT_CKPT = os.environ.get("EXT_CKPT", "checkpoints/dual-teacher/extractive_teacher.pt")
EXT_BACKBONE = os.environ.get("EXT_BACKBONE", "xlm-roberta-base")
EXT_TOKENIZER = os.environ.get("EXT_TOKENIZER") or (
    os.path.dirname(EXT_CKPT) if os.path.isdir(os.path.dirname(EXT_CKPT)) else EXT_BACKBONE
)
TRANSLATOR_MODEL = os.environ.get("TRANSLATOR_MODEL", "facebook/nllb-200-distilled-600M")

DEVICE = get_device()
LANG_CHOICES = ["auto"] + sorted(FASTTEXT_TO_NLLB.keys())  # en, vi, fr, de, ...

_DISAGREEMENT = DisagreementWeights()
# threshold=0.2 (thay vì 0.5) để hợp với inference: quad chỉ-T_G có
# final_score = 0.4*conf_g <= 0.4, ngưỡng 0.5 sẽ loại sạch mọi ngôn ngữ.
_FUSION = FusionWeights(threshold=0.5)


# --------------------------------------------------------------------------- #
# Lazy singletons (nạp model 1 lần)                                            #
# --------------------------------------------------------------------------- #
_gen = None
_ext = None
_ext_tok = None
_mv = None


def get_generative():
    global _gen
    if _gen is None:
        _gen = GenerativeTeacher(model_name_or_path=GEN_MODEL, device=DEVICE)
    return _gen


def get_extractive():
    """Trả về (model, tokenizer) hoặc (None, None) nếu chưa có checkpoint."""
    global _ext, _ext_tok
    if _ext is not None:
        return _ext, _ext_tok
    if not os.path.exists(EXT_CKPT):
        return None, None
    from transformers import AutoTokenizer

    _ext_tok = AutoTokenizer.from_pretrained(EXT_TOKENIZER)
    model = ExtractiveTeacher(backbone_name=EXT_BACKBONE).to(DEVICE)
    model.load_state_dict(torch.load(EXT_CKPT, map_location=DEVICE))
    model.eval()
    _ext = model
    return _ext, _ext_tok


def get_multiview():
    global _mv
    if _mv is None:
        translator = NLLBTranslator(model_name=TRANSLATOR_MODEL, device=DEVICE)
        _mv = MultiViewGenerativeTeacher(get_generative(), translator, weights=MultiViewWeights())
    return _mv


# --------------------------------------------------------------------------- #
# Inference cho 1 câu                                                          #
# --------------------------------------------------------------------------- #
def analyze(review: str, lang: str, use_extractive: bool, use_multiview: bool):
    review = (review or "").strip()
    if not review:
        return [], {"error": "Vui lòng nhập một câu review."}

    lang_code: Optional[str] = None if lang == "auto" else lang

    # ---- Generative (có/không multi-view) --------------------------------
    if use_multiview:
        gen_quads = get_multiview().predict([review], [lang_code])[0]
    else:
        gen_quads = get_generative().predict([review])[0]

    # ---- Extractive (tuỳ chọn) -------------------------------------------
    ext_model, ext_tok = (get_extractive() if use_extractive else (None, None))
    if ext_model is not None:
        ext_quads = ext_model.predict([review], ext_tok, DEVICE)[0]
        merged = compute_agreement(gen_quads, ext_quads, review, _DISAGREEMENT)
    else:
        merged = [
            MergedPrediction(
                aspect=q.aspect, opinion=q.opinion, category=q.category, sentiment=q.sentiment,
                conf_g=q.confidence, conf_e=0.0, agreement=0.0, sources=["generative"],
            )
            for q in gen_quads
        ]

    fused = fuse(merged, _FUSION)

    # Bảng hiển thị + JSON đầy đủ
    rows = [
        [q["aspect"], q["opinion"], q["category"], q["sentiment"],
         q.get("final_score"), ", ".join(q.get("sources", []))]
        for q in fused
    ]
    return rows, {"review": review, "quads": fused}


# --------------------------------------------------------------------------- #
# Giao diện                                                                    #
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MultiLABSA — ASQP Demo") as demo:
        gr.Markdown(
            "# 🏨 MultiLABSA — Phân tích cảm xúc theo khía cạnh (ASQP)\n"
            "Nhập **một câu review khách sạn** (bất kỳ ngôn ngữ nào) → trích các bộ tứ "
            "**(aspect, opinion, category, sentiment)**."
        )
        with gr.Row():
            with gr.Column(scale=3):
                review = gr.Textbox(
                    label="Câu review", lines=3,
                    placeholder="Ví dụ: Phòng rất sạch nhưng bữa sáng thì tệ.",
                )
            with gr.Column(scale=1):
                lang = gr.Dropdown(LANG_CHOICES, value="auto", label="Ngôn ngữ")
                use_ext = gr.Checkbox(value=True, label="Dùng Extractive Teacher (T_E)")
                use_mv = gr.Checkbox(value=False, label="Multi-view (cần NLLB)")
        btn = gr.Button("Phân tích", variant="primary")

        table = gr.Dataframe(
            headers=["aspect", "opinion", "category", "sentiment", "final_score", "sources"],
            label="Kết quả quad", wrap=True,
        )
        raw = gr.JSON(label="JSON đầy đủ (conf_g / conf_e / agreement / final_score)")

        btn.click(analyze, inputs=[review, lang, use_ext, use_mv], outputs=[table, raw])
        review.submit(analyze, inputs=[review, lang, use_ext, use_mv], outputs=[table, raw])

        gr.Examples(
            examples=[
                ["Phòng rất sạch nhưng bữa sáng thì tệ.", "vi", True, False],
                ["The staff were friendly but the room was overpriced.", "en", True, False],
                ["Das Zimmer war ruhig und das Personal sehr hilfsbereit.", "de", True, False],
            ],
            inputs=[review, lang, use_ext, use_mv],
        )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        share=os.environ.get("SHARE", "0") == "1",
    )
