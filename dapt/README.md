# Stage 0 — Multilingual Hotel Domain-Adaptive Pretraining (Hotel-DAPT)

Tiếp tục pretrain `google/mt5-base` bằng objective **denoising (span corruption)**
trên corpus review khách sạn đa ngôn ngữ, tạo ra backbone **`hotel-mt5/`**.

> Chỉ là Domain-Adaptive Pretraining — **không** chứa bất kỳ logic nào về
> ASQP / ACOS / teacher / student / pseudo-label / confidence / EMA / self-training.

---

## Các file làm nhiệm vụ gì

| File | Nhiệm vụ |
|------|----------|
| **`utils.py`** | Nền tảng dùng chung: `Config` (dataclass chứa **toàn bộ** hyperparameter), `set_seed`, `get_device`, `resolve_precision` (auto bf16/fp16/fp32), `setup_logging`, `count_parameters`, và **`LEXICON`** — từ điển đa ngôn ngữ (hotel / opinion / negation / intensifier) cho biased masking. |
| **`masking.py`** | Span corruption kiểu mT5. `SpanCorruption` class + `generate_masked_example()`: chọn span dài 1–5 token, mask ~15%, **tăng xác suất** mask token thuộc lexicon, thay span bằng `<extra_id_i>`, sinh target đúng định dạng mT5. Không có từ khoá → tự động fallback về random. |
| **`dataset.py`** | Nạp corpus từ `data_final/unlabeled_data/hotel_review*_lang.csv` (đọc theo chunk), gom review theo `language`. `HotelReviewDataset` với **temperature sampling** để cân bằng ngôn ngữ; trả về text thô (chưa mask). |
| **`collator.py`** | `DataCollatorForSpanCorruption`: tokenize text thô → gọi span corruption → pad động (`pad_token` cho input, `-100` cho labels) → trả `input_ids / attention_mask / labels`. Masking chạy **on-the-fly mỗi batch**. |
| **`trainer.py`** | `DAPTTrainer`: vòng lặp train (forward → loss → backward → clip → `optimizer.step` → `scheduler.step`), mixed precision, gradient accumulation, validation loop, lưu checkpoint + **resume**, export cuối cùng ra `hotel-mt5/`. |
| **`train_dapt.py`** | Entry point: parse CLI → dựng `Config` → load mT5 → dataset/loader/collator/trainer → train. |
| **`requirements.txt`** | torch, transformers, sentencepiece, protobuf, pandas, numpy, tqdm. |

---

## Flow tổng quát

```
train_dapt.main()
  parse_args ──► Config ──► set_seed / get_device
  │
  ├─► load AutoTokenizer + MT5ForConditionalGeneration ("google/mt5-base")
  ├─► build_datasets(cfg)               # dataset.py: đọc CSV, tách train/val, temperature sampling
  ├─► SpanCorruption(...)               # masking.py  (dùng LEXICON từ utils)
  ├─► DataCollatorForSpanCorruption(...)# collator.py
  ├─► DataLoader(train/val, collate_fn=collator)
  └─► DAPTTrainer(...).train()          # trainer.py
          for epoch:
            for batch:
              model(**batch) → loss (CE do mT5 tự tính)
              backward → (mỗi accum) clip → optimizer.step → scheduler.step
            định kỳ: evaluate() + save_checkpoint()
          → save_pretrained_final() ──► hotel-mt5/
```

**Luồng 1 mẫu:** CSV row → `dataset` (chọn theo ngôn ngữ, trả text thô) →
`collator` (tokenize + span-corrupt + pad) → `model` (tính loss) → backward.
Model **không bao giờ thấy text gốc** — text bị mask ngay trước khi vào model,
và mask sinh mới mỗi batch nên cùng một review qua các epoch bị mask khác nhau.

---

## Cách chạy

```bash
cd dapt
pip install -r requirements.txt

# huấn luyện
python train_dapt.py \
    --data_dir ../data_final/unlabeled_data \
    --num_epochs 3 --batch_size 8 --gradient_accumulation_steps 4 \
    --precision auto \
    --output_dir ../checkpoints/hotel-dapt --final_dir ../hotel-mt5

# resume từ checkpoint gần nhất trong --output_dir
python train_dapt.py --resume ...
```

Tham số đáng chú ý:

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--sampling_temperature` | 2.0 | Cao hơn ⇒ cân bằng ngôn ngữ hơn (nâng ngôn ngữ hiếm). `T=1` = phân phối gốc. |
| `--noise_density` | 0.15 | Tỉ lệ token bị mask (~15%). |
| `--max_span_length` | 5 | Độ dài span tối đa (span dài 1..5). |
| `--lexicon_boost` | 5.0 | Hệ số tăng xác suất mask cho token thuộc lexicon (`1.0` = tắt biasing). |
| `--precision` | auto | `auto`/`bf16`/`fp16`/`fp32`. |
| `--max_seq_length` | 256 | Số token tối đa trước khi corrupt. |

---

## Kết quả

Sau khi train xong, backbone lưu ở **`hotel-mt5/`** gồm `config.json`,
`generation_config.json`, `tokenizer.json`, `spiece.model`, `model.safetensors`.

Load lại:

```python
from transformers import MT5ForConditionalGeneration, AutoTokenizer
model = MT5ForConditionalGeneration.from_pretrained("hotel-mt5")
tokenizer = AutoTokenizer.from_pretrained("hotel-mt5")
```


Ý tưởng cốt lõi: bắt model đoán lại phần bị che, và mỗi lần đoán sai thì điều chỉnh trọng số. Lặp lại hàng triệu lần trên review khách sạn → model dần "thấm" ngôn ngữ và khái niệm của domain này.

1. Cơ chế học: denoising (điền vào chỗ trống)

Với mỗi review, ta che vài span rồi yêu cầu model tái tạo phần bị che:

Gốc  : The room was very clean but the breakfast was disappointing.
Input: The <extra_id_0> was very clean but the <extra_id_1> disappointing.
Target: <extra_id_0> room <extra_id_1> breakfast was <extra_id_2>
                     └─model phải đoán ra─┘

Model là seq2seq (mT5): encoder đọc câu đã che, decoder sinh ra target. Muốn đoán đúng room, breakfast was, model buộc phải hiểu:
- ngữ pháp/ngữ cảnh ("the ___ was" → một danh từ),
- kiến thức domain: sau "clean" thường nói về room; "disappointing" hay đi với breakfast, service...

Đó chính là tín hiệu học.

2. Vòng học thực tế (trong trainer.py)

model(**batch) → loss              # Cross-Entropy: so token model đoán vs token đúng
loss.backward()                    # tính gradient: "sai ở đâu, sửa hướng nào"
optimizer.step()                   # cập nhật trọng số để lần sau đoán đúng hơn
scheduler.step()                   # điều chỉnh learning rate

- Loss cao = đoán sai nhiều → gradient lớn → chỉnh mạnh.
- Loss giảm dần theo thời gian = model đoán ngày càng đúng = đã học được domain.
- Loss do chính MT5ForConditionalGeneration tính (mình không viết loss riêng); vị trí -100 (padding) bị bỏ qua.

3. "Domain-Adaptive" nghĩa là gì

Không train từ số 0. Ta khởi tạo từ trọng số google/mt5-base (đã biết đa ngôn ngữ tổng quát), rồi tiếp tục train trên riêng review khách sạn. Kết quả: model giữ khả năng ngôn ngữ chung nhưng lệch (adapt) về domain khách sạn — quen với từ vựng reception, check-in, phòng, lễ tân, cách người ta khen/chê khách sạn, ở nhiều ngôn ngữ.

4. Hai "chiêu" giúp học đúng trọng tâm

┌─────────────────────────┬──────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│         Cơ chế          │       File       │                                                        Model học được gì thêm                                                        │
├─────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Biased masking          │ masking.py +     │ Che nhiều hơn các từ hotel/opinion/negation/intensifier (room, clean, not, very...) → model bị "ép" học kỹ chính những từ quan trọng │
│ (build_lexicon)         │ utils.py         │  cho phân tích cảm xúc khách sạn, thay vì che đại từ vô nghĩa như "the".                                                             │
├─────────────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Temperature sampling    │ dataset.py       │ Nâng tần suất ngôn ngữ hiếm (T=2.0) → model không chỉ giỏi tiếng Anh/Việt mà học đều các ngôn ngữ khác.                              │
└─────────────────────────┴──────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

Đây là lý do bạn bôi đen build_lexicon: nó quyết định model bị ép học kỹ từ nào. Ví dụ, vì not/không được che thường xuyên, model buộc phải học ngữ cảnh phủ định — rất quan trọng ("not clean" ngược nghĩa "clean").

5. Vì sao che động mỗi batch lại tốt

Mask được sinh mới mỗi lần lấy batch (collator.py), nên cùng một review qua các epoch bị che ở chỗ khác nhau → model thấy nhiều "biến thể" của cùng một câu → học vững hơn, ít học vẹt.

---
Tóm lại: DAPT = liên tục cho model chơi trò "điền vào chỗ trống" trên review khách sạn; mỗi lần điền sai bị phạt (loss) và chỉnh trọng số (backward + optimizer). Nhờ biased masking + temperature sampling, phần bị che tập trung vào từ vựng/cảm xúc/ngôn ngữ quan trọng, nên model hotel-mt5 cuối cùng trở thành backbone hiểu sâu ngôn ngữ domain khách sạn đa ngôn ngữ — sẵn sàng cho các bước fine-tune sau.


1. load_reviews()

Đọc hotel_review*_lang.csv theo chunk, chỉ giữ các ngôn ngữ mục tiêu (en, vi, fr…), mỗi ngôn ngữ tối đa max_per_lang review (giới hạn để chạy nhanh). Trả {lang: [reviews]}.

2. build_for_language() — trái tim

với mỗi review: tokenize(text, lang)
   → đếm unigram_counts, bigram_counts, tổng token, vocab

→ terminology_unigrams()   # weirdness
→ terminology_bigrams()    # PMI
→ opinions  = seed ∩ vocab (+ fastText nếu --expand)
→ negations, intensifiers = seed đóng ∩ vocab

3. tokenize(text, lang)

Dùng wordfreq.tokenize — cắt từ theo từng ngôn ngữ (tiếng Anh khác tiếng Nhật/Trung). Nếu ngôn ngữ thiếu bộ segmenter thì fallback về regex \w+.

4. terminology_unigrams() — weirdness

Đây là phần khoa học nhất:

f_domain  = count(w) / total_tokens          # tần suất trong corpus khách sạn
f_general = word_frequency(w, lang)           # tần suất chung (từ wordfreq)
log_weirdness = log(f_domain / f_general)

- f_general lấy từ wordfreq (kho tần suất từ tổng quát đa ngôn ngữ) — đây là "corpus nền".
- weirdness cao = từ xuất hiện trong review khách sạn nhiều bất thường so với đời thường → chính là thuật ngữ domain.
  - Ví dụ: "reception", "check-in", "phòng" → weirdness cao.
  - "the", "và", "good" → weirdness ~1 (không đặc trưng) → bị loại.
- minimum=1e-8: chặn từ lạ/typo (f_general=0) khỏi weirdness = vô cực.
- Lọc min_count (bỏ nhiễu tần suất thấp) → lấy top_k → chuẩn hoá về [0,1] (_minmax).

5. terminology_bigrams() — PMI

Tìm cụm từ cố định (collocation):
PMI(w1,w2) = log( P(w1 w2) / (P(w1) · P(w2)) )
- PMI cao = 2 từ đi cùng nhau nhiều hơn ngẫu nhiên → cụm có nghĩa: "swimming pool", "front desk", "lễ tân".
- Cũng lọc min_count → top_k → chuẩn hoá [0,1].

6. expand_with_fasttext() (khi --expand)

- Ghi review ra file tạm → train fastText skipgram ngay trên corpus khách sạn.
- Với mỗi opinion seed (clean, friendly…) → lấy k từ gần nhất trong không gian embedding (get_nearest_neighbors) → thêm những từ có độ tương đồng ≥ min_similarity.
- Nhờ vậy lexicon bám sát cách dùng từ thực tế trong domain (vd clean → spotless, tidy).

7. Trọng số liên tục (điểm mới so với bản cũ)

Mỗi term có salience ∈ [0,1]. lexicon.json = {term: salience}. Trong masking.py:
weight(token) = 1 + lexicon_boost * salience
→ từ càng đặc trưng domain (salience cao) càng dễ bị mask — mượt hơn kiểu bật/tắt phẳng cũ (mọi từ boost = 5.0 như nhau).
- negation/intensifier gán salience = 1.0 (luôn quan trọng, boost tối đa).

8. masking.py khớp thế nào

_token_weights ghép sub-word thành từ, rồi so cả unigram (room) lẫn bigram (swimming pool — 2 từ liên tiếp) với lexicon; term nào khớp thì nâng weight các token nó phủ.

---
Tóm tắt một câu

build_lexicon.py = rút thuật ngữ domain từ chính dữ liệu bằng thống kê (weirdness so với wordfreq + PMI cho cụm), mở rộng opinion bằng fastText, và gán mỗi từ một điểm salience liên tục để masking ưu tiên mask đúng những từ quan trọng nhất cho phân tích cảm xúc khách sạn.

Image multilabsa-lexicon đang build xong sẽ chạy thử trên en+vi để xác nhận output hợp lý. Tôi sẽ báo kết quả.