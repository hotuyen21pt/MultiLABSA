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
