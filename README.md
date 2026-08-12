# MultiLABSA — Inference (Dual-Teacher ASQP)

Trích bộ tứ **(aspect, opinion, category, sentiment)** từ review khách sạn đa ngôn ngữ
bằng hai teacher: **T_G** (`hotel-mt5-asqp`, mô hình sinh) và **T_E** (XLM-R, mô hình trích),
hợp nhất qua **Confidence Fusion**. Có sẵn **UI (Gradio)** và **CLI**, đóng gói bằng **Docker**.

---

## 1. Yêu cầu (máy mới cần có)

| Thành phần | Ghi chú |
|---|---|
| **Docker Desktop** | Cách chạy khuyến nghị (không cần cài Python/torch thủ công) |
| **Git** | Để clone repo |
| **2 model** (tải riêng, không có trong git) | `hotel-mt5-asqp/` và `checkpoints/dual-teacher/` — xem Mục 3 |
| **Internet** (tuỳ chọn) | Chỉ cần khi bật T_E / Multi-view (tải model phụ vào `hf-cache/`) |
| GPU (tuỳ chọn) | Nhanh hơn ~10×; xem cuối `Dockerfile.infer` |

---

## 2. Cấu trúc thư mục để chạy được inference

```
MultiLABSA/
├── app.py                     ✅ UI Gradio
├── infer.py                   ✅ CLI inference
├── Dockerfile.infer           ✅ image inference (torch + transformers + gradio)
├── docker-compose.yml         ✅ chạy UI 1 lệnh
├── requirements.txt           ✅
├── teacher/                   ✅ generative/extractive/multiview/disagreement/fusion/translator
├── models/                    ✅ mt5.py, xlmr.py, heads.py
├── utils/                     ✅ schema, label_maps, data, text_alignment, common
│
├── hotel-mt5-asqp/            ⬇️  PHẢI TẢI THÊM — T_G (mô hình ASQP hoàn chỉnh)
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── checkpoints/
│   └── dual-teacher/          ⬇️  PHẢI TẢI THÊM — T_E
│       ├── extractive_teacher.pt
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
└── hf-cache/                  🔄 TỰ SINH khi chạy T_E/Multi-view (xem Mục 6)
```

Chú thích: ✅ có sẵn trong repo · ⬇️ phải tải thêm (đã `.gitignore`) · 🔄 tự sinh khi chạy.

---

## 3. Setup ban đầu trên máy mới (từng bước)

```bash
# 1) Clone repo
git clone <URL-repo> MultiLABSA
cd MultiLABSA

# 2) Cài Docker Desktop (nếu chưa có): https://www.docker.com/products/docker-desktop

# 3) Tải 2 model và đặt đúng chỗ (Mục 2):
#    - hotel-mt5-asqp/  -> đặt ở thư mục gốc repo
#    - checkpoints/dual-teacher/extractive_teacher.pt (+ tokenizer.json, tokenizer_config.json)
#    LƯU Ý: nếu tải từ Kaggle làm file bị đổi đuôi .json -> .txt, hãy đổi lại thành .json.
#           File .pt là archive PyTorch — KHÔNG tự giải nén; torch.load đọc trực tiếp.

# 4) Build + chạy UI
docker compose up -d --build          # -> http://localhost:7860

# 5) (tuỳ chọn) test nhanh 1 câu bằng CLI, chỉ T_G, không cần internet
docker run --rm \
  -v "$(pwd)/hotel-mt5-asqp:/app/hotel-mt5-asqp" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  multilabsa-infer \
  python infer.py --text "The room was clean but breakfast was bad." --generative_only
```

> Windows PowerShell: thay `$(pwd)` bằng `${PWD}` và dùng dấu `` ` `` để xuống dòng thay `\`.

Kiểm tra file model không hỏng (không cần torch):
```bash
docker run --rm -v "$(pwd)/hotel-mt5-asqp:/app/hotel-mt5-asqp" \
  -v "$(pwd)/checkpoints:/app/checkpoints" multilabsa-infer \
  python -c "import json,struct,os; \
p='hotel-mt5-asqp/model.safetensors'; f=open(p,'rb'); n=struct.unpack('<Q',f.read(8))[0]; \
print('safetensors OK,', len(json.loads(f.read(n))), 'tensors')"
```

---

## 4. Chạy inference

### UI (Gradio)
```bash
docker compose up -d --build      # bật  -> http://localhost:7860
docker compose logs -f            # xem log
docker compose down               # tắt
```
Trong UI: nhập 1 câu review (gõ dấu tiếng Việt thoải mái) → chọn ngôn ngữ → tick/bỏ tick
**Extractive Teacher** và **Multi-view** → bấm *Phân tích*.

### CLI (một hoặc nhiều câu)
```bash
# Tiếng Anh, nhanh nhất (chỉ T_G, offline)
docker run --rm -v "$(pwd)/hotel-mt5-asqp:/app/hotel-mt5-asqp" \
  -v "$(pwd)/checkpoints:/app/checkpoints" multilabsa-infer \
  python infer.py --text "The staff were friendly." --generative_only

# Tiếng Việt: DÙNG FILE UTF-8 (tránh lỗi dấu khi truyền --text qua PowerShell)
echo "Phòng rất sạch nhưng bữa sáng thì tệ." > reviews.txt
docker run --rm -v "$(pwd)/hotel-mt5-asqp:/app/hotel-mt5-asqp" \
  -v "$(pwd)/checkpoints:/app/checkpoints" -v "$(pwd):/data" multilabsa-infer \
  python infer.py --input_file /data/reviews.txt --lang vi --generative_only --out /data/pred.json
```

### Không dùng Docker
```bash
pip install torch transformers sentencepiece protobuf gradio
python app.py                                   # UI
python infer.py --text "..." --generative_only  # CLI
```

---

## 5. Các chế độ (checkbox trong UI / flag CLI)

| Chế độ | Cần gì | Tốc độ (CPU) |
|---|---|---|
| **Chỉ T_G** (`--generative_only`) | Chỉ `hotel-mt5-asqp/`, offline | ~2.6 s/câu |
| **+ T_E** (mặc định) | Thêm `xlm-roberta-base` (tải vào `hf-cache`) | ~3 s/câu |
| **+ Multi-view** (`--multiview`) | Thêm NLLB-200 (tải vào `hf-cache`) | ~15–40 s/câu |

**Định dạng mỗi quad trong output** (MERA-XQUAD §4.4–4.8):

```json
{
  "aspect": "phòng", "opinion": "sạch", "category": "FACILITY", "sentiment": "positive",
  "conf_g": 0.94, "conf_e": 0.80, "agreement": 1.0, "final_score": 0.90,
  "reliability": { "r_AT": .., "r_AC": .., "r_OT": .., "r_SP": .., "r_rel": .., "r_quad": .. },
  "route": "full",
  "sources": ["generative", "extractive"]
}
```

- **QUAD matching**: ghép quad T_G↔T_E bằng **Hungarian matching** (gán tối ưu toàn cục, `scipy`; có fallback thuần Python) thay vì greedy.
- **`reliability`**: reliability theo từng element (`r_AT/r_AC/r_OT/r_SP`), theo quan hệ (`r_rel`) và tổng hợp (`r_quad` = trung bình nhân).
- **`route`**: `full` / `partial` / `verifier` / `consistency` / `deferred` — quyết định cách dùng quad trong self-training (quad không được kiểm chứng chéo → `deferred`).

---

## 6. `hf-cache/` được tải như thế nào

`hf-cache/` **không có sẵn trong repo** và **không cần chuẩn bị trước** — nó là **cache
của HuggingFace Hub, tự động tải về khi chạy lần đầu**.

### Cơ chế
1. `Dockerfile.infer` đặt biến môi trường **`HF_HOME=/hf-cache`**, và
   `docker-compose.yml` mount **`./hf-cache:/hf-cache`** (host ↔ container).
2. Khi mã gọi `from_pretrained("<tên model trên Hub>")`, thư viện `transformers` /
   `huggingface_hub` **tải model từ Hub về `HF_HOME`** (tức `/hf-cache` trong container
   = `./hf-cache` trên máy) rồi dùng lại từ đó.
3. Nhờ volume mount, cache **được giữ lại giữa các lần chạy** → chỉ tải **một lần**;
   các lần sau chạy được **kể cả khi offline**.

### Những gì được tải vào `hf-cache/`
| Model (HuggingFace Hub) | Kích thước | Tải khi nào |
|---|---|---|
| `xlm-roberta-base` | ~1.1 GB | Khi **bật Extractive Teacher (T_E)** |
| `facebook/nllb-200-distilled-600M` | ~2.5 GB | Khi **bật Multi-view** |

> **T_G (`hotel-mt5-asqp`) KHÔNG vào `hf-cache`** vì nạp từ **folder local** đã mount.
> Do đó `--generative_only` **không cần internet** và **không tạo `hf-cache`**.
> Tổng `hf-cache/` thực tế ~5–6 GB do có thêm `blobs/`, `snapshots/`, `refs/`.

### Lưu ý
- **Lần đầu cần internet** để tải → lần chạy T_E/Multi-view đầu tiên chậm.
- Tải nhanh hơn / tránh rate-limit: đặt token `-e HF_TOKEN=hf_xxx`.
- **Pre-download (làm nóng cache)** trước khi demo:
  ```bash
  docker run --rm -v "$(pwd)/hf-cache:/hf-cache" multilabsa-infer python -c \
  "from transformers import AutoModel, AutoModelForSeq2SeqLM; \
   AutoModel.from_pretrained('xlm-roberta-base'); \
   AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')"
  ```
- **Chạy offline hoàn toàn:** dùng `--generative_only`, hoặc trỏ backbone về folder local:
  `python infer.py --extractive_backbone /path/xlm-roberta-base ...`
- `hf-cache/` đã được **`.gitignore`** — không đẩy lên GitHub.

### Biến môi trường
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `HF_HOME` | `/hf-cache` | Thư mục cache HuggingFace |
| `HF_TOKEN` | (trống) | Token Hub để tải nhanh hơn |
| `GEN_MODEL` | `hotel-mt5-asqp` | Đường dẫn T_G |
| `EXT_CKPT` | `checkpoints/dual-teacher/extractive_teacher.pt` | Đường dẫn T_E |
| `EXT_BACKBONE` | `xlm-roberta-base` | Backbone T_E (đổi sang path local để offline) |

---

## 7. Sự cố thường gặp

| Hiện tượng | Nguyên nhân & cách xử lý |
|---|---|
| `quads` rỗng khi chỉ T_G | Ngưỡng fusion 0.5 loại quad không được T_E kiểm chứng. Hạ ngưỡng: `--final_score_threshold 0.2` |
| Tiếng Việt qua `--text` ra rỗng/hỏng | PowerShell làm hỏng dấu ở command-line → **dùng `--input_file` (UTF-8)** hoặc UI |
| Lần đầu Multi-view rất chậm | Đang tải NLLB ~2.5 GB vào `hf-cache` (cần internet) |
| Không load được `hotel-mt5-asqp` | Thiếu `config.json` (bị đổi thành `.txt`?) → đổi lại `.json` |
| Cảnh báo `tie_word_embeddings` | Vô hại, model vẫn cho kết quả đúng |

---

## 8. Thời gian suy luận (CPU, tham khảo)

| Cấu hình | Thời gian/câu |
|---|---|
| Chỉ T_G | ~2.6 s |
| T_G + T_E | ~3 s |
| + Multi-view | ~15–40 s (chưa kể lần đầu tải NLLB) |

GPU nhanh hơn ~10×: cài torch CUDA trong `Dockerfile.infer` + chạy với `--gpus all`
(và bỏ comment khối `deploy` trong `docker-compose.yml`).

---

## 9. Thí nghiệm & bảng so sánh (MERA-XQUAD đầy đủ)

Các bước nghiên cứu còn lại (student self-training, baselines, đánh giá §6, ablation §7)
nằm trong khung thí nghiệm riêng, mỗi bước một folder + notebook Kaggle:

| Folder | Nội dung | Notebook |
|---|---|---|
| `experiments/common/` | metrics (§6.2), bảng so sánh (§6.3), significance (§7.3) — **đã unit-test** | — |
| `student/` | Hybrid student + implicit heads + multi-objective loss + EMA/curriculum (§4.7/4.9/4.10) | `student-kaggle.ipynb` |
| `baselines/` | M1 zero-shot, M3 translate-test, … (§3) | `baselines-kaggle.ipynb` |
| `evaluation/` | chấm điểm → `results.json` → Table 1–8 (§6), ablation (§7) | `evaluation-kaggle.ipynb` |

**Cách làm chi tiết:** xem `experiments/README.md`. Ý tưởng: mỗi lần chạy (baseline /
student / ablation) ghi một `results/<case_id>.json` cùng schema → `evaluation/make_tables.py`
gom cả thư mục thành các bảng so sánh (mean±std theo seed) — thêm trường hợp chỉ cần
thả thêm một file json, không sửa code.
