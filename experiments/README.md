# MERA-XQUAD — Khung thí nghiệm & bảng so sánh (MultiLABSA.docx §3–§7)

Phần này hiện thực **các bước còn lại** của MERA-XQUAD và một khung để **so sánh
mọi trường hợp** (baseline / mô hình đầy đủ / ablation) thành các bảng §6.3.
Mỗi bước là một **folder độc lập** kèm **notebook Kaggle** riêng.

---

## 1. Ánh xạ tới thiết kế (§4) — cái gì nằm ở đâu

| Thành phần §4 | Trạng thái | File |
|---|---|---|
| A. Multilingual Hotel-DAPT | ✅ (đã có) | `dapt/` |
| B. Dual teacher (T_G, T_E) | ✅ (đã có) | `teacher/` |
| C. Multi-view | ✅ (đã có) | `teacher/multiview.py` |
| D. QUAD matching (Hungarian) | ✅ | `teacher/disagreement.py` |
| E. Element reliability `r_e` | ✅ | `teacher/reliability.py` |
| F. Relation reliability `r_quad` | ✅ | `teacher/reliability.py` |
| G. **Implicit AT/OT (NULL heads)** | ✅ **mới** | `student/hybrid_student.py` |
| H. Confidence–stability routing | ✅ | `teacher/routing.py` |
| I. **Hybrid student + multi-objective loss** | ✅ **mới** | `student/hybrid_student.py`, `student/losses.py` |
| J. **EMA + dynamic curriculum + self-training** | ✅ **mới** | `student/self_training.py`, `student/train_student.py` |

---

## 2. Cấu trúc folder

```
experiments/
  common/            # hạ tầng CHUNG (bắt buộc dùng chung để so sánh công bằng §6.4)
    metrics.py       # §6.2: element/structure F1, implicit, macro/worst-lang, calibration
    results_schema.py# chuẩn results.json {method,track,language,seed,metrics,ablation}
    tables.py        # §6.3: gom results/*.json -> Table 1–8 (mean±std)
    significance.py  # §7.3: paired bootstrap, 95% CI
    data.py          # loader gold (không cần torch), group theo ngôn ngữ
    ontology.py      # 6 category + 29 subcategory (§2)
  configs/           # mỗi CASE = 1 file json (method/track/args)
student/             # G + I + J  (+ notebook)
baselines/           # §3  (+ notebook)
evaluation/          # §6 run_eval + make_tables + §7 ablation  (+ notebook)
```

---

## 3. Cơ chế bảng so sánh (điểm cốt lõi)

**Một schema duy nhất nối mọi thứ.** Mỗi lần chạy (baseline / student / ablation)
→ một `results/<case_id>.json` theo `results_schema.RunResult`:

```json
{ "case_id": "MERA-XQUAD__test__seed42", "method": "MERA-XQUAD",
  "track": "semi_supervised", "language": "all", "seed": 42, "ablation": null,
  "metrics": { "exact_quad": {...}, "element": {...}, "implicit": {...},
               "multilingual": {"macro_f1":.., "worst_language_f1":.., "lang_gap":..} } }
```

`evaluation/make_tables.py` đọc **cả thư mục** `results/` và tự pivot thành:

| Bảng | Nội dung | Nguồn |
|---|---|---|
| Table 2 | English supervised QUAD | các case `track=english_supervised` |
| Table 3 | Zero-shot & translation | `zero_shot / dapt_zeroshot / native_vs_translated` |
| Table 4 | Semi-supervised multilingual (chính) | `semi_supervised` |
| Table 5 | Pseudo-label precision/coverage | metrics.pseudo_label |
| Table 6 | Implicit & rare-category | metrics.implicit / category |
| Table 7 | Ablation | các case có `ablation` |
| Table 8 | Efficiency & LLM cost | metrics.efficiency |

→ **Thêm một trường hợp = thả thêm 1 file json**, không sửa code bảng. Nhiều seed
được gộp **mean±std** tự động (§6.4 yêu cầu ≥3 seed).

---

## 4. Quy trình chạy trên Kaggle

**Chuẩn bị dataset (Add data):**
1. `multilabsa-code` — toàn bộ repo (trừ model, đã `.gitignore`) → cung cấp code + `data_final/`.
2. `hotel-mt5-asqp` — T_G.
3. `dual-teacher` — `checkpoints/dual-teacher/extractive_teacher.pt`.
4. `hotel-unlabeled` — CSV review đa ngôn ngữ (cho self-training).

**Chạy theo thứ tự (mỗi notebook độc lập, cùng ghi vào `/kaggle/working/results`):**
1. `baselines/baselines-kaggle.ipynb` → M1, M3 → results.
2. `student/student-kaggle.ipynb` (cần GPU) → train MERA-XQUAD + các ablation → results.
3. `evaluation/evaluation-kaggle.ipynb` → gom results → **Table 1–8** (`tables.md`).

> `results/` tích luỹ qua các lần chạy. Muốn so sánh case nào thì chạy case đó rồi
> chạy lại notebook evaluation.

---

## 5. 7 track (§6.1) và 10 ablation (§7.1) = cùng pipeline, khác config

**Track** (đặt `--track` khi `run_eval`): `english_supervised`, `zero_shot`,
`dapt_zeroshot`, `semi_supervised`, `few_shot`, `hotel_disjoint`, `native_vs_translated`.

**Ablation** (`student/train_student.py` toggles → `evaluation/ablation.py`):
`no_partial`, `no_relation`, `single_teacher`, `no_multiview`, `no_dapt`,
`fixed_threshold`, `no_consistency`, `no_deferred`, `no_implicit`. Mỗi cái là một
dòng Table 7 (`--ablation <tag>` khi eval).

---

## 6. Quy tắc so sánh công bằng (§6.4) — được ép bởi thiết kế
- Cùng gold train/val/test và cùng target gold test (`common/data.load_gold_split`).
- Cùng metric implementation (`common/metrics.py`) cho MỌI case.
- Cùng backbone khi so cơ chế transfer; ≥3 seed (5 cho bảng chính) — `tables.py` gộp mean±std.
- Kết luận vượt trội phải qua `significance.paired_bootstrap` (CI 95% loại trừ 0).

---

## 7. Trạng thái thực thi (thẳng thắn)
- ✅ **Chạy & test được ngay (không cần GPU):** toàn bộ `common/` (metrics, tables,
  significance), `evaluation/`, baseline inference `zero_shot`/`translate_test`.
- 🖥️ **Cần GPU Kaggle:** `student/train_student.py` (self-training), và các baseline
  huấn luyện (translate-train/k-shot — dùng `train_asqp_mt5.py` trên dữ liệu tương ứng).
- Các phần torch được viết theo đúng pattern `train.py`/`train_asqp_mt5.py` sẵn có;
  logic thuần Python (metrics/tables/routing/reliability) đã có unit test.

---

## 8. Bổ sung theo rà soát docx (rule còn thiếu)

Các rule/step trước còn thiếu, nay **đã thêm & unit-test** (không cần GPU/LLM):

| Rule (docx) | File | Vai trò |
|---|---|---|
| §6.2 / Table 5 | `evaluation/pseudo_label_audit.py` | precision/coverage pseudo-label theo **element + route** |
| §7.2 | `evaluation/error_analysis.py` | phân loại lỗi: boundary / category / sentiment / relation / implicit / cross-lingual / spurious / missed |
| §6.2 / Table 8 | `experiments/common/efficiency.py` | đo train/infer time, GPU-h, LLM calls, token cost |
| §5.1 P0 | `experiments/data_prep/audit.py` | audit schema/duplicate/**leak train∩test**/implicit ratio/phân bố (Table 1) |
| §6.1 Hotel-disjoint | `experiments/data_prep/hotel_disjoint_split.py` | tách khách sạn train/test không giao |
| §6.1 Few-shot (M6) | `experiments/data_prep/few_shot_builder.py` | lấy k nhãn/ngôn ngữ (10/25/50/100) |
| §4.7 | `teacher/implicit_gate.py` | luật chấp nhận NULL (agreement+stability+implicit prob) → accept / verifier / reject |

**Còn lại — cần GPU hoặc dịch vụ ngoài (chưa cài):**
- §4.3 **Code-switch view** (view thứ 4) — cần translator.
- §4.4/§4.5 **Semantic similarity + span projection** trong matching & reliability 6-thành-phần — cần embedding model.
- §4.8 **LLM verifier** thực thi cho route `verifier` — cần LLM ngoài.
- §4.9 Nối đủ `L_partial/L_cons/L_relation/L_align` vào vòng train student — cần GPU để chạy/kiểm.
