# student/ — MERA-XQUAD hybrid student (§4.7, §4.9, §4.10)

Thành phần **G + I + J** của thiết kế. Xem `experiments/README.md` cho bức tranh chung.

| File | Vai trò |
|---|---|
| `hybrid_student.py` | mT5 (sinh full-quad) + aux heads (AT/OT span, AC/SP, AT–OT relation) + **implicit NULL heads** `p(AT=NULL)/p(OT=NULL)` (§4.7). Một encoder dùng chung cho cả decoder sinh lẫn các head → học được từ **partial** pseudo-label. |
| `losses.py` | `L = L_sup + λ_f L_full + λ_p L_partial + λ_c L_cons + λ_r L_relation + λ_a L_align + λ_b L_balance` (§4.9). Route nào cấp thành phần nào → route điều khiển loss nào bật. |
| `self_training.py` | **EMA teacher** `θ_T←μθ_T+(1−μ)θ_S`, **dynamic curriculum** (chấm ngôn ngữ theo confidence/agreement/stability/coverage/entropy, nạp ngôn ngữ "chín" trước, sampler cân bằng), **adaptive threshold** (§4.10). |
| `train_student.py` | Vòng self-training §5.3: label pool bằng dual-teacher → route → train student → EMA → chỉnh threshold → cập nhật curriculum. Config-driven; ablation = toggle. |

**Chạy (Kaggle GPU):**
```bash
python -m student.train_student \
  --student_backbone hotel-mt5-asqp --generative_model hotel-mt5-asqp \
  --extractive_checkpoint checkpoints/dual-teacher/extractive_teacher.pt \
  --unlabeled_csv <unlabeled.csv> --rounds 3 --output_dir checkpoints/mera-student
```
Ablation: thêm `--no_multiview` / `--single_teacher` / `--fixed_threshold` / `--no_curriculum`.

Notebook: `student-kaggle.ipynb`.
