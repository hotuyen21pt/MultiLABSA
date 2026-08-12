# baselines/ — §3

Sinh predictions của baseline → `evaluation/run_eval.py` chấm điểm → một dòng bảng.

| Baseline | Loại | Cách chạy |
|---|---|---|
| **M1 Zero-shot** | inference (chạy ngay) | `run_baseline.py --baseline zero_shot` |
| **M3 Translate-test** | inference (cần NLLB) | `run_baseline.py --baseline translate_test` |
| Paraphrase-mT5 / GAS / supervised QUAD | train | `train_asqp_mt5.py` (đã có) trên gold EN |
| M2 DAPT-zeroshot | inference | dùng backbone `hotel-mt5` (DAPT) làm T_G rồi như M1 |
| M4/M5 Translate-train, M6 k-shot | train | `train_asqp_mt5.py` trên gold đã dịch / few-shot |
| SSL (Full-QUAD ST, Mean Teacher, …) | train | `student/train_student.py` với toggle tương ứng |

**Ví dụ:**
```bash
python -m baselines.run_baseline --baseline zero_shot \
  --input data_final/labeled_data/hamos26/test.json \
  --generative_model hotel-mt5-asqp --out preds/m1.json
python -m evaluation.run_eval --predictions preds/m1.json \
  --method M1_zeroshot --track zero_shot --split test --seed 42
```

Notebook: `baselines-kaggle.ipynb`.
