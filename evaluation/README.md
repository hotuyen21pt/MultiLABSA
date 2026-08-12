# evaluation/ — §6 chấm điểm + §7 ablation

| File | Vai trò |
|---|---|
| `run_eval.py` | predictions json + gold split → tính §6.2 metrics (per-language + macro/worst/LangGap) → `results/<case_id>.json` |
| `make_tables.py` | gom `results/*.json` → **Table 1–8** (markdown/json), mean±std theo seed |
| `ablation.py` | khai báo 10 ablation §7.1 (tag → delta config) để driver/notebook chạy |

**Chạy:**
```bash
# 1) chấm điểm từng case (baseline / student) → results/
python -m evaluation.run_eval --predictions preds/m1.json \
  --method M1_zeroshot --track zero_shot --split test --seed 42

# 2) dựng toàn bộ bảng so sánh
python -m evaluation.make_tables --results_dir results --out tables.md
```

Metric (§6.2): `exact_quad`, `partial_quad` (soft-span), `element` (AT/AC/OT/SP),
`implicit` (EA-EO/IA-EO/EA-IO/IA-IO), `category` (macro/rare/worst),
`multilingual` (macro/worst-lang/LangGap), `calibration_ece`.

Kiểm định (§7.3): `experiments/common/significance.paired_bootstrap`.

Notebook: `evaluation-kaggle.ipynb`.
