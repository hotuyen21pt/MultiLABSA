"""Baselines for the comparison tables (MultiLABSA.docx §3).

Runnable inference baselines live here (M1 zero-shot, M3 translate-test).
Training-based baselines reuse existing entry points, documented in README.md:
    * Paraphrase-mT5 / GAS / supervised QUAD  -> train_asqp_mt5.py
    * M2 DAPT-zeroshot                          -> dapt/ backbone + M1
    * M4/M5 translate-train, M6 k-shot          -> train_asqp_mt5.py on translated/few-shot data
    * SSL baselines (Full-QUAD ST, Mean Teacher) -> student/train_student.py with ablation toggles
"""
