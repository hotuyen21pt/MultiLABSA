"""Build Table 1–8 from a results folder (MultiLABSA.docx §6.3).

    python -m evaluation.make_tables --results_dir results --out tables.md

Reads every ``results/<case_id>.json`` and renders the paper tables (mean±std
over seeds). Add a case = drop another json in ``--results_dir``; no code change.
"""

from __future__ import annotations

import argparse

from experiments.common.tables import build_all_tables, render_all_tables


def main():
    p = argparse.ArgumentParser(description="Render comparison tables from results/")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out", default=None, help="write markdown here (default: stdout)")
    p.add_argument("--json", default=None, help="also dump raw table rows as json")
    a = p.parse_args()

    md = render_all_tables(a.results_dir)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[make_tables] wrote {a.out}")
    else:
        print(md)
    if a.json:
        import json
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(build_all_tables(a.results_dir), f, ensure_ascii=False, indent=2)
        print(f"[make_tables] wrote {a.json}")


if __name__ == "__main__":
    main()
