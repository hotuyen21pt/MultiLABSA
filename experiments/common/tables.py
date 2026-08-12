"""Comparison-table builder (MultiLABSA.docx §6.3, Table 1–8).

Reads a folder of ``results/<case_id>.json`` (see results_schema.py) and pivots
them into the paper tables. Pure Python (no pandas needed) so it runs anywhere;
each table is a list-of-row-dicts plus a markdown renderer.

Seeds are aggregated as mean±std per (method, language) so Table 2–4 report the
§6.4-required multi-seed summary automatically.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from experiments.common.results_schema import RunResult, load_results


# --------------------------------------------------------------------------- #
# Aggregation helpers                                                           #
# --------------------------------------------------------------------------- #
def _mean_std(values: List[float]) -> str:
    if not values:
        return "-"
    m = sum(values) / len(values)
    if len(values) == 1:
        return f"{m:.3f}"
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))
    return f"{m:.3f}±{sd:.3f}"


def _get(metrics: Dict[str, Any], path: str, default: float = 0.0) -> float:
    """Dotted lookup: 'exact_quad.f1' or 'element.AT.f1'."""
    cur: Any = metrics
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else default


def _pivot(
    results: List[RunResult],
    row_key: Callable[[RunResult], str],
    col_key: Callable[[RunResult], str],
    value_path: str,
) -> List[Dict[str, str]]:
    """Group by row_key × col_key, aggregating value_path over seeds (mean±std)."""
    cells: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    cols: List[str] = []
    for r in results:
        rk, ck = row_key(r), col_key(r)
        cells[rk][ck].append(_get(r.metrics, value_path))
        if ck not in cols:
            cols.append(ck)
    rows: List[Dict[str, str]] = []
    for rk in sorted(cells):
        row = {"row": rk}
        for ck in cols:
            row[ck] = _mean_std(cells[rk][ck]) if ck in cells[rk] else "-"
        rows.append(row)
    return rows


def to_markdown(rows: List[Dict[str, str]], title: str = "") -> str:
    if not rows:
        return f"### {title}\n(no results)\n"
    headers = list(rows[0].keys())
    out = [f"### {title}"] if title else []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(r.get(h, "-")) for h in headers) + " |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# The paper tables (§6.3)                                                        #
# --------------------------------------------------------------------------- #
def table2_english_supervised(results):
    r = [x for x in results if x.track == "english_supervised"]
    return _pivot(r, lambda x: x.method, lambda x: "exact_Quad_F1", "exact_quad.f1")


def table3_transfer(results):
    r = [x for x in results if x.track in ("zero_shot", "dapt_zeroshot", "native_vs_translated")]
    return _pivot(r, lambda x: x.method, lambda x: x.language, "exact_quad.f1")


def table4_semi_supervised(results):
    r = [x for x in results if x.track == "semi_supervised"]
    return _pivot(r, lambda x: x.method, lambda x: x.language, "exact_quad.f1")


def table5_pseudo_label(results):
    """Pseudo-label precision/coverage by element+route (from metrics['pseudo_label'])."""
    r = [x for x in results if "pseudo_label" in x.metrics]
    rows = []
    for x in r:
        pl = x.metrics["pseudo_label"]
        for element, stats in pl.items():
            rows.append({
                "method": x.method, "element": element,
                "precision": f"{stats.get('precision', 0):.3f}",
                "coverage": f"{stats.get('coverage', 0):.3f}",
            })
    return rows


def table6_implicit_rare(results):
    r = [x for x in results if x.track in ("semi_supervised", "english_supervised")]
    rows = []
    for x in r:
        imp = x.metrics.get("implicit", {})
        cat = x.metrics.get("category", {})
        rows.append({
            "method": x.method, "language": x.language,
            "EA-EO": f"{_get(imp, 'EA-EO.f1'):.3f}", "IA-EO": f"{_get(imp, 'IA-EO.f1'):.3f}",
            "EA-IO": f"{_get(imp, 'EA-IO.f1'):.3f}", "IA-IO": f"{_get(imp, 'IA-IO.f1'):.3f}",
            "rare_cat_F1": f"{cat.get('rare_category_f1') or 0:.3f}",
            "worst_cat_F1": f"{cat.get('worst_category_f1', 0):.3f}",
        })
    return rows


def table7_ablation(results):
    r = [x for x in results if x.ablation]
    return _pivot(r, lambda x: x.ablation, lambda x: x.language, "exact_quad.f1")


def table8_efficiency(results):
    r = [x for x in results if "efficiency" in x.metrics]
    rows = []
    for x in r:
        eff = x.metrics["efficiency"]
        rows.append({
            "method": x.method,
            "train_h": f"{eff.get('train_hours', 0):.2f}",
            "infer_s": f"{eff.get('inference_seconds', 0):.2f}",
            "gpu_h": f"{eff.get('gpu_hours', 0):.2f}",
            "llm_calls": str(eff.get("llm_calls", 0)),
            "token_cost": str(eff.get("token_cost", 0)),
        })
    return rows


def build_all_tables(results_dir: str = "results") -> Dict[str, List[Dict[str, str]]]:
    results = load_results(results_dir)
    return {
        "table2_english_supervised": table2_english_supervised(results),
        "table3_transfer": table3_transfer(results),
        "table4_semi_supervised": table4_semi_supervised(results),
        "table5_pseudo_label": table5_pseudo_label(results),
        "table6_implicit_rare": table6_implicit_rare(results),
        "table7_ablation": table7_ablation(results),
        "table8_efficiency": table8_efficiency(results),
    }


def render_all_tables(results_dir: str = "results") -> str:
    tables = build_all_tables(results_dir)
    titles = {
        "table2_english_supervised": "Table 2 — English supervised QUAD baselines",
        "table3_transfer": "Table 3 — Zero-shot & translation transfer",
        "table4_semi_supervised": "Table 4 — Semi-supervised multilingual",
        "table5_pseudo_label": "Table 5 — Pseudo-label precision/coverage by element",
        "table6_implicit_rare": "Table 6 — Implicit & rare-category",
        "table7_ablation": "Table 7 — Ablation",
        "table8_efficiency": "Table 8 — Efficiency & LLM cost",
    }
    return "\n".join(to_markdown(rows, titles[name]) for name, rows in tables.items())
