"""Shared experiment infrastructure for MERA-XQUAD (MultiLABSA.docx §3–§7).

Everything that must be *identical* across every compared case lives here, so
baselines / the MERA student / ablations differ only by config — the fair-
comparison requirement of §6.4. Import surface:

    from experiments.common import (
        RunResult, save_result, load_results,      # results_schema
        compute_quad_metrics,                       # metrics
        build_all_tables,                           # tables
        paired_bootstrap,                           # significance
    )
"""

from experiments.common.results_schema import RunResult, load_results, save_result
from experiments.common.metrics import compute_quad_metrics

__all__ = ["RunResult", "save_result", "load_results", "compute_quad_metrics"]
