"""Aggregate sweep results into CSV / Markdown tables.

A sweep (see :mod:`langlab.training.sweep`) writes one ``results.json`` per run
and seed::

    {"run": "mlp", "seed": 1, "params": {...}, "metrics": {"iid_acc": 1.0, ...}}

This module groups them by ``run`` and reports mean ± std over seeds. It only
uses the standard library.
"""

import csv
import glob
import json
import os
import re
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Sequence

from ..utils.utils import get_logger

logger = get_logger(__name__)

# Metric columns shown in reports, in order, with their display names.
REPORT_METRICS = {
    "iid_acc": "IID acc",
    "compo_acc": "Compo acc",
    "compo_target_acc": "Held-out target acc",
    "iid_acc_min": "Worst pair acc",
    "agreement": "Agreement",
    "topsim": "TopSim",
    "posdis": "PosDis",
    "n_messages": "# messages",
}

README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"


def load_results(sweep_dir: str) -> List[Dict[str, Any]]:
    """Load every ``results.json`` under ``sweep_dir``, sorted by run then seed."""
    results = []
    for path in glob.glob(
        os.path.join(sweep_dir, "**", "results.json"), recursive=True
    ):
        with open(path) as f:
            results.append(json.load(f))
    return sorted(results, key=lambda r: (r.get("order", 0), r["run"], r["seed"]))


def aggregate(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per run: ``{"run", "n_seeds", "<metric>_mean", "<metric>_std", ...}``."""
    runs: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        runs.setdefault(r["run"], []).append(r)

    rows = []
    for run, rs in runs.items():
        row: Dict[str, Any] = {"run": run, "n_seeds": len(rs)}
        for metric in REPORT_METRICS:
            values = [r["metrics"][metric] for r in rs if metric in r["metrics"]]
            if values:
                row[f"{metric}_mean"] = mean(values)
                row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def write_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    fields: List[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(metric: str, m: float, s: float) -> str:
    if metric == "n_messages":
        return f"{m:.1f} ± {s:.1f}"
    if metric.endswith("acc") or metric in ("agreement", "iid_acc_min"):
        return f"{100 * m:.1f} ± {100 * s:.1f}%"
    return f"{m:.2f} ± {s:.2f}"


def to_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    """Markdown table with one column per metric present in any row."""
    metrics = [m for m in REPORT_METRICS if any(f"{m}_mean" in r for r in rows)]
    header = ["Run", "Seeds"] + [REPORT_METRICS[m] for m in metrics]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for r in rows:
        cells = [str(r["run"]), str(r["n_seeds"])]
        for m in metrics:
            cells.append(
                _fmt(m, r[f"{m}_mean"], r[f"{m}_std"]) if f"{m}_mean" in r else "–"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_readme(readme_path: str, markdown: str) -> None:
    """Replace the text between the results markers in ``readme_path``."""
    with open(readme_path) as f:
        text = f.read()
    pattern = re.compile(
        re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL
    )
    if not pattern.search(text):
        raise ValueError(f"{readme_path} has no {README_START} ... {README_END} block")
    replacement = f"{README_START}\n{markdown}\n{README_END}"
    with open(readme_path, "w") as f:
        f.write(pattern.sub(lambda _: replacement, text))


def create_report(sweep_dir: str, readme: Optional[str] = None) -> str:
    """Aggregate ``sweep_dir`` into ``summary.csv`` and ``summary.md``.

    Returns:
        The Markdown table (also written into ``readme`` if given).
    """
    results = load_results(sweep_dir)
    if not results:
        raise ValueError(f"No results.json found under {sweep_dir}")
    rows = aggregate(results)
    markdown = to_markdown(rows)
    write_csv(rows, os.path.join(sweep_dir, "summary.csv"))
    with open(os.path.join(sweep_dir, "summary.md"), "w") as f:
        f.write(markdown + "\n")
    if readme:
        update_readme(readme, markdown)
    logger.info(f"Aggregated {len(results)} runs into {len(rows)} rows")
    return markdown
