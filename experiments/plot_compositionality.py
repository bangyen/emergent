"""Scatter held-out-target accuracy against TopSim for a finished sweep.

Usage: python experiments/plot_compositionality.py SWEEP_DIR [OUT_PNG]
(e.g. after `langlab sweep experiments/compositionality.json`).
"""

import json
import os
import sys
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from langlab.analysis.language import spearman  # noqa: E402


def main() -> None:
    sweep_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "docs/compositionality.png"
    runs: dict = {}
    for path in glob(os.path.join(sweep_dir, "*", "seed*", "results.json")):
        with open(path) as f:
            result = json.load(f)
        runs.setdefault(result["run"], []).append(result["metrics"])

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (run, metrics), marker in zip(sorted(runs.items()), "so^D"):
        x = [m["topsim"] for m in metrics]
        y = [m["compo_target_acc"] for m in metrics]
        rho = spearman(x, y)
        ax.scatter(
            x, y, marker=marker, alpha=0.8, label=f"{run} (ρ = {rho:.2f}, n = {len(x)})"
        )
    ax.set_xlabel("TopSim of the speaker's lexicon")
    ax.set_ylabel("held-out target accuracy")
    ax.set_ylim(-0.02, 1.04)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
