"""Config-driven experiment sweeps.

A sweep config is a JSON file::

    {
      "base": {"n_steps": 10000, "k": 5, "v": 16, "message_length": 2},
      "variants": [
        {"name": "mlp"},
        {"name": "gru", "use_sequence_models": true, "learning_rate": 0.001},
        {"name": "population", "runner": "population", "n_agents": 3}
      ],
      "grid": {"v": [8, 16]},
      "seeds": [1, 2, 3]
    }

Every variant is combined with every point of the (optional) ``grid`` and run
once per seed. Keys are keyword arguments of :func:`~langlab.training.train.train`
or, with ``"runner": "population"``, of
:func:`~langlab.training.population.train_population`. Each run writes into
``<out_dir>/<run>/seed<seed>/`` and records ``results.json``; runs whose
``results.json`` already exists are skipped, so an interrupted sweep resumes.
"""

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from ..analysis.report import create_report
from ..utils.utils import get_logger

logger = get_logger(__name__)


def expand(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a sweep config into a list of run specs."""
    base = dict(config.get("base", {}))
    variants = config.get("variants") or [{}]
    grid = config.get("grid", {})
    seeds = config.get("seeds", [base.pop("seed", 7)])
    grid_points = [
        dict(zip(grid, values)) for values in itertools.product(*grid.values())
    ] or [{}]

    specs = []
    order = 0
    for variant in variants:
        variant = dict(variant)
        name = variant.pop("name", "run")
        for point in grid_points:
            run = name + "".join(f"_{k}={v}" for k, v in point.items())
            params = {**base, **variant, **point}
            runner = params.pop("runner", "train")
            if params.get("heldout_pairs"):
                params["heldout_pairs"] = [tuple(p) for p in params["heldout_pairs"]]
            for seed in seeds:
                specs.append(
                    {
                        "run": run,
                        "order": order,
                        "runner": runner,
                        "seed": seed,
                        "params": {**params, "seed": seed},
                    }
                )
            order += 1
    return specs


def run_one(spec: Dict[str, Any], out_dir: str, threads: int = 0) -> Dict[str, Any]:
    """Run a single spec (skipping it if its results already exist)."""
    run_dir = os.path.join(out_dir, spec["run"], f"seed{spec['seed']}")
    results_path = os.path.join(run_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            return dict(json.load(f))

    import torch

    if threads:
        torch.set_num_threads(threads)

    from .population import train_population
    from .train import train

    runners: Dict[str, Callable[..., Dict[str, float]]] = {
        "train": train,
        "population": train_population,
    }
    if spec["runner"] not in runners:
        raise ValueError(f"Unknown runner '{spec['runner']}'")
    metrics = runners[spec["runner"]](**spec["params"], out_dir=run_dir)

    result = {**spec, "metrics": metrics}
    os.makedirs(run_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_sweep(
    config: Dict[str, Any],
    out_dir: str,
    jobs: int = 1,
    readme: Optional[str] = None,
) -> str:
    """Run every spec in ``config`` and aggregate the results.

    Args:
        config: Parsed sweep config.
        out_dir: Directory for run outputs and ``summary.{csv,md}``.
        jobs: Number of runs to execute in parallel processes.
        readme: If given, write the summary table into this README's results block.

    Returns:
        The Markdown summary table.
    """
    specs = expand(config)
    logger.info(f"Sweep: {len(specs)} runs -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sweep.json"), "w") as f:
        json.dump(config, f, indent=2)

    if jobs > 1:
        threads = max(1, (os.cpu_count() or 1) // jobs)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            list(
                pool.map(run_one, specs, [out_dir] * len(specs), [threads] * len(specs))
            )
    else:
        for spec in specs:
            run_one(spec, out_dir)

    return create_report(out_dir, readme=readme)
