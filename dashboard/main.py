"""Flask application for the Language Emergence Lab dashboard.

Serves experiment data and metrics through a web interface for visualization
and analysis of referential game results.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from flask import Flask, Response, jsonify, render_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = Path(__file__).parent.parent
EXPERIMENTS_DIR = BASE_DIR / "outputs" / "experiments"
LOGS_DIR = BASE_DIR / "outputs" / "logs"


def load_experiment_data(exp_dir: Path) -> Union[Dict[str, Any], None]:
    """Load metrics and configuration from an experiment directory."""
    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path, "r") as f:
        data: Dict[str, Any] = json.load(f)

    zipf_path = exp_dir / "zipf_slope.json"
    if zipf_path.exists():
        with open(zipf_path, "r") as f:
            zipf_data: Dict[str, Any] = json.load(f)
            data["zipf_slope"] = zipf_data.get("slope", data.get("zipf_slope"))

    return data


def load_training_logs() -> List[Dict[str, float]]:
    """Load training metrics from available CSV logs."""
    log_sources = [
        ("grid_metrics.csv", "episode"),
        ("population_metrics.csv", "step"),
        ("metrics.csv", "step"),
    ]

    for filename, step_col in log_sources:
        path = LOGS_DIR / filename
        if not path.exists():
            continue

        logs = []
        try:
            with open(path, "r") as f:
                header = f.readline().strip().split(",")
                if not header:
                    continue

                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    values = line.split(",")
                    if len(values) == len(header):
                        try:
                            row = {}
                            for k, v in zip(header, values):
                                try:
                                    row[k] = float(v)
                                except (ValueError, TypeError):
                                    row[k] = 0.0
                            logs.append(row)
                        except Exception:
                            continue

            if len(logs) > 1:
                return logs

        except Exception:
            continue

    json_path = LOGS_DIR / "improved_training_results.json"
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data: Dict[str, Any] = json.load(f)
                if "training_logs" in data and data["training_logs"]:
                    training_logs: List[Dict[str, float]] = data["training_logs"]
                    return training_logs
        except Exception:
            pass

    return []


@app.route("/")
def index() -> str:
    """Render the main dashboard page."""
    logger.info("Dashboard accessed")
    return str(render_template("dashboard.html"))


@app.route("/api/experiments")
def get_experiments() -> Response:
    """Retrieve all experiment results."""
    experiments = []

    if EXPERIMENTS_DIR.exists():
        for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
            if exp_dir.is_dir():
                data = load_experiment_data(exp_dir)
                if data:
                    experiments.append(data)

    logger.info(f"Retrieved {len(experiments)} experiments")
    return jsonify(experiments)


@app.route("/api/training-logs")
def get_training_logs() -> Response:
    """Retrieve training metrics over time."""
    logs = load_training_logs()
    return jsonify(logs)


@app.route("/api/stats")
def get_stats() -> Response:
    """Retrieve aggregate statistics across all experiments."""
    experiments = []

    if EXPERIMENTS_DIR.exists():
        for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
            if exp_dir.is_dir():
                data = load_experiment_data(exp_dir)
                if data:
                    experiments.append(data)

    if not experiments:
        return jsonify({"total_experiments": 0})

    total = len(experiments)
    avg_train_acc = sum(e["metrics"]["train"]["acc"] for e in experiments) / total
    avg_iid_acc = sum(e["metrics"]["iid"]["acc"] for e in experiments) / total
    avg_compo_acc = sum(e["metrics"]["compo"]["acc"] for e in experiments) / total

    return jsonify(
        {
            "total_experiments": total,
            "avg_train_accuracy": round(avg_train_acc, 3),
            "avg_iid_accuracy": round(avg_iid_acc, 3),
            "avg_compositional_accuracy": round(avg_compo_acc, 3),
        }
    )


@app.route("/api/healthz")
def health_check() -> Response:
    """Health check endpoint for monitoring dashboard availability."""
    return jsonify({"status": "healthy", "service": "language-emergence-dashboard"})


if __name__ == "__main__":
    import os

    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_RUN_PORT", "5000"))

    logger.info(f"Starting Language Emergence Lab Dashboard on {host}:{port}")
    app.run(debug=True, host=host, port=port)
