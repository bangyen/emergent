import json
import os
import shutil
import tempfile
from typing import Generator

import pandas as pd
import pytest
from langlab.analysis.report import (
    aggregate_results,
    create_report,
    generate_summary_report,
)


@pytest.fixture
def experiment_data() -> Generator[str, None, None]:
    temp_dir = tempfile.mkdtemp()

    # Create dummy experiment results
    exp1_dir = os.path.join(temp_dir, "exp1")
    os.makedirs(exp1_dir)

    metrics = {
        "experiment_id": "exp1",
        "params": {"V": 10, "channel_noise": 0.1, "length_cost": 0.01},
        "metrics": {"train": {"acc": 0.8}, "compo": {"acc": 0.5}},
        "zipf_slope": -1.2,
    }

    with open(os.path.join(exp1_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    exp2_dir = os.path.join(temp_dir, "exp2")
    os.makedirs(exp2_dir)

    metrics2 = {
        "experiment_id": "exp2",
        "params": {"V": 20, "channel_noise": 0.0, "length_cost": 0.0},
        "metrics": {"train": {"acc": 0.9}, "compo": {"acc": 0.6}},
        "zipf_slope": -1.1,
    }

    with open(os.path.join(exp2_dir, "metrics.json"), "w") as f:
        json.dump(metrics2, f)

    yield temp_dir

    shutil.rmtree(temp_dir)


def test_aggregate_results(experiment_data: str) -> None:
    pattern = os.path.join(experiment_data, "**/metrics.json")
    df = aggregate_results(pattern)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "V" in df.columns
    assert "acc" in df.columns
    assert "zipf_slope" in df.columns


def test_generate_summary_report() -> None:
    df = pd.DataFrame(
        [
            {
                "V": 10,
                "channel_noise": 0.1,
                "length_cost": 0.01,
                "acc": 0.8,
                "compo_acc": 0.5,
                "zipf_slope": -1.2,
                "experiment_id": "1",
            },
            {
                "V": 20,
                "channel_noise": 0.0,
                "length_cost": 0.0,
                "acc": 0.9,
                "compo_acc": 0.6,
                "zipf_slope": -1.1,
                "experiment_id": "2",
            },
        ]
    )

    summary = generate_summary_report(df)
    assert summary["total_experiments"] == 2
    assert summary["performance_stats"]["accuracy"]["mean"] == pytest.approx(0.85)
    assert "best_performing" in summary


def test_create_report(experiment_data: str) -> None:
    output_dir = os.path.join(experiment_data, "summary")
    pattern = os.path.join(experiment_data, "**/metrics.json")

    report_info = create_report(pattern, output_dir=output_dir, create_charts=False)

    assert "csv_path" in report_info
    assert "summary_path" in report_info
    assert os.path.exists(report_info["csv_path"])
    assert os.path.exists(report_info["summary_path"])
