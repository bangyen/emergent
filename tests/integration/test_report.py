"""Tests for reporting functionality.

This module tests the reporting system including result aggregation,
CSV generation, and chart creation for ablation studies.
"""

import pytest
import tempfile
import os
import json
import pandas as pd
from typing import Any
from unittest.mock import patch

from langlab.analysis.report import (
    load_experiment_results,
    aggregate_results,
    save_aggregated_csv,
    generate_summary_report,
)


def test_load_experiment_results() -> None:
    """Test loading experiment results from JSON file."""
    # Create temporary directory and file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample experiment results
        sample_results = {
            "experiment_id": "test_exp",
            "params": {"V": 6, "channel_noise": 0.0, "length_cost": 0.01},
            "metrics": {"train": {"acc": 0.8}, "compo": {"acc": 0.7}},
            "zipf_slope": -0.8,
        }

        metrics_path = os.path.join(temp_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(sample_results, f)

        # Test loading
        results = load_experiment_results(temp_dir)

        assert results is not None
        assert results["experiment_id"] == "test_exp"
        assert results["params"]["V"] == 6
        assert results["metrics"]["train"]["acc"] == 0.8


def test_load_experiment_results_missing_file() -> None:
    """Test loading from non-existent file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results = load_experiment_results(temp_dir)
        assert results is None


def test_load_experiment_results_invalid_json() -> None:
    """Test loading from invalid JSON file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = os.path.join(temp_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            f.write("invalid json content")

        results = load_experiment_results(temp_dir)
        assert results is None


def test_aggregate_results() -> None:
    """Test aggregating results from multiple experiments."""
    # Create temporary directories with experiment results
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple experiment directories
        exp_dirs = []
        for i in range(3):
            exp_dir = os.path.join(temp_dir, f"exp_{i}")
            os.makedirs(exp_dir)
            exp_dirs.append(exp_dir)

            # Create metrics.json
            sample_results = {
                "experiment_id": f"exp_{i}",
                "params": {
                    "V": 6 + i * 6,
                    "channel_noise": 0.0 + i * 0.05,
                    "length_cost": 0.0 + i * 0.01,
                },
                "metrics": {
                    "train": {"acc": 0.8 - i * 0.1},
                    "compo": {"acc": 0.7 - i * 0.1},
                },
                "zipf_slope": -0.8 - i * 0.1,
            }

            metrics_path = os.path.join(exp_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(sample_results, f)

        # Test aggregation
        pattern = os.path.join(temp_dir, "*")
        df = aggregate_results(pattern)

        # Verify DataFrame structure
        assert len(df) == 3
        assert "V" in df.columns
        assert "channel_noise" in df.columns
        assert "length_cost" in df.columns
        assert "acc" in df.columns
        assert "compo_acc" in df.columns
        assert "zipf_slope" in df.columns

    # Verify data values (order may vary due to glob)
    assert set(df["V"].tolist()) == {6, 12, 18}
    assert set([round(x, 1) for x in df["acc"].tolist()]) == {0.8, 0.7, 0.6}


def test_aggregate_results_empty() -> None:
    """Test aggregating results when no experiments exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        pattern = os.path.join(temp_dir, "*")
        df = aggregate_results(pattern)

        assert df.empty


def test_report_columns() -> None:
    """Test that aggregated CSV has correct columns."""
    # Create sample DataFrame
    sample_data = [
        {
            "V": 6,
            "channel_noise": 0.0,
            "length_cost": 0.0,
            "acc": 0.8,
            "compo_acc": 0.7,
            "zipf_slope": -0.8,
        },
        {
            "V": 12,
            "channel_noise": 0.05,
            "length_cost": 0.01,
            "acc": 0.75,
            "compo_acc": 0.65,
            "zipf_slope": -0.9,
        },
        {
            "V": 24,
            "channel_noise": 0.1,
            "length_cost": 0.05,
            "acc": 0.7,
            "compo_acc": 0.6,
            "zipf_slope": -1.0,
        },
    ]

    df = pd.DataFrame(sample_data)

    # Test saving CSV
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as temp_file:
        temp_path = temp_file.name

    try:
        save_aggregated_csv(df, temp_path)

        # Verify CSV was created and has correct columns
        assert os.path.exists(temp_path)

        loaded_df = pd.read_csv(temp_path)
        expected_columns = {
            "V",
            "channel_noise",
            "length_cost",
            "acc",
            "compo_acc",
            "zipf_slope",
        }
        actual_columns = set(loaded_df.columns)

        assert expected_columns.issubset(actual_columns)
        assert len(loaded_df) == 3

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_generate_summary_report() -> None:
    """Test summary report generation."""
    # Create sample DataFrame
    sample_data = [
        {
            "V": 6,
            "channel_noise": 0.0,
            "length_cost": 0.0,
            "acc": 0.8,
            "compo_acc": 0.7,
            "zipf_slope": -0.8,
        },
        {
            "V": 12,
            "channel_noise": 0.05,
            "length_cost": 0.01,
            "acc": 0.75,
            "compo_acc": 0.65,
            "zipf_slope": -0.9,
        },
        {
            "V": 24,
            "channel_noise": 0.1,
            "length_cost": 0.05,
            "acc": 0.7,
            "compo_acc": 0.6,
            "zipf_slope": -1.0,
        },
    ]

    df = pd.DataFrame(sample_data)
    summary = generate_summary_report(df)

    # Verify summary structure
    assert "total_experiments" in summary
    assert "parameter_ranges" in summary
    assert "performance_stats" in summary
    assert "best_performing" in summary

    # Verify values
    assert summary["total_experiments"] == 3

    assert summary["parameter_ranges"]["vocabulary_size"]["min"] == 6
    assert summary["parameter_ranges"]["vocabulary_size"]["max"] == 24

    assert summary["performance_stats"]["accuracy"]["mean"] == pytest.approx(
        0.75, abs=0.01
    )
    assert summary["performance_stats"]["accuracy"]["max"] == 0.8

    assert summary["best_performing"]["accuracy"]["V"] == 6
    assert summary["best_performing"]["accuracy"]["acc"] == 0.8


def test_generate_summary_report_empty() -> None:
    """Test summary report generation with empty DataFrame."""
    df = pd.DataFrame()
    summary = generate_summary_report(df)

    assert "error" in summary
    assert summary["error"] == "No data available for summary"


@patch("langlab.analysis.report.create_comparative_chart")
@patch("langlab.analysis.report.create_heatmap_chart")
def test_create_report_with_charts(mock_heatmap: Any, mock_comparative: Any) -> None:
    """Test report creation with chart generation."""
    from langlab.analysis.report import create_report

    # Create sample DataFrame
    sample_data = [
        {
            "V": 6,
            "channel_noise": 0.0,
            "length_cost": 0.0,
            "acc": 0.8,
            "compo_acc": 0.7,
            "zipf_slope": -0.8,
        },
    ]

    df = pd.DataFrame(sample_data)

    with patch("langlab.analysis.report.aggregate_results", return_value=df):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_info = create_report(
                input_pattern="test_pattern", output_dir=temp_dir, create_charts=True
            )

            # Verify report structure
            assert "csv_path" in report_info
            assert "summary_path" in report_info
            assert "total_experiments" in report_info

            # Verify charts were requested
            assert mock_comparative.call_count >= 2  # Accuracy and compositional charts
            assert mock_heatmap.call_count >= 1


def test_create_report_no_charts() -> None:
    """Test report creation without chart generation."""
    from langlab.analysis.report import create_report

    # Create sample DataFrame
    sample_data = [
        {
            "V": 6,
            "channel_noise": 0.0,
            "length_cost": 0.0,
            "acc": 0.8,
            "compo_acc": 0.7,
            "zipf_slope": -0.8,
        },
    ]

    df = pd.DataFrame(sample_data)

    with patch("langlab.analysis.report.aggregate_results", return_value=df):
        with patch(
            "langlab.analysis.report.create_comparative_chart"
        ) as mock_comparative:
            with patch("langlab.analysis.report.create_heatmap_chart") as mock_heatmap:
                with tempfile.TemporaryDirectory() as temp_dir:
                    report_info = create_report(
                        input_pattern="test_pattern",
                        output_dir=temp_dir,
                        create_charts=False,
                    )

                    # Verify no charts were created
                    mock_comparative.assert_not_called()
                    mock_heatmap.assert_not_called()

                    # Verify report structure
                    assert "csv_path" in report_info
                    assert "summary_path" in report_info
