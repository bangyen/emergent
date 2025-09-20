"""Reporting module for ablation study analysis and visualization.

This module provides functionality for aggregating ablation study results,
generating comparative visualizations, and producing summary reports
to understand parameter effects on emergent language performance.
"""

import glob
import json
import os
from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.utils import get_logger

logger = get_logger(__name__)


def load_experiment_results(experiment_dir: str) -> Optional[Dict[str, Any]]:
    """Load results from a single experiment directory.

    This function loads the metrics.json file from an experiment directory
    and returns the parsed results for aggregation.

    Args:
        experiment_dir: Path to the experiment directory.

    Returns:
        Dictionary containing experiment results, or None if loading fails.
    """
    metrics_path = os.path.join(experiment_dir, "metrics.json")

    if not os.path.exists(metrics_path):
        logger.warning(f"Metrics file not found: {metrics_path}")
        return None

    try:
        with open(metrics_path, "r") as f:
            results = json.load(f)
        return results  # type: ignore
    except Exception as e:
        logger.error(f"Failed to load results from {experiment_dir}: {e}")
        return None


def aggregate_results(input_pattern: str) -> pd.DataFrame:
    """Aggregate results from multiple experiments into a single DataFrame.

    This function searches for experiment directories matching the input pattern,
    loads their results, and combines them into a structured DataFrame for analysis.

    Args:
        input_pattern: Glob pattern to match experiment directories (e.g., "outputs/experiments/**/metrics.json").

    Returns:
        DataFrame with columns: V, channel_noise, length_cost, acc, compo_acc, zipf_slope.
    """
    # Find all matching experiment directories
    experiment_dirs = glob.glob(input_pattern.replace("/metrics.json", ""))

    logger.info(f"Found {len(experiment_dirs)} experiment directories")

    aggregated_data = []

    for exp_dir in experiment_dirs:
        results = load_experiment_results(exp_dir)

        if results is None:
            continue

        # Extract parameters and metrics
        params = results["params"]
        metrics = results["metrics"]
        zipf_slope = results["zipf_slope"]

        row = {
            "V": params["V"],
            "channel_noise": params["channel_noise"],
            "length_cost": params["length_cost"],
            "acc": metrics["train"]["acc"],
            "compo_acc": metrics["compo"]["acc"],
            "zipf_slope": zipf_slope,
            "experiment_id": results["experiment_id"],
        }

        aggregated_data.append(row)

    if not aggregated_data:
        logger.warning("No valid experiment results found")
        return pd.DataFrame()

    df = pd.DataFrame(aggregated_data)
    logger.info(f"Aggregated {len(df)} experiment results")

    return df


def save_aggregated_csv(df: pd.DataFrame, output_path: str) -> None:
    """Save aggregated results to CSV file.

    This function saves the aggregated DataFrame to a CSV file for further
    analysis and sharing of results.

    Args:
        df: DataFrame containing aggregated results.
        output_path: Path where to save the CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Aggregated results saved to: {output_path}")


def create_comparative_chart(
    df: pd.DataFrame,
    output_path: str,
    metric: str = "acc",
    title: str = "Ablation Study Results",
) -> None:
    """Create comparative bar chart for ablation study results.

    This function generates a bar chart comparing performance across different
    parameter combinations, with separate bars for each condition.

    Args:
        df: DataFrame containing aggregated results.
        output_path: Path where to save the chart.
        metric: Metric to plot ("acc" or "compo_acc").
        title: Title for the chart.
    """
    if df.empty:
        logger.warning("No data to plot")
        return

    # Set up the plot style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create condition labels
    df["condition"] = df.apply(
        lambda row: f"V={row['V']}, noise={row['channel_noise']:.2f}, len={row['length_cost']:.2f}",
        axis=1,
    )

    # Sort by metric value for better visualization
    df_sorted = df.sort_values(metric, ascending=True)

    # Create bar plot
    bars = ax.barh(range(len(df_sorted)), df_sorted[metric], alpha=0.7)

    # Customize plot
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["condition"], fontsize=10)
    ax.set_xlabel(f"{metric.replace('_', ' ').title()}", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, df_sorted[metric])):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=9)

    # Add grid
    ax.grid(axis="x", alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Comparative chart saved to: {output_path}")


def create_heatmap_chart(
    df: pd.DataFrame,
    output_path: str,
    metric: str = "acc",
    title: str = "Ablation Study Heatmap",
) -> None:
    """Create heatmap visualization for ablation study results.

    This function generates a heatmap showing how different parameter combinations
    affect performance, with one parameter on each axis.

    Args:
        df: DataFrame containing aggregated results.
        output_path: Path where to save the chart.
        metric: Metric to plot ("acc" or "compo_acc").
        title: Title for the chart.
    """
    if df.empty:
        logger.warning("No data to plot")
        return

    # Create pivot table for heatmap
    # Use vocabulary size and channel noise as axes, average across length costs
    pivot_data = df.groupby(["V", "channel_noise"])[metric].mean().unstack()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": f"{metric.replace('_', ' ').title()}"},
    )

    # Customize plot
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Channel Noise", fontsize=12)
    ax.set_ylabel("Vocabulary Size", fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Heatmap chart saved to: {output_path}")


def generate_summary_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for ablation study results.

    This function computes summary statistics across all experiments to
    identify trends and significant effects in the ablation study.

    Args:
        df: DataFrame containing aggregated results.

    Returns:
        Dictionary containing summary statistics.
    """
    if df.empty:
        return {"error": "No data available for summary"}

    summary = {
        "total_experiments": len(df),
        "parameter_ranges": {
            "vocabulary_size": {"min": df["V"].min(), "max": df["V"].max()},
            "channel_noise": {
                "min": df["channel_noise"].min(),
                "max": df["channel_noise"].max(),
            },
            "length_cost": {
                "min": df["length_cost"].min(),
                "max": df["length_cost"].max(),
            },
        },
        "performance_stats": {
            "accuracy": {
                "mean": df["acc"].mean(),
                "std": df["acc"].std(),
                "min": df["acc"].min(),
                "max": df["acc"].max(),
            },
            "compositional_accuracy": {
                "mean": df["compo_acc"].mean(),
                "std": df["compo_acc"].std(),
                "min": df["compo_acc"].min(),
                "max": df["compo_acc"].max(),
            },
            "zipf_slope": {
                "mean": df["zipf_slope"].mean(),
                "std": df["zipf_slope"].std(),
                "min": df["zipf_slope"].min(),
                "max": df["zipf_slope"].max(),
            },
        },
        "best_performing": {
            "accuracy": df.loc[df["acc"].idxmax()].to_dict(),
            "compositional_accuracy": df.loc[df["compo_acc"].idxmax()].to_dict(),
        },
    }

    return summary


def create_report(
    input_pattern: str,
    output_dir: str = "outputs/summary",
    create_charts: bool = True,
) -> Dict[str, Any]:
    """Create comprehensive ablation study report.

    This function aggregates results from experiments, generates visualizations,
    and creates a summary report with key findings.

    Args:
        input_pattern: Glob pattern to match experiment directories.
        output_dir: Directory where to save report files.
        create_charts: Whether to generate visualization charts.

    Returns:
        Dictionary containing summary statistics and report paths.
    """
    logger.info(f"Creating ablation study report from: {input_pattern}")

    # Aggregate results
    df = aggregate_results(input_pattern)

    if df.empty:
        logger.error("No experiment results found to create report")
        return {"error": "No experiment results found"}

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save aggregated CSV
    csv_path = os.path.join(output_dir, "ablation.csv")
    save_aggregated_csv(df, csv_path)

    # Generate summary statistics
    summary = generate_summary_report(df)

    # Save summary to JSON
    summary_path = os.path.join(output_dir, "summary.json")

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        else:
            return obj

    summary_serializable = convert_numpy_types(summary)

    with open(summary_path, "w") as f:
        json.dump(summary_serializable, f, indent=2)

    report_info = {
        "csv_path": csv_path,
        "summary_path": summary_path,
        "total_experiments": len(df),
        "summary": summary,
    }

    # Create charts if requested
    if create_charts:
        figures_dir = os.path.join(output_dir, "..", "figures")

        # Accuracy chart
        acc_chart_path = os.path.join(figures_dir, "ablation_accuracy_bars.png")
        create_comparative_chart(
            df, acc_chart_path, "acc", "Accuracy by Parameter Configuration"
        )

        # Compositional accuracy chart
        compo_chart_path = os.path.join(figures_dir, "ablation_compo_bars.png")
        create_comparative_chart(
            df,
            compo_chart_path,
            "compo_acc",
            "Compositional Accuracy by Parameter Configuration",
        )

        # Heatmap
        heatmap_path = os.path.join(figures_dir, "ablation_heatmap.png")
        create_heatmap_chart(
            df,
            heatmap_path,
            "acc",
            "Accuracy Heatmap: Vocabulary Size vs Channel Noise",
        )

        report_info.update(
            {
                "accuracy_chart": acc_chart_path,
                "compo_chart": compo_chart_path,
                "heatmap": heatmap_path,
            }
        )

    logger.info("Ablation study report created successfully")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Summary: {summary_path}")

    return report_info
