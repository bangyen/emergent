"""Analysis module for language emergence experiments.

This module provides tools for analyzing emergent language patterns,
including Zipf's law analysis and token frequency distributions.
"""

import os

import pandas as pd


def load_training_logs(log_path: str) -> pd.DataFrame:
    """Load training logs from CSV file.

    This function loads training metrics from the standard metrics.csv
    format and provides additional analysis capabilities.

    Args:
        log_path: Path to the metrics CSV file.

    Returns:
        DataFrame containing training metrics with proper data types.
    """
    try:
        df = pd.read_csv(log_path)

        # Ensure numeric columns are properly typed
        numeric_columns = [
            "step",
            "episode",  # Handle grid training logs
            "total_loss",
            "listener_loss",
            "speaker_loss",
            "accuracy",
            "baseline",
            "iid_acc",
            "compo_acc",
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
    except Exception as e:
        print(f"Error loading logs from {log_path}: {e}")
        return pd.DataFrame()


def plot_training_curve(metrics_path: str, out_path: str, window: int = 10) -> None:
    """Plot smoothed training accuracy and evaluation accuracy from metrics.csv.

    Args:
        metrics_path: Path to the metrics.csv written by training.
        out_path: Where to save the figure.
        window: Rolling-mean window (in logged rows) for training accuracy.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_training_logs(metrics_path)
    if df.empty:
        raise ValueError(f"No metrics found in {metrics_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        df["step"],
        df["accuracy"].rolling(window, min_periods=1).mean(),
        label="train (smoothed)",
        alpha=0.8,
    )
    for col, label in [("iid_acc", "eval: iid"), ("compo_acc", "eval: compo")]:
        if col in df.columns and df[col].notna().any():
            evals = df.dropna(subset=[col])
            ax.plot(evals["step"], evals[col], marker="o", label=label)

    ax.set_xlabel("step")
    ax.set_ylabel("referential accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
