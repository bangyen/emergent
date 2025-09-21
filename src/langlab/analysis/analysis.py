"""Analysis module for language emergence experiments.

This module provides tools for analyzing emergent language patterns,
including Zipf's law analysis and token frequency distributions.
"""

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
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
    except Exception as e:
        print(f"Error loading logs from {log_path}: {e}")
        return pd.DataFrame()
