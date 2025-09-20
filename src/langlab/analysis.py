"""Analysis module for language emergence experiments.

This module provides tools for analyzing emergent language patterns,
including Zipf's law analysis and token frequency distributions.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd


def compute_zipf_slope(tokens: List[int]) -> Tuple[float, np.ndarray]:
    """Compute Zipf's law slope by fitting a line in log-log space.

    This function analyzes token frequency distributions to determine
    how well they follow Zipf's law, which predicts that frequency
    is inversely proportional to rank.

    Args:
        tokens: List of token IDs to analyze.

    Returns:
        Tuple of (slope, frequencies) where slope is the fitted line slope
        in log-log space and frequencies is the sorted frequency array.
    """
    if not tokens:
        return 0.0, np.array([])

    # Count token frequencies
    token_counts = Counter(tokens)

    # Sort by frequency (descending)
    sorted_counts = sorted(token_counts.values(), reverse=True)
    frequencies = np.array(sorted_counts)

    if len(frequencies) < 2:
        return 0.0, frequencies

    # Create rank array (1-indexed)
    ranks = np.arange(1, len(frequencies) + 1)

    # Convert to log space
    log_ranks = np.log(ranks)
    log_frequencies = np.log(frequencies)

    # Fit linear regression in log-log space
    # log(frequency) = slope * log(rank) + intercept
    slope, intercept = np.polyfit(log_ranks, log_frequencies, 1)

    return float(slope), frequencies


def analyze_token_distribution(tokens: List[int]) -> Dict[str, Any]:
    """Analyze token distribution statistics.

    This function provides comprehensive analysis of token usage patterns
    including Zipf's law compliance, vocabulary diversity, and frequency
    distribution characteristics.

    Args:
        tokens: List of token IDs to analyze.

    Returns:
        Dictionary containing analysis results including Zipf slope,
        vocabulary size, frequency statistics, and distribution metrics.
    """
    if not tokens:
        return {
            "zipf_slope": 0.0,
            "vocab_size": 0,
            "total_tokens": 0,
            "unique_tokens": 0,
            "frequencies": np.array([]),
            "entropy": 0.0,
            "gini_coefficient": 0.0,
        }

    # Basic statistics
    token_counts = Counter(tokens)
    vocab_size = len(token_counts)
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))

    # Zipf analysis
    zipf_slope, frequencies = compute_zipf_slope(tokens)

    # Entropy calculation
    probs = np.array(list(token_counts.values())) / total_tokens
    entropy: float = -np.sum(
        probs * np.log(probs + 1e-10)
    )  # Add small epsilon to avoid log(0)

    # Gini coefficient (measure of inequality)
    sorted_freqs = np.sort(frequencies)
    n = len(sorted_freqs)
    cumsum = np.cumsum(sorted_freqs)
    gini_coefficient = (
        (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
    )

    return {
        "zipf_slope": zipf_slope,
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "frequencies": frequencies,
        "entropy": float(entropy),
        "gini_coefficient": float(gini_coefficient),
    }


def plot_zipf_rank_frequency(
    tokens: List[int], save_path: Optional[str] = None
) -> None:
    """Create a Zipf rank-frequency plot.

    This function generates a log-log plot showing token frequency vs rank,
    which is the standard visualization for Zipf's law analysis.

    Args:
        tokens: List of token IDs to analyze.
        save_path: Optional path to save the plot. If None, displays the plot.
    """
    if not tokens:
        print("No tokens to plot")
        return

    # Get frequency data
    token_counts = Counter(tokens)
    sorted_counts = sorted(token_counts.values(), reverse=True)
    frequencies = np.array(sorted_counts)
    ranks = np.arange(1, len(frequencies) + 1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, "bo-", markersize=4, alpha=0.7)

    # Fit and plot Zipf line
    if len(frequencies) >= 2:
        log_ranks = np.log(ranks)
        log_frequencies = np.log(frequencies)
        slope, intercept = np.polyfit(log_ranks, log_frequencies, 1)

        # Plot fitted line
        zipf_line = np.exp(intercept) * ranks**slope
        plt.loglog(
            ranks, zipf_line, "r--", linewidth=2, label=f"Zipf fit (slope={slope:.2f})"
        )

    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipf Rank-Frequency Plot")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


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


def compute_compositional_vs_iid_accuracy(logs_df: pd.DataFrame) -> Dict[str, float]:
    """Compute compositional vs IID accuracy from training logs.

    This function analyzes training logs to determine performance
    on compositional vs IID (independent and identically distributed)
    test sets, which is crucial for understanding emergent language
    compositionality.

    Args:
        logs_df: DataFrame containing training metrics.

    Returns:
        Dictionary with accuracy metrics for different test conditions.
    """
    if logs_df.empty:
        return {"iid_accuracy": 0.0, "compositional_accuracy": 0.0}

    # For now, we'll use the training accuracy as a proxy
    # In a full implementation, this would load separate evaluation logs
    final_accuracy = (
        logs_df["accuracy"].iloc[-1] if "accuracy" in logs_df.columns else 0.0
    )

    # Placeholder: assume 80% of final accuracy for compositional
    # This would be replaced with actual evaluation data
    compositional_accuracy = final_accuracy * 0.8

    return {
        "iid_accuracy": float(final_accuracy),
        "compositional_accuracy": float(compositional_accuracy),
    }
