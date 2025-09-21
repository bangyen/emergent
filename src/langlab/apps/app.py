"""Streamlit dashboard for language emergence experiments.

This module provides an interactive web dashboard for visualizing
language emergence experiments, including training metrics, Zipf analysis,
and interactive probing of agent behavior.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import os
import torch

from langlab.analysis.analysis import (
    load_training_logs,
)
from collections import Counter
from typing import Tuple


def load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Load a model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        Dictionary containing model state and metadata, or None if loading fails.
    """
    try:
        checkpoint: Dict[str, Any] = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        return checkpoint
    except Exception as e:
        st.error(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None


def create_accuracy_plot(logs_df: pd.DataFrame) -> None:
    """Create accuracy over time plot.

    Args:
        logs_df: DataFrame containing training metrics.
    """
    if logs_df.empty or "accuracy" not in logs_df.columns:
        st.warning("No accuracy data available")
        return

    # Handle different column names for step/episode
    step_col = "step" if "step" in logs_df.columns else "episode"
    if step_col not in logs_df.columns:
        st.warning("No step/episode column found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(logs_df[step_col], logs_df["accuracy"], "b-", linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training Accuracy Over Time")
    ax.grid(True, alpha=0.3)

    # Add baseline if available
    if "baseline" in logs_df.columns:
        ax.plot(
            logs_df[step_col], logs_df["baseline"], "r--", alpha=0.7, label="Baseline"
        )
        ax.legend()

    st.pyplot(fig)
    plt.close()


def create_entropy_plot(logs_df: pd.DataFrame) -> None:
    """Create entropy over time plot.

    Args:
        logs_df: DataFrame containing training metrics.
    """
    if logs_df.empty:
        st.warning("No training data available")
        return

    # For now, we'll compute entropy from loss data
    # In a full implementation, this would come from the logs
    if "speaker_loss" in logs_df.columns:
        # Handle different column names for step/episode
        step_col = "step" if "step" in logs_df.columns else "episode"
        if step_col not in logs_df.columns:
            st.warning("No step/episode column found")
            return

        # Approximate entropy from speaker loss (higher loss = higher entropy)
        entropy_approx = logs_df["speaker_loss"].rolling(window=10).mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(logs_df[step_col], entropy_approx, "g-", linewidth=2)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Approximate Entropy")
        ax.set_title("Message Entropy Over Time")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()
    else:
        st.warning("No entropy data available")


def create_loss_plot(logs_df: pd.DataFrame) -> None:
    """Create training loss over time plot.

    Args:
        logs_df: DataFrame containing training metrics.
    """
    if logs_df.empty:
        st.warning("No training data available")
        return

    # Check if loss data is available
    if "total_loss" not in logs_df.columns:
        st.info("Loss analysis requires additional logging data")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Handle different column names for step/episode
    step_col = "step" if "step" in logs_df.columns else "episode"
    if step_col not in logs_df.columns:
        st.warning("No step/episode column found")
        return

    # Plot total loss over time
    ax.plot(
        logs_df[step_col],
        logs_df["total_loss"],
        "red",
        linewidth=2,
        label="Total Loss",
    )

    # Plot individual losses if available
    if "speaker_loss" in logs_df.columns:
        ax.plot(
            logs_df[step_col],
            logs_df["speaker_loss"],
            "blue",
            linewidth=1,
            alpha=0.7,
            label="Speaker Loss",
        )

    if "listener_loss" in logs_df.columns:
        ax.plot(
            logs_df[step_col],
            logs_df["listener_loss"],
            "green",
            linewidth=1,
            alpha=0.7,
            label="Listener Loss",
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Over Time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add some statistics
    final_loss = logs_df["total_loss"].iloc[-1]
    min_loss = logs_df["total_loss"].min()
    ax.text(
        0.02,
        0.98,
        f"Final Loss: {final_loss:.4f}\nMin Loss: {min_loss:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    st.pyplot(fig)
    plt.close()


def create_zipf_plot(tokens: List[int]) -> None:
    """Create Zipf rank-frequency plot.

    Args:
        tokens: List of token IDs to analyze.
    """
    if not tokens:
        st.warning("No token data available for Zipf analysis")
        return

    # Generate the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get frequency data
    token_counts = Counter(tokens)
    sorted_counts = sorted(token_counts.values(), reverse=True)
    frequencies = np.array(sorted_counts)
    ranks = np.arange(1, len(frequencies) + 1)

    # Create the plot
    ax.loglog(ranks, frequencies, "bo-", markersize=4, alpha=0.7)

    # Fit and plot Zipf line
    if len(frequencies) >= 2:
        log_ranks = np.log(ranks)
        log_frequencies = np.log(frequencies)
        slope, intercept = np.polyfit(log_ranks, log_frequencies, 1)

        # Plot fitted line
        zipf_line = np.exp(intercept) * ranks**slope
        ax.loglog(
            ranks, zipf_line, "r--", linewidth=2, label=f"Zipf fit (slope={slope:.2f})"
        )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.set_title("Zipf Rank-Frequency Plot")
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)
    plt.close()


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


def main() -> None:
    """Main Streamlit application entry point.

    This function sets up the Streamlit dashboard with all visualization
    components and interactive features for analyzing language emergence.
    """
    st.set_page_config(
        page_title="Language Emergence Dashboard", page_icon="🧠", layout="wide"
    )

    st.title("🧠 Language Emergence Dashboard")
    st.markdown("Interactive visualization of emergent language experiments")

    # Sidebar for file selection
    st.sidebar.header("Data Selection")

    # Check for available log files
    log_dir = "outputs/logs"
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith(".csv")]
        if log_files:
            selected_log = st.sidebar.selectbox("Select log file", log_files)
            log_path = os.path.join(log_dir, selected_log)
        else:
            st.sidebar.warning("No CSV log files found")
            log_path = None
    else:
        st.sidebar.warning("Log directory not found")
        log_path = None

    # Load data
    if log_path and os.path.exists(log_path):
        logs_df = load_training_logs(log_path)

        if not logs_df.empty:
            st.sidebar.success(f"Loaded {len(logs_df)} training steps")

            # Display basic stats
            st.sidebar.subheader("Training Summary")
            st.sidebar.metric("Total Steps", len(logs_df))
            if "accuracy" in logs_df.columns:
                final_acc = logs_df["accuracy"].iloc[-1]
                st.sidebar.metric("Final Accuracy", f"{final_acc:.3f}")
        else:
            st.error("Failed to load training data")
            logs_df = pd.DataFrame()
    else:
        st.warning("No training logs available. Run training first.")
        logs_df = pd.DataFrame()

    # Main dashboard content
    if not logs_df.empty:
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(
            [
                "Training Metrics",
                "Language Analysis",
            ]
        )

        with tab1:
            st.header("Training Metrics")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Accuracy Over Time")
                create_accuracy_plot(logs_df)

            with col2:
                st.subheader("Entropy Over Time")
                create_entropy_plot(logs_df)

            st.subheader("Training Loss Over Time")
            create_loss_plot(logs_df)

        with tab2:
            st.header("Language Analysis")

            # Check if we have message logs available
            message_logs_path = "outputs/logs/message_logs.csv"
            if os.path.exists(message_logs_path):
                try:
                    # Load message logs
                    message_logs_df = pd.read_csv(message_logs_path)

                    if not message_logs_df.empty:
                        st.subheader("Zipf Rank-Frequency Analysis")

                        # Extract all message tokens
                        all_tokens = []
                        for _, row in message_logs_df.iterrows():
                            # Parse message tokens (assuming they're stored as space-separated strings)
                            if "message_tokens" in row and pd.notna(
                                row["message_tokens"]
                            ):
                                tokens = [
                                    int(x) for x in str(row["message_tokens"]).split()
                                ]
                                all_tokens.extend(tokens)

                        if all_tokens:
                            # Create Zipf plot
                            create_zipf_plot(all_tokens)

                            # Display analysis statistics
                            st.subheader("Analysis Statistics")
                            analysis = analyze_token_distribution(all_tokens)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Zipf Slope", f"{analysis['zipf_slope']:.3f}")
                                st.metric("Vocabulary Size", analysis["vocab_size"])
                                st.metric("Entropy", f"{analysis['entropy']:.3f}")

                            with col2:
                                st.metric("Total Tokens", analysis["total_tokens"])
                                st.metric("Unique Tokens", analysis["unique_tokens"])
                                st.metric(
                                    "Gini Coefficient",
                                    f"{analysis['gini_coefficient']:.3f}",
                                )
                        else:
                            st.warning("No message tokens found in logs")
                    else:
                        st.warning("Message logs file is empty")

                except Exception as e:
                    st.error(f"Error loading message logs: {e}")
                    st.info("Message logs may not be in the expected format")
            else:
                st.info(
                    """
                **Message logs not found**
                
                To enable language analysis, the training pipeline needs to be modified to save message tokens.
                Currently, only aggregate metrics are saved during training.
                
                **To enable this feature:**
                1. Modify the training code to save message tokens to `outputs/logs/message_logs.csv`
                2. The message logs should contain columns: `step`, `message_tokens`
                3. Message tokens should be stored as space-separated integers
                """
                )

                # Show placeholder analysis
                st.subheader("Message Length Analysis (Available)")
                if "avg_message_length" in logs_df.columns:
                    # Create message length plot
                    fig, ax = plt.subplots(figsize=(10, 6))

                    step_col = "step" if "step" in logs_df.columns else "episode"
                    if step_col in logs_df.columns:
                        ax.plot(
                            logs_df[step_col],
                            logs_df["avg_message_length"],
                            "b-",
                            linewidth=2,
                            label="Average Message Length",
                        )
                        if "message_length_std" in logs_df.columns:
                            ax.fill_between(
                                logs_df[step_col],
                                logs_df["avg_message_length"]
                                - logs_df["message_length_std"],
                                logs_df["avg_message_length"]
                                + logs_df["message_length_std"],
                                alpha=0.3,
                                color="blue",
                                label="±1 Std Dev",
                            )

                        ax.set_xlabel("Training Step")
                        ax.set_ylabel("Message Length")
                        ax.set_title("Message Length Evolution During Training")
                        ax.grid(True, alpha=0.3)
                        ax.legend()

                        st.pyplot(fig)
                        plt.close()

                        # Display message length statistics
                        st.subheader("Message Length Statistics")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric(
                                "Final Avg Length",
                                f"{logs_df['avg_message_length'].iloc[-1]:.2f}",
                            )
                            st.metric(
                                "Min Length",
                                f"{logs_df['avg_message_length'].min():.2f}",
                            )
                            st.metric(
                                "Max Length",
                                f"{logs_df['avg_message_length'].max():.2f}",
                            )

                        with col2:
                            st.metric(
                                "Length Trend",
                                (
                                    "Increasing"
                                    if logs_df["avg_message_length"].iloc[-1]
                                    > logs_df["avg_message_length"].iloc[0]
                                    else "Decreasing"
                                ),
                            )
                            st.metric(
                                "Length Stability",
                                (
                                    f"{logs_df['message_length_std'].mean():.2f}"
                                    if "message_length_std" in logs_df.columns
                                    else "N/A"
                                ),
                            )
                            st.metric("Training Steps", len(logs_df))
                    else:
                        st.warning("No step/episode column found in training data")
                else:
                    st.warning("No message length data available in training logs")

    else:
        st.info(
            """
        ## Getting Started
        
        To use this dashboard:
        
        1. **Run Training**: Use `langlab train` to generate training logs
        2. **View Results**: The dashboard will automatically load available log files
        3. **Explore**: Use the tabs above to explore different aspects of language emergence
        
        ### Available Commands:
        - `langlab train --steps 1000` - Train agents for 1000 steps
        - `langlab dash` - Launch this dashboard
        """
        )


if __name__ == "__main__":
    main()
