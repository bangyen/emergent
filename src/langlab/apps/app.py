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
    analyze_token_distribution,
    load_training_logs,
    compute_compositional_vs_iid_accuracy,
)
from langlab.data.world import sample_scene, COLORS, SHAPES, SIZES


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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(logs_df["step"], logs_df["accuracy"], "b-", linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training Accuracy Over Time")
    ax.grid(True, alpha=0.3)

    # Add baseline if available
    if "baseline" in logs_df.columns:
        ax.plot(
            logs_df["step"], logs_df["baseline"], "r--", alpha=0.7, label="Baseline"
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
        # Approximate entropy from speaker loss (higher loss = higher entropy)
        entropy_approx = logs_df["speaker_loss"].rolling(window=10).mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(logs_df["step"], entropy_approx, "g-", linewidth=2)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Approximate Entropy")
        ax.set_title("Message Entropy Over Time")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close()
    else:
        st.warning("No entropy data available")


def create_message_length_plot(logs_df: pd.DataFrame) -> None:
    """Create message length over time plot.

    Args:
        logs_df: DataFrame containing training metrics.
    """
    if logs_df.empty:
        st.warning("No training data available")
        return

    # Check if message length data is available
    if "avg_message_length" not in logs_df.columns:
        st.info("Message length analysis requires additional logging data")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot average message length over time
    ax.plot(
        logs_df["step"],
        logs_df["avg_message_length"],
        "purple",
        linewidth=2,
        label="Average Message Length",
    )

    # If we have std data, show it as error bars
    if (
        "message_length_std" in logs_df.columns
        and logs_df["message_length_std"].sum() > 0
    ):
        ax.fill_between(
            logs_df["step"],
            logs_df["avg_message_length"] - logs_df["message_length_std"],
            logs_df["avg_message_length"] + logs_df["message_length_std"],
            alpha=0.3,
            color="purple",
            label="±1 std",
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Message Length")
    ax.set_title("Message Length Over Time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add some statistics
    final_length = logs_df["avg_message_length"].iloc[-1]
    ax.text(
        0.02,
        0.98,
        f"Final Length: {final_length:.2f}",
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
    from collections import Counter

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


def create_compositional_vs_iid_plot(logs_df: pd.DataFrame) -> None:
    """Create compositional vs IID accuracy comparison plot.

    Args:
        logs_df: DataFrame containing training metrics.
    """
    if logs_df.empty:
        st.warning("No training data available")
        return

    # Compute accuracies
    accuracies = compute_compositional_vs_iid_accuracy(logs_df)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ["IID Accuracy", "Compositional Accuracy"]
    values = [accuracies["iid_accuracy"], accuracies["compositional_accuracy"]]

    bars = ax.bar(categories, values, color=["skyblue", "lightcoral"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Compositional vs IID Accuracy")
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    st.pyplot(fig)
    plt.close()


def interactive_probe(logs_df: pd.DataFrame) -> None:
    """Interactive probe for examining agent behavior.

    Args:
        logs_df: DataFrame containing training metrics.
    """
    st.subheader("Interactive Agent Probe")

    # Checkpoint selection
    checkpoint_dir = "outputs/checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        if checkpoint_files:
            selected_checkpoint = st.selectbox(
                "Select Model Checkpoint", checkpoint_files
            )
            checkpoint_path = os.path.join(checkpoint_dir, selected_checkpoint)
        else:
            st.warning("No checkpoint files found")
            checkpoint_path = None
    else:
        st.warning("Checkpoint directory not found")
        checkpoint_path = None

    # Object selection (for future use - currently using random scenes)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.selectbox("Color", COLORS)

    with col2:
        st.selectbox("Shape", SHAPES)

    with col3:
        st.selectbox("Size", SIZES)

    # Generate scene
    if st.button("Generate Scene"):
        scene_objects, target_idx = sample_scene(k=5, seed=42)

        st.write("**Scene Objects:**")
        for i, obj in enumerate(scene_objects):
            marker = " (TARGET)" if i == target_idx else ""
            st.write(f"{i}: {obj['color']} {obj['size']} {obj['shape']}{marker}")

        # Load model and generate predictions if checkpoint is available
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                # Load checkpoint
                checkpoint = load_checkpoint(checkpoint_path)
                if checkpoint is None:
                    st.error("Failed to load checkpoint")
                    return

                # Extract model configurations from checkpoint
                comm_config = checkpoint.get("config")
                if comm_config is None:
                    st.error("No configuration found in checkpoint")
                    return

                # Create models (we'll use the basic Speaker/Listener for now)
                from langlab.core.agents import Speaker, Listener

                # Create models
                speaker = Speaker(comm_config)
                listener = Listener(comm_config)

                # Load model states
                speaker.load_state_dict(checkpoint["speaker_state_dict"])
                listener.load_state_dict(checkpoint["listener_state_dict"])

                speaker.eval()
                listener.eval()

                # Convert scene to tensor format
                from langlab.data.world import encode_object
                import torch

                # Encode all objects in the scene
                encoded_objects = []
                for obj in scene_objects:
                    encoded_obj = encode_object(obj)
                    encoded_objects.append(encoded_obj)

                scene_tensor = torch.stack(encoded_objects).unsqueeze(
                    0
                )  # Add batch dimension
                target_object = scene_tensor[0, target_idx].unsqueeze(
                    0
                )  # Target object

                # Generate message
                with torch.no_grad():
                    speaker_logits, message_tokens, _, _ = speaker(target_object)

                    # Convert message tokens to readable format
                    message_str = " ".join(
                        [str(token.item()) for token in message_tokens[0]]
                    )

                    st.write(f"**Generated Message:** `{message_str}`")

                    # Listener prediction
                    listener_logits = listener(message_tokens, scene_tensor)
                    predicted_idx = torch.argmax(listener_logits, dim=1).item()

                    # Display results
                    st.write("**Listener Prediction:**")
                    if predicted_idx == target_idx:
                        st.success(f"✅ Correct! Predicted object {predicted_idx}")
                    else:
                        st.error(
                            f"❌ Incorrect! Predicted object {predicted_idx}, actual target was {target_idx}"
                        )

                    # Show prediction probabilities
                    probs = torch.softmax(listener_logits, dim=1)[0]
                    st.write("**Prediction Probabilities:**")
                    for i, prob in enumerate(probs):
                        marker = (
                            " ← PREDICTED"
                            if i == predicted_idx
                            else " ← TARGET" if i == target_idx else ""
                        )
                        st.write(f"Object {i}: {prob.item():.3f}{marker}")

            except Exception as e:
                st.error(f"Error loading model or generating predictions: {e}")
                st.info(
                    "This might be due to model architecture differences. Try a different checkpoint."
                )
        else:
            st.info(
                "Select a checkpoint to enable message generation and listener prediction"
            )


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
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Training Metrics",
                "Language Analysis",
                "Compositionality",
                "Interactive Probe",
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

            st.subheader("Message Length Over Time")
            create_message_length_plot(logs_df)

        with tab2:
            st.header("Language Analysis")

            # Generate sample tokens for Zipf analysis
            # In a real implementation, this would come from actual message logs
            st.subheader("Zipf Rank-Frequency Analysis")

            # Create sample token data for demonstration
            np.random.seed(42)
            vocab_size = 10
            n_tokens = 1000

            # Generate tokens following a Zipf-like distribution
            ranks = np.arange(1, vocab_size + 1)
            frequencies = 1.0 / ranks  # Perfect Zipf distribution
            probs = frequencies / np.sum(frequencies)

            sample_tokens = np.random.choice(vocab_size, size=n_tokens, p=probs)

            # Analyze and display results
            analysis = analyze_token_distribution(sample_tokens.tolist())

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Zipf Slope", f"{analysis['zipf_slope']:.3f}")
                st.metric("Vocabulary Size", analysis["vocab_size"])
                st.metric("Entropy", f"{analysis['entropy']:.3f}")

            with col2:
                st.metric("Total Tokens", analysis["total_tokens"])
                st.metric("Unique Tokens", analysis["unique_tokens"])
                st.metric("Gini Coefficient", f"{analysis['gini_coefficient']:.3f}")

            # Create Zipf plot
            create_zipf_plot(sample_tokens.tolist())

        with tab3:
            st.header("Compositionality Analysis")
            create_compositional_vs_iid_plot(logs_df)

        with tab4:
            st.header("Interactive Agent Probe")
            interactive_probe(logs_df)

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
