"""Smoke tests for training functionality.

This module contains smoke tests to verify that the training loop runs
without errors and produces reasonable learning signals.
"""

import torch
import tempfile
import os

from langlab.train import train_step, train, MovingAverage
from langlab.agents import Speaker, Listener
from langlab.config import CommunicationConfig
from langlab.data import ReferentialGameDataset


class TestTrainingSmoke:
    """Test that training runs without errors and shows learning signals."""

    def test_one_step_runs(self) -> None:
        """Test that training for 10 steps on tiny dataset returns metrics dict."""
        # Set up minimal configuration
        config = CommunicationConfig(
            vocabulary_size=5, message_length=1, hidden_size=32, seed=42
        )

        # Create agents
        speaker = Speaker(config)
        listener = Listener(config)

        # Create optimizers
        speaker_optimizer = torch.optim.Adam(speaker.parameters(), lr=1e-3)
        listener_optimizer = torch.optim.Adam(listener.parameters(), lr=1e-3)

        # Create baseline
        speaker_baseline = MovingAverage(window_size=10)

        # Create tiny dataset
        dataset = ReferentialGameDataset(n_scenes=20, k=3, seed=42)

        # Test single step
        batch = dataset[0]
        # Convert single sample to batch
        scene_tensor = batch[0].unsqueeze(0)  # Add batch dimension
        target_idx = torch.tensor([batch[1]])
        candidate_objects = batch[2].unsqueeze(0)  # Add batch dimension
        batch_tuple = (scene_tensor, target_idx, candidate_objects)

        metrics = train_step(
            speaker,
            listener,
            batch_tuple,
            speaker_optimizer,
            listener_optimizer,
            speaker_baseline,
            lambda_speaker=1.0,
        )

        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert "total_loss" in metrics
        assert "listener_loss" in metrics
        assert "speaker_loss" in metrics
        assert "accuracy" in metrics
        assert "baseline" in metrics

        # Verify all metrics are scalars
        for key, value in metrics.items():
            assert isinstance(value, float)
            assert not torch.is_tensor(value)

    def test_learning_signal(self) -> None:
        """Test that 300 steps show learning signal above random baseline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Temporarily change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                # Run training for 300 steps
                train(
                    n_steps=300,
                    k=5,
                    v=8,
                    message_length=1,
                    seed=123,
                    log_every=50,
                    eval_every=100,
                    batch_size=16,
                    learning_rate=1e-3,
                    hidden_size=32,
                )

                # Check that metrics file was created
                metrics_file = os.path.join(temp_dir, "outputs", "logs", "metrics.csv")
                assert os.path.exists(metrics_file)

                # Read metrics and check for learning signal
                import csv

                metrics_data: list = []
                with open(metrics_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        metrics_data.append(row)

                assert len(metrics_data) > 0

                # Check that final accuracy is above random baseline
                # For K=5, random baseline is 0.2
                random_baseline = 0.2
                final_accuracy = float(metrics_data[-1]["accuracy"])

                # We expect some learning signal (accuracy > random baseline + margin)
                margin = 0.10  # Allow for some learning
                assert (
                    final_accuracy > random_baseline + margin
                ), f"Final accuracy {final_accuracy:.3f} should be > {random_baseline + margin:.3f}"

                # Check that accuracy generally improves over time
                early_accuracy = float(metrics_data[0]["accuracy"])
                late_accuracy = float(metrics_data[-1]["accuracy"])

                # At minimum, accuracy shouldn't get significantly worse
                assert (
                    late_accuracy >= early_accuracy - 0.1
                ), f"Accuracy shouldn't degrade significantly: {early_accuracy:.3f} -> {late_accuracy:.3f}"

            finally:
                os.chdir(original_cwd)
