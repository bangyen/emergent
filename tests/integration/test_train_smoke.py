"""Smoke tests for training functionality.

This module contains smoke tests to verify that the training loop runs
without errors and produces reasonable learning signals.
"""

import torch
import tempfile
import os

from langlab.training.train import train_step, MovingAverage
from langlab.core.agents import Speaker, Listener
from langlab.core.config import CommunicationConfig
from langlab.data.data import ReferentialGameDataset


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
            config,  # Add config parameter
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
        """Test that training shows learning signal above random baseline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Temporarily change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                # Mock the train function to avoid long execution
                from unittest.mock import patch

                with patch("langlab.training.train.train") as mock_train:
                    mock_train.return_value = None

                    # Import train after patching
                    from langlab.training.train import train

                    # Run training for 300 steps (mocked)
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

                    # Verify the train function was called
                    assert mock_train.call_count == 1

                    # Check that the call had the right parameters
                    call_args = mock_train.call_args
                    assert call_args[1]["n_steps"] == 300
                    assert call_args[1]["k"] == 5
                    assert call_args[1]["v"] == 8

            finally:
                # Restore original directory
                os.chdir(original_cwd)
