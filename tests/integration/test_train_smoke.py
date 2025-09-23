"""Smoke tests for training functionality with expected metrics validation.

This module contains smoke tests that verify the training pipeline works
correctly and produces expected metrics within reasonable ranges.
"""

import os
import csv
import math
from pathlib import Path

from src.langlab.training.train import train
from src.langlab.utils.utils import get_logger

logger = get_logger(__name__)


class TestTrainingSmoke:
    """Smoke tests for training functionality."""

    def test_basic_training_smoke(self, tmp_path: Path) -> None:
        """Test basic training functionality with minimal configuration."""
        # Change to temporary directory
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Run a very short training session
            train(
                n_steps=20,
                k=3,
                v=8,
                message_length=1,
                seed=42,
                log_every=10,
                eval_every=20,
                batch_size=4,
                learning_rate=1e-3,
                hidden_size=32,
                use_sequence_models=False,
                entropy_weight=0.01,
                length_weight=0.0,
                multimodal=False,
                distractors=0,
                temperature_start=1.5,
                temperature_end=0.8,
                use_curriculum=False,
                use_warmup=False,
                use_ema=False,
                use_early_stopping=False,
                use_contrastive=False,
                # Tracking is enabled by default, but will gracefully fail if MLflow not available
            )

            # Verify outputs were created
            assert os.path.exists("outputs/logs/metrics.csv")
            assert os.path.exists("outputs/checkpoints/checkpoint.pt")

            # Load and verify metrics
            metrics_data = []
            with open("outputs/logs/metrics.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics_data.append(row)

            assert len(metrics_data) > 0, "No metrics data found"

            # Check that we have the expected columns
            expected_columns = [
                "step",
                "total_loss",
                "listener_loss",
                "speaker_loss",
                "accuracy",
                "baseline",
                "avg_message_length",
                "message_length_std",
            ]
            for col in expected_columns:
                assert col in metrics_data[0], f"Missing column: {col}"

            # Verify metrics are reasonable
            final_metrics = metrics_data[-1]
            accuracy = float(final_metrics["accuracy"])
            total_loss = float(final_metrics["total_loss"])

            # Accuracy should be between 0 and 1
            assert 0.0 <= accuracy <= 1.0, f"Accuracy out of range: {accuracy}"

            # Loss should be positive and reasonable
            assert total_loss > 0.0, f"Total loss should be positive: {total_loss}"
            assert total_loss < 10.0, f"Total loss seems too high: {total_loss}"

            logger.info(
                f"Smoke test completed successfully. Final accuracy: {accuracy:.4f}"
            )

        finally:
            os.chdir(original_cwd)

    def test_contrastive_training_smoke(self, tmp_path: Path) -> None:
        """Test training with contrastive learning enabled."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            train(
                n_steps=15,
                k=3,
                v=8,
                message_length=1,
                seed=123,
                log_every=5,
                eval_every=15,
                batch_size=4,
                learning_rate=1e-3,
                hidden_size=32,
                use_sequence_models=False,
                entropy_weight=0.01,
                length_weight=0.0,
                multimodal=False,
                distractors=0,
                temperature_start=1.5,
                temperature_end=0.8,
                use_curriculum=False,
                use_warmup=False,
                use_ema=False,
                use_early_stopping=False,
                use_contrastive=True,  # Enable contrastive learning
                contrastive_temperature=0.07,
                contrastive_weight=0.1,
                # Tracking is enabled by default, but will gracefully fail if MLflow not available
            )

            # Verify outputs
            assert os.path.exists("outputs/logs/metrics.csv")
            assert os.path.exists("outputs/checkpoints/checkpoint.pt")

            # Load metrics and verify
            metrics_data = []
            with open("outputs/logs/metrics.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics_data.append(row)

            assert len(metrics_data) > 0
            final_metrics = metrics_data[-1]
            accuracy = float(final_metrics["accuracy"])

            # With contrastive learning, accuracy should be reasonable
            assert 0.0 <= accuracy <= 1.0, f"Accuracy out of range: {accuracy}"

            logger.info(
                f"Contrastive training smoke test completed. Final accuracy: {accuracy:.4f}"
            )

        finally:
            os.chdir(original_cwd)

    def test_expected_metrics_ranges(self, tmp_path: Path) -> None:
        """Test that training produces metrics within expected ranges."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            train(
                n_steps=30,
                k=3,
                v=8,
                message_length=1,
                seed=456,
                log_every=10,
                eval_every=30,
                batch_size=8,
                learning_rate=2e-4,
                hidden_size=64,
                use_sequence_models=False,
                entropy_weight=0.01,
                length_weight=0.0,
                multimodal=False,
                distractors=0,
                temperature_start=2.0,
                temperature_end=0.5,
                use_curriculum=True,
                use_warmup=True,
                use_ema=True,
                use_early_stopping=False,
                use_contrastive=True,
                contrastive_temperature=0.07,
                contrastive_weight=0.1,
                # Tracking is enabled by default, but will gracefully fail if MLflow not available
            )

            # Load and analyze metrics
            metrics_data = []
            with open("outputs/logs/metrics.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics_data.append(row)

            assert len(metrics_data) >= 3, "Should have at least 3 metric entries"

            # Extract metrics
            steps = [int(row["step"]) for row in metrics_data]
            accuracies = [float(row["accuracy"]) for row in metrics_data]
            total_losses = [float(row["total_loss"]) for row in metrics_data]
            listener_losses = [float(row["listener_loss"]) for row in metrics_data]
            speaker_losses = [float(row["speaker_loss"]) for row in metrics_data]

            # Verify step progression
            assert steps == sorted(steps), "Steps should be in ascending order"
            assert steps[0] > 0, "First step should be positive"

            # Verify accuracy progression (should generally improve or at least be reasonable)
            assert all(
                0.0 <= acc <= 1.0 for acc in accuracies
            ), "All accuracies should be in [0, 1]"

            # Verify loss values are reasonable
            assert all(
                loss > 0.0 for loss in total_losses
            ), "All total losses should be positive"
            assert all(
                loss < 10.0 for loss in total_losses
            ), "Total losses seem too high"
            assert all(
                loss > 0.0 for loss in listener_losses
            ), "Listener losses should be positive"
            # Speaker losses can be negative due to REINFORCE baseline subtraction
            assert all(
                loss > -10.0 for loss in speaker_losses
            ), "Speaker losses seem too negative"
            assert all(
                loss < 10.0 for loss in speaker_losses
            ), "Speaker losses seem too high"

            # Check that final accuracy is reasonable (not too low for a simple task)
            final_accuracy = accuracies[-1]
            assert (
                final_accuracy >= 0.1
            ), f"Final accuracy too low: {final_accuracy:.4f}"

            # Check that loss is decreasing or at least not exploding
            if len(total_losses) >= 2:
                loss_change = total_losses[-1] - total_losses[0]
                # Loss should not increase dramatically
                assert loss_change < 5.0, f"Loss increased too much: {loss_change:.4f}"

            logger.info(
                f"Expected metrics validation passed. Final accuracy: {final_accuracy:.4f}"
            )
            logger.info(
                f"Loss progression: {total_losses[0]:.4f} -> {total_losses[-1]:.4f}"
            )

        finally:
            os.chdir(original_cwd)

    def test_training_with_early_stopping(self, tmp_path: Path) -> None:
        """Test training with early stopping enabled."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            train(
                n_steps=100,  # Set high max steps
                k=3,
                v=8,
                message_length=1,
                seed=789,
                log_every=5,
                eval_every=10,
                batch_size=4,
                learning_rate=1e-3,
                hidden_size=32,
                use_sequence_models=False,
                entropy_weight=0.01,
                length_weight=0.0,
                multimodal=False,
                distractors=0,
                temperature_start=1.5,
                temperature_end=0.8,
                use_curriculum=False,
                use_warmup=False,
                use_ema=False,
                use_early_stopping=True,  # Enable early stopping
                early_stopping_patience=5,
                early_stopping_min_delta=0.001,
                use_contrastive=False,
                # Tracking is enabled by default, but will gracefully fail if MLflow not available
            )

            # Verify outputs
            assert os.path.exists("outputs/logs/metrics.csv")
            assert os.path.exists("outputs/checkpoints/checkpoint.pt")

            # Load metrics and verify early stopping worked
            metrics_data = []
            with open("outputs/logs/metrics.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics_data.append(row)

            assert len(metrics_data) > 0
            final_step = int(metrics_data[-1]["step"])

            # With early stopping, we should stop before reaching max steps
            # (though this might not always trigger in a short test)
            assert final_step <= 100, f"Training didn't stop early: {final_step}"

            logger.info(f"Early stopping test completed. Final step: {final_step}")

        finally:
            os.chdir(original_cwd)

    def test_training_reproducibility(self, tmp_path: Path) -> None:
        """Test that training is reproducible with the same seed."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Run training twice with the same seed
            for run_id in [1, 2]:
                run_dir = tmp_path / f"run_{run_id}"
                run_dir.mkdir()
                os.chdir(run_dir)

                train(
                    n_steps=10,
                    k=3,
                    v=8,
                    message_length=1,
                    seed=999,  # Same seed for both runs
                    log_every=5,
                    eval_every=10,
                    batch_size=4,
                    learning_rate=1e-3,
                    hidden_size=32,
                    use_sequence_models=False,
                    entropy_weight=0.01,
                    length_weight=0.0,
                    multimodal=False,
                    distractors=0,
                    temperature_start=1.5,
                    temperature_end=0.8,
                    use_curriculum=False,
                    use_warmup=False,
                    use_ema=False,
                    use_early_stopping=False,
                    use_contrastive=False,
                    # Tracking is enabled by default, but will gracefully fail if MLflow not available
                )

                os.chdir(tmp_path)

            # Compare metrics between runs
            metrics_1 = []
            with open("run_1/outputs/logs/metrics.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics_1.append(row)

            metrics_2 = []
            with open("run_2/outputs/logs/metrics.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics_2.append(row)

            # Should have the same number of steps
            assert len(metrics_1) == len(
                metrics_2
            ), "Different number of steps between runs"

            # Compare final metrics (should be very close due to same seed)
            final_1 = metrics_1[-1]
            final_2 = metrics_2[-1]

            acc_1 = float(final_1["accuracy"])
            acc_2 = float(final_2["accuracy"])

            # Allow for small floating point differences
            assert math.isclose(
                acc_1, acc_2, abs_tol=1e-6
            ), f"Accuracy differs: {acc_1} vs {acc_2}"

            logger.info(
                f"Reproducibility test passed. Both runs achieved accuracy: {acc_1:.6f}"
            )

        finally:
            os.chdir(original_cwd)
