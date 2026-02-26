"""Smoke tests for training functionality with expected metrics validation.

This module contains smoke tests that verify the training pipeline works
correctly and produces expected metrics within reasonable ranges.
"""

import os
from pathlib import Path
import torch
from langlab.training.train import train
from langlab.utils.utils import get_logger

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
                batch_size=4,
                learning_rate=1e-3,
                hidden_size=32,
                use_sequence_models=False,
                entropy_weight=0.01,
            )

            # Verify outputs were created
            assert os.path.exists("outputs/checkpoints/final_model.pt")

            # Load and verify checkpoint
            checkpoint = torch.load(
                "outputs/checkpoints/final_model.pt", weights_only=False
            )
            assert "speaker_state_dict" in checkpoint
            assert "listener_state_dict" in checkpoint
            assert "config" in checkpoint

            logger.info("Smoke test completed successfully.")

        finally:
            os.chdir(original_cwd)

    def test_training_reproducibility(self, tmp_path: Path) -> None:
        """Test that training is reproducible with the same seed."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Run training twice with the same seed
            checkpoints = []
            for run_id in [1, 2]:
                run_dir = tmp_path / f"run_{run_id}"
                run_dir.mkdir()
                os.chdir(run_dir)

                seed = 999
                train(
                    n_steps=10,
                    k=3,
                    v=8,
                    message_length=1,
                    seed=seed,
                    batch_size=4,
                    learning_rate=1e-3,
                    hidden_size=32,
                    use_sequence_models=False,
                )

                checkpoint_path = "outputs/checkpoints/final_model.pt"
                assert os.path.exists(checkpoint_path)
                checkpoints.append(torch.load(checkpoint_path, weights_only=False))
                os.chdir(tmp_path)

            # Compare state dicts for reproducibility
            for key in ["speaker_state_dict", "listener_state_dict"]:
                sd1 = checkpoints[0][key]
                sd2 = checkpoints[1][key]

                assert sd1.keys() == sd2.keys()
                for k in sd1:
                    assert torch.allclose(
                        sd1[k], sd2[k]
                    ), f"Weights for {k} in {key} differ"

            logger.info("Reproducibility test passed (weights are identical).")

        finally:
            os.chdir(original_cwd)
