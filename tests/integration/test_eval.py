"""Tests for evaluation functionality.

This module contains tests for the evaluation functions, ensuring
proper model evaluation and metrics computation.
"""

import pytest
import torch
import tempfile
import os

from src.langlab.analysis.eval import evaluate, evaluate_all_splits
from src.langlab.core.config import CommunicationConfig
from src.langlab.core.agents import Speaker, Listener


class TestEvaluation:
    """Test evaluation functionality."""

    def test_eval_api(self) -> None:
        """Ensure eval returns dict with keys {'acc'} and acc in [0,1]."""
        # Create a dummy model checkpoint
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=1,
            hidden_size=32,
            seed=42,
        )

        speaker = Speaker(config)
        listener = Listener(config)

        # Create dummy checkpoint
        checkpoint = {
            "step": 100,
            "speaker_state_dict": speaker.state_dict(),
            "listener_state_dict": listener.state_dict(),
            "speaker_optimizer_state_dict": {},
            "listener_optimizer_state_dict": {},
            "config": config,
            "metrics": {"accuracy": 0.5},
        }

        # Save checkpoint to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(checkpoint, f.name)
            checkpoint_path = f.name

        try:
            # Test evaluation
            results = evaluate(
                model_path=checkpoint_path,
                split="train",
                n_scenes=10,
                k=3,
                batch_size=5,
            )

            # Check return format
            assert isinstance(results, dict), "Results should be a dictionary"
            assert "acc" in results, "Results should contain 'acc' key"

            accuracy = results["acc"]
            assert isinstance(accuracy, float), "Accuracy should be a float"
            assert 0.0 <= accuracy <= 1.0, f"Accuracy {accuracy} should be in [0,1]"

        finally:
            # Clean up temporary file
            os.unlink(checkpoint_path)

    def test_eval_compositional_splits(self) -> None:
        """Test evaluation on compositional splits."""
        # Create a dummy model checkpoint
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=1,
            hidden_size=32,
            seed=42,
        )

        speaker = Speaker(config)
        listener = Listener(config)

        # Create dummy checkpoint
        checkpoint = {
            "step": 100,
            "speaker_state_dict": speaker.state_dict(),
            "listener_state_dict": listener.state_dict(),
            "speaker_optimizer_state_dict": {},
            "listener_optimizer_state_dict": {},
            "config": config,
            "metrics": {"accuracy": 0.5},
        }

        # Save checkpoint to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(checkpoint, f.name)
            checkpoint_path = f.name

        try:
            heldout_pairs = [("blue", "triangle")]

            # Test evaluation on different splits
            for split in ["train", "iid", "compo"]:
                results = evaluate(
                    model_path=checkpoint_path,
                    split=split,
                    heldout_pairs=heldout_pairs,
                    n_scenes=10,
                    k=3,
                    batch_size=5,
                )

                assert isinstance(
                    results, dict
                ), f"Results for {split} should be a dictionary"
                assert "acc" in results, f"Results for {split} should contain 'acc' key"

                accuracy = results["acc"]
                assert isinstance(
                    accuracy, float
                ), f"Accuracy for {split} should be a float"
                assert (
                    0.0 <= accuracy <= 1.0
                ), f"Accuracy {accuracy} for {split} should be in [0,1]"

        finally:
            # Clean up temporary file
            os.unlink(checkpoint_path)

    def test_eval_all_splits(self) -> None:
        """Test evaluation on all splits."""
        # Create a dummy model checkpoint
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=1,
            hidden_size=32,
            seed=42,
        )

        speaker = Speaker(config)
        listener = Listener(config)

        # Create dummy checkpoint
        checkpoint = {
            "step": 100,
            "speaker_state_dict": speaker.state_dict(),
            "listener_state_dict": listener.state_dict(),
            "speaker_optimizer_state_dict": {},
            "listener_optimizer_state_dict": {},
            "config": config,
            "metrics": {"accuracy": 0.5},
        }

        # Save checkpoint to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(checkpoint, f.name)
            checkpoint_path = f.name

        try:
            heldout_pairs = [("blue", "triangle")]

            # Test evaluation on all splits
            results = evaluate_all_splits(
                model_path=checkpoint_path,
                heldout_pairs=heldout_pairs,
                n_scenes=10,
                k=3,
                batch_size=5,
            )

            # Check return format
            assert isinstance(results, dict), "Results should be a dictionary"

            expected_splits = ["train", "iid", "compo"]
            for split in expected_splits:
                assert split in results, f"Results should contain {split} split"

                split_results = results[split]
                assert isinstance(
                    split_results, dict
                ), f"Results for {split} should be a dictionary"
                assert (
                    "acc" in split_results
                ), f"Results for {split} should contain 'acc' key"

                accuracy = split_results["acc"]
                assert isinstance(
                    accuracy, float
                ), f"Accuracy for {split} should be a float"
                assert (
                    0.0 <= accuracy <= 1.0
                ), f"Accuracy {accuracy} for {split} should be in [0,1]"

            # Check that metrics.json was created
            assert os.path.exists(
                "outputs/metrics.json"
            ), "metrics.json should be created"

        finally:
            # Clean up temporary file
            os.unlink(checkpoint_path)

            # Clean up metrics.json if it exists
            if os.path.exists("outputs/metrics.json"):
                os.unlink("outputs/metrics.json")

    def test_eval_invalid_split(self) -> None:
        """Test evaluation with invalid split raises ValueError."""
        # Create a dummy model checkpoint
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=1,
            hidden_size=32,
            seed=42,
        )

        speaker = Speaker(config)
        listener = Listener(config)

        # Create dummy checkpoint
        checkpoint = {
            "step": 100,
            "speaker_state_dict": speaker.state_dict(),
            "listener_state_dict": listener.state_dict(),
            "speaker_optimizer_state_dict": {},
            "listener_optimizer_state_dict": {},
            "config": config,
            "metrics": {"accuracy": 0.5},
        }

        # Save checkpoint to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(checkpoint, f.name)
            checkpoint_path = f.name

        try:
            # Test evaluation with invalid split
            with pytest.raises(ValueError, match="Unsupported split"):
                evaluate(
                    model_path=checkpoint_path,
                    split="invalid_split",
                    n_scenes=10,
                    k=3,
                    batch_size=5,
                )

        finally:
            # Clean up temporary file
            os.unlink(checkpoint_path)

    def test_eval_missing_heldout_pairs(self) -> None:
        """Test evaluation with compositional split but missing heldout_pairs."""
        # Create a dummy model checkpoint
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=1,
            hidden_size=32,
            seed=42,
        )

        speaker = Speaker(config)
        listener = Listener(config)

        # Create dummy checkpoint
        checkpoint = {
            "step": 100,
            "speaker_state_dict": speaker.state_dict(),
            "listener_state_dict": listener.state_dict(),
            "speaker_optimizer_state_dict": {},
            "listener_optimizer_state_dict": {},
            "config": config,
            "metrics": {"accuracy": 0.5},
        }

        # Save checkpoint to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(checkpoint, f.name)
            checkpoint_path = f.name

        try:
            # Test evaluation with compositional split but no heldout_pairs
            with pytest.raises(ValueError, match="heldout_pairs must be provided"):
                evaluate(
                    model_path=checkpoint_path,
                    split="compo",
                    heldout_pairs=None,
                    n_scenes=10,
                    k=3,
                    batch_size=5,
                )

        finally:
            # Clean up temporary file
            os.unlink(checkpoint_path)
