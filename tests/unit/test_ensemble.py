"""Tests for ensemble methods.

This module tests the ensemble techniques for combining multiple trained models
to improve performance in emergent language learning.
"""

import pytest
import torch
import tempfile
import os
from typing import List

from langlab.core.ensemble import (
    EnsembleListener,
    EnsembleSpeaker,
    create_ensemble_from_checkpoints,
)
from langlab.core.agents import Speaker, Listener
from langlab.core.config import CommunicationConfig


class TestEnsembleListener:
    """Test the EnsembleListener class."""

    @pytest.fixture
    def config(self) -> CommunicationConfig:
        """Create a test configuration."""
        return CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            hidden_size=32,
            multimodal=False,
        )

    @pytest.fixture
    def listeners(self, config: CommunicationConfig) -> List[Listener]:
        """Create test listener models."""
        listeners = []
        for _ in range(3):
            listener = Listener(config)
            listeners.append(listener)
        return listeners

    def test_ensemble_listener_init_equal_weights(
        self, listeners: List[Listener]
    ) -> None:
        """Test ensemble listener initialization with equal weights."""
        ensemble = EnsembleListener(listeners)

        assert ensemble.num_models == 3
        assert len(ensemble.listeners) == 3
        assert len(ensemble.weights) == 3
        # All weights should be equal (1/3)
        for weight in ensemble.weights:
            assert abs(weight - 1.0 / 3) < 1e-6

    def test_ensemble_listener_init_custom_weights(
        self, listeners: List[Listener]
    ) -> None:
        """Test ensemble listener initialization with custom weights."""
        custom_weights = [0.5, 0.3, 0.2]
        ensemble = EnsembleListener(listeners, weights=custom_weights)

        assert ensemble.num_models == 3
        assert len(ensemble.weights) == 3
        # Weights should be normalized
        assert abs(sum(ensemble.weights) - 1.0) < 1e-6
        assert abs(ensemble.weights[0] - 0.5) < 1e-6
        assert abs(ensemble.weights[1] - 0.3) < 1e-6
        assert abs(ensemble.weights[2] - 0.2) < 1e-6

    def test_ensemble_listener_init_weight_mismatch(
        self, listeners: List[Listener]
    ) -> None:
        """Test ensemble listener initialization with mismatched weights."""
        wrong_weights = [0.5, 0.3]  # Only 2 weights for 3 listeners

        with pytest.raises(
            AssertionError, match="Number of weights must match number of models"
        ):
            EnsembleListener(listeners, weights=wrong_weights)

    def test_ensemble_listener_forward(
        self, listeners: List[Listener], config: CommunicationConfig
    ) -> None:
        """Test ensemble listener forward pass."""
        ensemble = EnsembleListener(listeners)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(batch_size, num_candidates, config.object_dim)

        output = ensemble(message_tokens, candidate_objects)

        assert output.shape == (batch_size, num_candidates)
        # Probabilities should sum to 1 for each batch
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_ensemble_listener_forward_with_gestures(
        self, listeners: List[Listener], config: CommunicationConfig
    ) -> None:
        """Test ensemble listener forward pass with gesture tokens."""
        # Create multimodal config
        multimodal_config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            hidden_size=32,
            multimodal=True,
            gesture_size=5,
        )

        multimodal_listeners = [Listener(multimodal_config) for _ in range(2)]
        ensemble = EnsembleListener(multimodal_listeners)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randint(
            0,
            multimodal_config.vocabulary_size,
            (batch_size, multimodal_config.message_length),
        )
        candidate_objects = torch.randn(
            batch_size, num_candidates, multimodal_config.object_dim
        )
        gesture_tokens = torch.randint(
            0,
            multimodal_config.gesture_size,
            (batch_size, multimodal_config.message_length),
        )

        output = ensemble(message_tokens, candidate_objects, gesture_tokens)

        assert output.shape == (batch_size, num_candidates)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)


class TestEnsembleSpeaker:
    """Test the EnsembleSpeaker class."""

    @pytest.fixture
    def config(self) -> CommunicationConfig:
        """Create a test configuration."""
        return CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            hidden_size=32,
            multimodal=False,
        )

    @pytest.fixture
    def speakers(self, config: CommunicationConfig) -> List[Speaker]:
        """Create test speaker models."""
        speakers = []
        for _ in range(3):
            speaker = Speaker(config)
            speakers.append(speaker)
        return speakers

    def test_ensemble_speaker_init_equal_weights(self, speakers: List[Speaker]) -> None:
        """Test ensemble speaker initialization with equal weights."""
        ensemble = EnsembleSpeaker(speakers)

        assert ensemble.num_models == 3
        assert len(ensemble.speakers) == 3
        assert len(ensemble.weights) == 3
        # All weights should be equal (1/3)
        for weight in ensemble.weights:
            assert abs(weight - 1.0 / 3) < 1e-6

    def test_ensemble_speaker_init_custom_weights(
        self, speakers: List[Speaker]
    ) -> None:
        """Test ensemble speaker initialization with custom weights."""
        custom_weights = [0.6, 0.3, 0.1]
        ensemble = EnsembleSpeaker(speakers, weights=custom_weights)

        assert ensemble.num_models == 3
        assert len(ensemble.weights) == 3
        # Weights should be normalized
        assert abs(sum(ensemble.weights) - 1.0) < 1e-6
        assert abs(ensemble.weights[0] - 0.6) < 1e-6
        assert abs(ensemble.weights[1] - 0.3) < 1e-6
        assert abs(ensemble.weights[2] - 0.1) < 1e-6

    def test_ensemble_speaker_init_weight_mismatch(
        self, speakers: List[Speaker]
    ) -> None:
        """Test ensemble speaker initialization with mismatched weights."""
        wrong_weights = [0.7, 0.3]  # Only 2 weights for 3 speakers

        with pytest.raises(
            AssertionError, match="Number of weights must match number of models"
        ):
            EnsembleSpeaker(speakers, weights=wrong_weights)

    def test_ensemble_speaker_forward(
        self, speakers: List[Speaker], config: CommunicationConfig
    ) -> None:
        """Test ensemble speaker forward pass."""
        ensemble = EnsembleSpeaker(speakers)

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)

        logits, tokens, gesture_logits, gesture_tokens = ensemble(object_encoding)

        assert logits.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert tokens.shape == (
            batch_size,
            config.message_length,
        )  # Integer indices, not one-hot
        assert gesture_logits is None  # Non-multimodal
        assert gesture_tokens is None  # Non-multimodal

    def test_ensemble_speaker_forward_with_temperature(
        self, speakers: List[Speaker], config: CommunicationConfig
    ) -> None:
        """Test ensemble speaker forward pass with temperature scaling."""
        ensemble = EnsembleSpeaker(speakers)

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)
        temperature = 0.5

        logits, tokens, gesture_logits, gesture_tokens = ensemble(
            object_encoding, temperature
        )

        assert logits.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert tokens.shape == (
            batch_size,
            config.message_length,
        )  # Integer indices, not one-hot

    def test_ensemble_speaker_forward_multimodal(
        self, config: CommunicationConfig
    ) -> None:
        """Test ensemble speaker forward pass with multimodal config."""
        multimodal_config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            hidden_size=32,
            multimodal=True,
            gesture_size=5,
        )

        speakers = [Speaker(multimodal_config) for _ in range(2)]
        ensemble = EnsembleSpeaker(speakers)

        batch_size = 2
        object_encoding = torch.randn(batch_size, multimodal_config.object_dim)

        logits, tokens, gesture_logits, gesture_tokens = ensemble(object_encoding)

        assert logits.shape == (
            batch_size,
            multimodal_config.message_length,
            multimodal_config.vocabulary_size,
        )
        assert tokens.shape == (
            batch_size,
            multimodal_config.message_length,
        )  # Integer indices, not one-hot
        assert gesture_logits.shape == (
            batch_size,
            multimodal_config.message_length,
            multimodal_config.gesture_size,
        )
        assert gesture_tokens.shape == (
            batch_size,
            multimodal_config.message_length,
        )  # Integer indices, not one-hot


class TestCreateEnsembleFromCheckpoints:
    """Test the create_ensemble_from_checkpoints function."""

    @pytest.fixture
    def config(self) -> CommunicationConfig:
        """Create a test configuration."""
        return CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            hidden_size=32,
            multimodal=False,
        )

    def test_create_ensemble_from_checkpoints(
        self, config: CommunicationConfig
    ) -> None:
        """Test creating ensemble from checkpoint files."""
        # Create temporary checkpoint files
        checkpoint_paths = []

        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(3):
                checkpoint_path = os.path.join(temp_dir, f"checkpoint_{i}.pt")

                # Create dummy checkpoint data
                speaker = Speaker(config)
                listener = Listener(config)

                checkpoint = {
                    "speaker_state_dict": speaker.state_dict(),
                    "listener_state_dict": listener.state_dict(),
                }

                torch.save(checkpoint, checkpoint_path)
                checkpoint_paths.append(checkpoint_path)

            # Create ensemble from checkpoints
            ensemble_speaker, ensemble_listener = create_ensemble_from_checkpoints(
                checkpoint_paths, config
            )

            assert isinstance(ensemble_speaker, EnsembleSpeaker)
            assert isinstance(ensemble_listener, EnsembleListener)
            assert ensemble_speaker.num_models == 3
            assert ensemble_listener.num_models == 3

    def test_create_ensemble_from_checkpoints_with_device(
        self, config: CommunicationConfig
    ) -> None:
        """Test creating ensemble from checkpoints with specific device."""
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")

            # Create dummy checkpoint
            speaker = Speaker(config)
            listener = Listener(config)

            checkpoint = {
                "speaker_state_dict": speaker.state_dict(),
                "listener_state_dict": listener.state_dict(),
            }

            torch.save(checkpoint, checkpoint_path)

            # Create ensemble with specific device
            ensemble_speaker, ensemble_listener = create_ensemble_from_checkpoints(
                [checkpoint_path], config, device=device
            )

            assert isinstance(ensemble_speaker, EnsembleSpeaker)
            assert isinstance(ensemble_listener, EnsembleListener)

    def test_create_ensemble_from_checkpoints_empty_list(
        self, config: CommunicationConfig
    ) -> None:
        """Test creating ensemble from empty checkpoint list."""
        with pytest.raises(ZeroDivisionError):
            create_ensemble_from_checkpoints([], config)

    def test_create_ensemble_from_checkpoints_nonexistent_file(
        self, config: CommunicationConfig
    ) -> None:
        """Test creating ensemble from nonexistent checkpoint file."""
        nonexistent_path = "/nonexistent/path/checkpoint.pt"

        with pytest.raises(FileNotFoundError):
            create_ensemble_from_checkpoints([nonexistent_path], config)
