"""Comprehensive tests for improved neural network agents.

This module tests the enhanced Speaker and Listener agents with advanced
architectures and training techniques for emergent language learning.
"""

import pytest
import torch
import torch.nn as nn

from langlab.core.improved_agents import (
    ImprovedSpeaker,
    ImprovedListener,
    ImprovedSpeakerSeq,
    ImprovedListenerSeq,
)
from langlab.core.config import CommunicationConfig


class TestImprovedSpeaker:
    """Test the ImprovedSpeaker class."""

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
    def multimodal_config(self) -> CommunicationConfig:
        """Create a multimodal test configuration."""
        return CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            hidden_size=32,
            multimodal=True,
            gesture_size=5,
        )

    def test_improved_speaker_init(self, config: CommunicationConfig) -> None:
        """Test improved speaker initialization."""
        speaker = ImprovedSpeaker(config)

        assert isinstance(speaker, nn.Module)
        assert speaker.config == config
        assert (
            speaker.input_dim == config.object_dim
        )  # Use object_dim instead of TOTAL_ATTRIBUTES
        assert hasattr(speaker, "object_encoder")
        assert hasattr(speaker, "self_attention")
        assert hasattr(speaker, "message_transformer")
        assert hasattr(speaker, "output_layers")
        assert not hasattr(speaker, "gesture_layers")  # Non-multimodal

    def test_improved_speaker_init_multimodal(
        self, multimodal_config: CommunicationConfig
    ) -> None:
        """Test improved speaker initialization with multimodal config."""
        speaker = ImprovedSpeaker(multimodal_config)

        assert isinstance(speaker, nn.Module)
        assert speaker.config == multimodal_config
        assert hasattr(speaker, "gesture_layers")  # Multimodal

    def test_improved_speaker_forward(self, config: CommunicationConfig) -> None:
        """Test improved speaker forward pass."""
        speaker = ImprovedSpeaker(config)
        speaker.eval()  # Set to eval mode for deterministic behavior

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)

        logits, tokens, gesture_logits, gesture_tokens = speaker(object_encoding)

        assert logits.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert tokens.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert gesture_logits is None  # Non-multimodal
        assert gesture_tokens is None  # Non-multimodal

    def test_improved_speaker_forward_multimodal(
        self, multimodal_config: CommunicationConfig
    ) -> None:
        """Test improved speaker forward pass with multimodal config."""
        speaker = ImprovedSpeaker(multimodal_config)
        speaker.eval()

        batch_size = 2
        object_encoding = torch.randn(batch_size, multimodal_config.object_dim)

        logits, tokens, gesture_logits, gesture_tokens = speaker(object_encoding)

        assert logits.shape == (
            batch_size,
            multimodal_config.message_length,
            multimodal_config.vocabulary_size,
        )
        assert tokens.shape == (
            batch_size,
            multimodal_config.message_length,
            multimodal_config.vocabulary_size,
        )
        assert gesture_logits.shape == (
            batch_size,
            multimodal_config.message_length,
            multimodal_config.gesture_size,
        )
        assert gesture_tokens.shape == (
            batch_size,
            multimodal_config.message_length,
            multimodal_config.gesture_size,
        )

    def test_improved_speaker_forward_with_temperature(
        self, config: CommunicationConfig
    ) -> None:
        """Test improved speaker forward pass with temperature scaling."""
        speaker = ImprovedSpeaker(config)
        speaker.eval()

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)
        temperature = 0.5

        logits, tokens, gesture_logits, gesture_tokens = speaker(
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
            config.vocabulary_size,
        )

    def test_improved_speaker_training_mode(self, config: CommunicationConfig) -> None:
        """Test improved speaker behavior in training mode."""
        speaker = ImprovedSpeaker(config)
        speaker.train()  # Set to training mode

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)

        logits, tokens, gesture_logits, gesture_tokens = speaker(object_encoding)

        assert logits.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert tokens.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )

    def test_improved_speaker_weight_initialization(
        self, config: CommunicationConfig
    ) -> None:
        """Test that weights are properly initialized."""
        speaker = ImprovedSpeaker(config)

        # Check that weights are initialized (not all zeros)
        for name, param in speaker.named_parameters():
            if "weight" in name and len(param.shape) > 1:  # Skip bias and 1D weights
                assert not torch.allclose(param, torch.zeros_like(param))

    def test_improved_speaker_different_temperatures(
        self, config: CommunicationConfig
    ) -> None:
        """Test improved speaker with different temperature values."""
        speaker = ImprovedSpeaker(config)
        speaker.eval()

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)

        # Test with different temperatures
        for temp in [0.1, 0.5, 1.0, 2.0]:
            logits, tokens, _, _ = speaker(object_encoding, temperature=temp)
            assert logits.shape == (
                batch_size,
                config.message_length,
                config.vocabulary_size,
            )
            assert tokens.shape == (
                batch_size,
                config.message_length,
                config.vocabulary_size,
            )


class TestImprovedListener:
    """Test the ImprovedListener class."""

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
    def multimodal_config(self) -> CommunicationConfig:
        """Create a multimodal test configuration."""
        return CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            hidden_size=32,
            multimodal=True,
            gesture_size=5,
        )

    def test_improved_listener_init(self, config: CommunicationConfig) -> None:
        """Test improved listener initialization."""
        listener = ImprovedListener(config)

        assert isinstance(listener, nn.Module)
        assert listener.config == config
        assert listener.message_dim == config.vocabulary_size
        assert listener.object_dim == config.object_dim
        assert hasattr(listener, "message_encoder")
        assert hasattr(listener, "object_encoder")
        assert hasattr(listener, "cross_attention")
        assert hasattr(listener, "scorer")

    def test_improved_listener_forward(self, config: CommunicationConfig) -> None:
        """Test improved listener forward pass."""
        listener = ImprovedListener(config)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randn(
            batch_size, config.message_length, config.vocabulary_size
        )
        candidate_objects = torch.randn(batch_size, num_candidates, config.object_dim)

        output = listener(message_tokens, candidate_objects)

        assert output.shape == (batch_size, num_candidates)
        # Probabilities should sum to 1 for each batch
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_improved_listener_forward_with_gestures(
        self, multimodal_config: CommunicationConfig
    ) -> None:
        """Test improved listener forward pass with gesture tokens."""
        listener = ImprovedListener(multimodal_config)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randn(
            batch_size,
            multimodal_config.message_length,
            multimodal_config.vocabulary_size,
        )
        candidate_objects = torch.randn(
            batch_size, num_candidates, multimodal_config.object_dim
        )
        gesture_tokens = torch.randn(
            batch_size, multimodal_config.message_length, multimodal_config.gesture_size
        )

        output = listener(message_tokens, candidate_objects, gesture_tokens)

        assert output.shape == (batch_size, num_candidates)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_improved_listener_forward_without_gestures(
        self, multimodal_config: CommunicationConfig
    ) -> None:
        """Test improved listener forward pass without gesture tokens in multimodal config."""
        listener = ImprovedListener(multimodal_config)

        batch_size = 2
        num_candidates = 3
        # For multimodal config, we need to provide the full input dimension
        # The listener expects message + gesture dimensions even when gestures are None
        message_input_dim = (
            multimodal_config.message_length * multimodal_config.vocabulary_size
            + multimodal_config.message_length * multimodal_config.gesture_size
        )
        message_tokens = torch.randn(batch_size, message_input_dim)
        candidate_objects = torch.randn(
            batch_size, num_candidates, multimodal_config.object_dim
        )

        output = listener(message_tokens, candidate_objects)

        assert output.shape == (batch_size, num_candidates)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_improved_listener_weight_initialization(
        self, config: CommunicationConfig
    ) -> None:
        """Test that weights are properly initialized."""
        listener = ImprovedListener(config)

        # Check that weights are initialized (not all zeros)
        for name, param in listener.named_parameters():
            if "weight" in name and len(param.shape) > 1:  # Skip bias and 1D weights
                assert not torch.allclose(param, torch.zeros_like(param))

    def test_improved_listener_different_candidate_counts(
        self, config: CommunicationConfig
    ) -> None:
        """Test improved listener with different numbers of candidates."""
        listener = ImprovedListener(config)

        batch_size = 2
        message_tokens = torch.randn(
            batch_size, config.message_length, config.vocabulary_size
        )

        # Test with different numbers of candidates
        for num_candidates in [1, 2, 5, 10]:
            candidate_objects = torch.randn(
                batch_size, num_candidates, config.object_dim
            )
            output = listener(message_tokens, candidate_objects)

            assert output.shape == (batch_size, num_candidates)
            assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)


class TestImprovedSpeakerSeq:
    """Test the ImprovedSpeakerSeq class."""

    @pytest.fixture
    def config(self) -> CommunicationConfig:
        """Create a test configuration."""
        return CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            hidden_size=32,
            multimodal=False,
        )

    def test_improved_speaker_seq_init(self, config: CommunicationConfig) -> None:
        """Test improved sequence speaker initialization."""
        speaker = ImprovedSpeakerSeq(config)

        assert isinstance(speaker, nn.Module)
        assert speaker.config == config
        assert speaker.input_dim == config.object_dim
        assert hasattr(speaker, "object_encoder")
        assert hasattr(speaker, "sequence_transformer")
        assert hasattr(speaker, "token_embedding")
        assert hasattr(speaker, "output_layer")

    def test_improved_speaker_seq_forward(self, config: CommunicationConfig) -> None:
        """Test improved sequence speaker forward pass."""
        speaker = ImprovedSpeakerSeq(config)
        speaker.eval()

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)

        logits, tokens = speaker(object_encoding)

        assert logits.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert tokens.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )

    def test_improved_speaker_seq_forward_with_temperature(
        self, config: CommunicationConfig
    ) -> None:
        """Test improved sequence speaker forward pass with temperature."""
        speaker = ImprovedSpeakerSeq(config)
        speaker.eval()

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)
        temperature = 0.5

        logits, tokens = speaker(object_encoding, temperature)

        assert logits.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert tokens.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )

    def test_improved_speaker_seq_forward_with_teacher_tokens(
        self, config: CommunicationConfig
    ) -> None:
        """Test improved sequence speaker forward pass with teacher tokens."""
        speaker = ImprovedSpeakerSeq(config)
        speaker.train()  # Must be in training mode for teacher forcing

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)
        teacher_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )

        logits, tokens = speaker(object_encoding, teacher_tokens=teacher_tokens)

        assert logits.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert tokens.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )

    def test_improved_speaker_seq_weight_initialization(
        self, config: CommunicationConfig
    ) -> None:
        """Test that weights are properly initialized."""
        speaker = ImprovedSpeakerSeq(config)

        # Check that weights are initialized (not all zeros)
        for name, param in speaker.named_parameters():
            if "weight" in name and len(param.shape) > 1:  # Skip bias and 1D weights
                assert not torch.allclose(param, torch.zeros_like(param))


class TestImprovedListenerSeq:
    """Test the ImprovedListenerSeq class."""

    @pytest.fixture
    def config(self) -> CommunicationConfig:
        """Create a test configuration."""
        return CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            hidden_size=32,
            multimodal=False,
        )

    def test_improved_listener_seq_init(self, config: CommunicationConfig) -> None:
        """Test improved sequence listener initialization."""
        listener = ImprovedListenerSeq(config)

        assert isinstance(listener, nn.Module)
        assert listener.config == config
        assert listener.vocab_size == config.vocabulary_size
        assert listener.object_dim == config.object_dim
        assert listener.hidden_size == config.hidden_size
        assert hasattr(listener, "token_embedding")
        assert hasattr(listener, "message_encoder")
        assert hasattr(listener, "object_encoder")
        assert hasattr(listener, "cross_attention")
        assert hasattr(listener, "scorer")

    def test_improved_listener_seq_forward_one_hot(
        self, config: CommunicationConfig
    ) -> None:
        """Test improved sequence listener forward pass with one-hot tokens."""
        listener = ImprovedListenerSeq(config)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randn(
            batch_size, config.message_length, config.vocabulary_size
        )
        candidate_objects = torch.randn(batch_size, num_candidates, config.object_dim)

        output = listener(message_tokens, candidate_objects)

        assert output.shape == (batch_size, num_candidates)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_improved_listener_seq_forward_indices(
        self, config: CommunicationConfig
    ) -> None:
        """Test improved sequence listener forward pass with token indices."""
        listener = ImprovedListenerSeq(config)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(batch_size, num_candidates, config.object_dim)

        output = listener(message_tokens, candidate_objects)

        assert output.shape == (batch_size, num_candidates)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_improved_listener_seq_weight_initialization(
        self, config: CommunicationConfig
    ) -> None:
        """Test that weights are properly initialized."""
        listener = ImprovedListenerSeq(config)

        # Check that weights are initialized (not all zeros)
        for name, param in listener.named_parameters():
            if "weight" in name and len(param.shape) > 1:  # Skip bias and 1D weights
                assert not torch.allclose(param, torch.zeros_like(param))

    def test_improved_listener_seq_different_message_lengths(
        self, config: CommunicationConfig
    ) -> None:
        """Test improved sequence listener with different message lengths."""
        # Create config with different message length
        long_config = CommunicationConfig(
            vocabulary_size=10,
            message_length=5,  # Longer messages
            hidden_size=32,
            multimodal=False,
        )

        listener = ImprovedListenerSeq(long_config)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randint(
            0, long_config.vocabulary_size, (batch_size, long_config.message_length)
        )
        candidate_objects = torch.randn(
            batch_size, num_candidates, long_config.object_dim
        )

        output = listener(message_tokens, candidate_objects)

        assert output.shape == (batch_size, num_candidates)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)
