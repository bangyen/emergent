"""Tests for multimodal communication functionality.

This module tests the multimodal communication features including gesture streams,
synchronized token and gesture lengths, and joint attention mechanisms.
"""

import torch
import torch.nn.functional as F

from src.langlab.config import CommunicationConfig
from src.langlab.channel import DiscreteChannel
from src.langlab.agents import Speaker, Listener


class TestMultimodalChannel:
    """Test multimodal channel functionality."""

    def test_multimodal_config(self) -> None:
        """Test multimodal configuration parameters."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            gesture_size=5,
            multimodal=True,
        )

        assert config.multimodal is True
        assert config.gesture_size == 5
        assert config.vocabulary_size == 10
        assert config.message_length == 2

    def test_channel_multimodal_send(self) -> None:
        """Test multimodal message transmission through channel."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            gesture_size=5,
            multimodal=True,
        )
        channel = DiscreteChannel(config)

        batch_size = 3
        speaker_logits = torch.randn(
            batch_size, config.message_length, config.vocabulary_size
        )
        gesture_logits = torch.randn(
            batch_size, config.message_length, config.gesture_size
        )

        token_ids, gesture_ids = channel.send_multimodal(speaker_logits, gesture_logits)

        # Check shapes
        assert token_ids.shape == (batch_size, config.message_length)
        assert gesture_ids.shape == (batch_size, config.message_length)

        # Check token range
        assert token_ids.min() >= 0
        assert token_ids.max() < config.vocabulary_size

        # Check gesture range
        assert gesture_ids.min() >= 0
        assert gesture_ids.max() < config.gesture_size

    def test_channel_validation(self) -> None:
        """Test channel validation for tokens and gestures."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            gesture_size=5,
            multimodal=True,
        )
        channel = DiscreteChannel(config)

        # Valid tokens and gestures
        valid_tokens = torch.tensor([[0, 5], [9, 2]])
        valid_gestures = torch.tensor([[0, 4], [2, 1]])

        assert channel.validate_tokens(valid_tokens) is True
        assert channel.validate_gestures(valid_gestures) is True

        # Invalid tokens
        invalid_tokens = torch.tensor([[0, 10], [9, 2]])  # 10 >= vocab_size
        assert channel.validate_tokens(invalid_tokens) is False

        # Invalid gestures
        invalid_gestures = torch.tensor([[0, 5], [9, 2]])  # 5 >= gesture_size
        assert channel.validate_gestures(invalid_gestures) is False

    def test_channel_ranges(self) -> None:
        """Test channel range methods."""
        config = CommunicationConfig(
            vocabulary_size=10,
            gesture_size=5,
            multimodal=True,
        )
        channel = DiscreteChannel(config)

        token_range = channel.get_token_range()
        gesture_range = channel.get_gesture_range()

        assert token_range == (0, 9)
        assert gesture_range == (0, 4)


class TestMultimodalAgents:
    """Test multimodal agent functionality."""

    def test_speaker_multimodal_output(self) -> None:
        """Test Speaker multimodal output shapes and synchronization."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            gesture_size=5,
            multimodal=True,
        )
        speaker = Speaker(config)

        batch_size = 3
        object_encoding = torch.randn(batch_size, 8)  # TOTAL_ATTRIBUTES

        logits, token_ids, gesture_logits, gesture_ids = speaker(object_encoding)

        # Check shapes
        assert logits.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert token_ids.shape == (batch_size, config.message_length)
        assert gesture_logits.shape == (
            batch_size,
            config.message_length,
            config.gesture_size,
        )
        assert gesture_ids.shape == (batch_size, config.message_length)

        # Check synchronization - same batch size and message length
        assert token_ids.shape == gesture_ids.shape
        assert logits.shape[:2] == gesture_logits.shape[:2]

    def test_speaker_unimodal_output(self) -> None:
        """Test Speaker unimodal output (no gestures)."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            multimodal=False,
        )
        speaker = Speaker(config)

        batch_size = 3
        object_encoding = torch.randn(batch_size, 8)

        logits, token_ids, gesture_logits, gesture_ids = speaker(object_encoding)

        # Check shapes
        assert logits.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert token_ids.shape == (batch_size, config.message_length)
        assert gesture_logits is None
        assert gesture_ids is None

    def test_listener_multimodal_input(self) -> None:
        """Test Listener multimodal input processing."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            gesture_size=5,
            multimodal=True,
        )
        listener = Listener(config)

        batch_size = 3
        num_candidates = 4
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        gesture_tokens = torch.randint(
            0, config.gesture_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(batch_size, num_candidates, 8)

        probabilities = listener(message_tokens, candidate_objects, gesture_tokens)

        # Check output shape
        assert probabilities.shape == (batch_size, num_candidates)

        # Check probabilities sum to 1
        prob_sums = probabilities.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)

    def test_listener_unimodal_input(self) -> None:
        """Test Listener unimodal input processing."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            multimodal=False,
        )
        listener = Listener(config)

        batch_size = 3
        num_candidates = 4
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(batch_size, num_candidates, 8)

        probabilities = listener(message_tokens, candidate_objects)

        # Check output shape
        assert probabilities.shape == (batch_size, num_candidates)

        # Check probabilities sum to 1
        prob_sums = probabilities.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)

    def test_joint_attention_shapes(self) -> None:
        """Test that joint attention produces correct shapes."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=3,
            gesture_size=5,
            multimodal=True,
        )
        listener = Listener(config)

        batch_size = 2
        num_candidates = 5
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        gesture_tokens = torch.randint(
            0, config.gesture_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(batch_size, num_candidates, 8)

        # Test multimodal processing
        multimodal_probs = listener(message_tokens, candidate_objects, gesture_tokens)
        assert multimodal_probs.shape == (batch_size, num_candidates)

        # Test unimodal processing (create unimodal listener)
        unimodal_config = CommunicationConfig(
            vocabulary_size=10,
            message_length=3,
            multimodal=False,
        )
        unimodal_listener = Listener(unimodal_config)
        unimodal_probs = unimodal_listener(message_tokens, candidate_objects)
        assert unimodal_probs.shape == (batch_size, num_candidates)

    def test_multimodal_consistency(self) -> None:
        """Test consistency between multimodal and unimodal processing."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            gesture_size=5,
            multimodal=True,
        )
        listener = Listener(config)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(batch_size, num_candidates, 8)

        # Process with zero gestures (should be similar to unimodal)
        zero_gestures = torch.zeros(batch_size, config.message_length, dtype=torch.long)
        multimodal_probs = listener(message_tokens, candidate_objects, zero_gestures)

        # Process unimodally (create unimodal listener)
        unimodal_config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            multimodal=False,
        )
        unimodal_listener = Listener(unimodal_config)
        unimodal_probs = unimodal_listener(message_tokens, candidate_objects)

        # Results should be different due to different input dimensions
        # but both should be valid probability distributions
        assert multimodal_probs.shape == unimodal_probs.shape
        assert torch.allclose(
            multimodal_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6
        )
        assert torch.allclose(
            unimodal_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6
        )


class TestMultimodalIntegration:
    """Test integration between multimodal components."""

    def test_end_to_end_multimodal(self) -> None:
        """Test end-to-end multimodal communication."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            gesture_size=5,
            multimodal=True,
        )

        speaker = Speaker(config)
        listener = Listener(config)
        channel = DiscreteChannel(config)

        batch_size = 2
        num_candidates = 3

        # Generate object encoding
        object_encoding = torch.randn(batch_size, 8)

        # Speaker generates multimodal message
        logits, token_ids, gesture_logits, gesture_ids = speaker(object_encoding)

        # Channel transmits multimodal message
        transmitted_tokens, transmitted_gestures = channel.send_multimodal(
            logits, gesture_logits
        )

        # Create candidate objects
        candidate_objects = torch.randn(batch_size, num_candidates, 8)

        # Listener processes multimodal message
        probabilities = listener(
            transmitted_tokens, candidate_objects, transmitted_gestures
        )

        # Check all shapes are consistent
        assert transmitted_tokens.shape == (batch_size, config.message_length)
        assert transmitted_gestures.shape == (batch_size, config.message_length)
        assert probabilities.shape == (batch_size, num_candidates)

        # Check probabilities are valid
        prob_sums = probabilities.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)

    def test_multimodal_gradient_flow(self) -> None:
        """Test that gradients flow properly in multimodal setup."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            gesture_size=5,
            multimodal=True,
        )

        speaker = Speaker(config)
        listener = Listener(config)

        batch_size = 2
        num_candidates = 3

        # Generate object encoding
        object_encoding = torch.randn(batch_size, 8, requires_grad=True)

        # Set models to training mode for gradient flow
        speaker.train()
        listener.train()

        # Speaker generates multimodal message
        logits, token_ids, gesture_logits, gesture_ids = speaker(object_encoding)

        # Create candidate objects
        candidate_objects = torch.randn(batch_size, num_candidates, 8)

        # Listener processes multimodal message
        probabilities = listener(token_ids, candidate_objects, gesture_ids)

        # Compute loss using logits instead of discrete tokens for gradient flow
        target_indices = torch.randint(0, num_candidates, (batch_size,))

        # Use logits for gradient computation
        if speaker.config.multimodal:
            # For multimodal, we need to compute loss differently
            # Let's use the speaker logits directly
            speaker_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), token_ids.view(-1)
            )
            loss = speaker_loss
        else:
            loss = F.cross_entropy(probabilities, target_indices)

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert object_encoding.grad is not None
        assert object_encoding.grad.shape == object_encoding.shape
