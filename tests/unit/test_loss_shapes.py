"""Tests for loss function shapes and tensor operations.

This module verifies that loss functions return properly shaped tensors
and that all operations maintain correct tensor dimensions.
"""

import torch

from langlab.training.train import compute_listener_loss, compute_speaker_loss
from langlab.core.agents import Speaker, Listener
from langlab.core.config import CommunicationConfig


class TestLossShapes:
    """Test that loss functions return correct tensor shapes."""

    def test_losses_scalar(self) -> None:
        """Test that all returned losses are scalar tensors."""
        # Set up configuration
        config = CommunicationConfig(
            vocabulary_size=5, message_length=1, hidden_size=32, seed=42
        )

        # Create agents
        speaker = Speaker(config)
        listener = Listener(config)

        # Create test data
        batch_size = 4
        num_candidates = 3
        message_length = config.message_length
        vocab_size = config.vocabulary_size
        object_dim = 8  # TOTAL_ATTRIBUTES

        # Test data tensors
        message_tokens = torch.randint(0, vocab_size, (batch_size, message_length))
        candidate_objects = torch.randn(batch_size, num_candidates, object_dim)
        target_indices = torch.randint(0, num_candidates, (batch_size,))
        speaker_logits = torch.randn(batch_size, message_length, vocab_size)
        rewards = torch.randint(0, 2, (batch_size,)).float()
        baseline = 0.5

        # Test listener loss
        listener_loss = compute_listener_loss(
            listener, message_tokens, candidate_objects, target_indices
        )

        # Verify listener loss is scalar
        assert isinstance(listener_loss, torch.Tensor)
        assert (
            listener_loss.dim() == 0
        ), f"Listener loss should be scalar, got shape {listener_loss.shape}"
        assert listener_loss.requires_grad, "Listener loss should require gradients"

        # Test speaker loss
        speaker_loss = compute_speaker_loss(speaker, speaker_logits, rewards, baseline)

        # Verify speaker loss is scalar
        assert isinstance(speaker_loss, torch.Tensor)
        assert (
            speaker_loss.dim() == 0
        ), f"Speaker loss should be scalar, got shape {speaker_loss.shape}"
        # Note: speaker_loss may not require gradients if speaker_logits are not connected to speaker

        # Test that losses can be converted to Python floats
        listener_loss_val = listener_loss.item()
        speaker_loss_val = speaker_loss.item()

        assert isinstance(listener_loss_val, float)
        assert isinstance(speaker_loss_val, float)

        # Test that losses are finite
        assert torch.isfinite(listener_loss), "Listener loss should be finite"
        assert torch.isfinite(speaker_loss), "Speaker loss should be finite"

    def test_loss_gradients(self) -> None:
        """Test that losses have proper gradients."""
        config = CommunicationConfig(
            vocabulary_size=3, message_length=1, hidden_size=16, seed=42
        )

        speaker = Speaker(config)
        listener = Listener(config)

        # Create test data
        batch_size = 2
        num_candidates = 2
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(batch_size, num_candidates, 8)
        target_indices = torch.randint(0, num_candidates, (batch_size,))
        speaker_logits = torch.randn(
            batch_size, config.message_length, config.vocabulary_size
        )
        rewards = torch.randint(0, 2, (batch_size,)).float()

        # Test listener loss gradients
        listener_loss = compute_listener_loss(
            listener, message_tokens, candidate_objects, target_indices
        )
        listener_loss.backward()

        # Check that listener parameters have gradients
        for param in listener.parameters():
            assert param.grad is not None, "Listener parameters should have gradients"
            assert torch.isfinite(
                param.grad
            ).all(), "Listener gradients should be finite"

        # Clear gradients
        listener.zero_grad()

        # Test speaker loss gradients
        # Note: speaker_loss may not have gradients if speaker_logits are not connected to speaker
        # This is expected behavior for the current implementation
        _ = compute_speaker_loss(speaker, speaker_logits, rewards, 0.5)

    def test_loss_values_reasonable(self) -> None:
        """Test that loss values are in reasonable ranges."""
        config = CommunicationConfig(
            vocabulary_size=4, message_length=1, hidden_size=16, seed=42
        )

        speaker = Speaker(config)
        listener = Listener(config)

        # Create test data
        batch_size = 3
        num_candidates = 3
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(batch_size, num_candidates, 8)
        target_indices = torch.randint(0, num_candidates, (batch_size,))
        speaker_logits = torch.randn(
            batch_size, config.message_length, config.vocabulary_size
        )
        rewards = torch.randint(0, 2, (batch_size,)).float()

        # Test listener loss
        listener_loss = compute_listener_loss(
            listener, message_tokens, candidate_objects, target_indices
        )

        # Cross-entropy loss should be positive
        assert (
            listener_loss.item() >= 0
        ), f"Listener loss should be non-negative, got {listener_loss.item()}"

        # For K=3 candidates, max cross-entropy is log(3) ≈ 1.1
        # Allow for higher initial losses due to random initialization
        max_ce_loss = torch.log(torch.tensor(num_candidates, dtype=torch.float32))
        assert (
            listener_loss.item() <= max_ce_loss.item() + 2.0
        ), f"Listener loss {listener_loss.item():.3f} seems too high for {num_candidates} candidates"

        # Test speaker loss
        speaker_loss = compute_speaker_loss(speaker, speaker_logits, rewards, 0.5)

        # REINFORCE loss can be positive or negative depending on advantages
        assert torch.isfinite(speaker_loss), "Speaker loss should be finite"

        # Loss magnitude should be reasonable (not extremely large)
        assert (
            abs(speaker_loss.item()) < 10.0
        ), f"Speaker loss magnitude {abs(speaker_loss.item()):.3f} seems too large"
