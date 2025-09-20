"""Tests for sequence-aware Speaker and Listener models.

This module contains tests to validate the shape and behavior of the
autoregressive SpeakerSeq and sequence-aware ListenerSeq models.
"""

import pytest
import torch

from langlab.core.agents import SpeakerSeq, ListenerSeq
from langlab.core.config import CommunicationConfig
from langlab.data.world import TOTAL_ATTRIBUTES


@pytest.fixture
def config() -> CommunicationConfig:
    """Create a test configuration."""
    return CommunicationConfig(
        vocabulary_size=12,
        message_length=3,
        hidden_size=64,
        seed=42,
    )


@pytest.fixture
def speaker_seq(config: CommunicationConfig) -> SpeakerSeq:
    """Create a SpeakerSeq model."""
    return SpeakerSeq(config)


@pytest.fixture
def listener_seq(config: CommunicationConfig) -> ListenerSeq:
    """Create a ListenerSeq model."""
    return ListenerSeq(config)


def test_speaker_seq_shapes(
    speaker_seq: SpeakerSeq, config: CommunicationConfig
) -> None:
    """Test that SpeakerSeq produces correct output shapes.

    Validates that logits have shape [L, V] and tokens have shape [L]
    for batch dimension.
    """
    batch_size = 4
    object_encoding = torch.randn(batch_size, TOTAL_ATTRIBUTES)

    logits, tokens = speaker_seq(object_encoding)

    # Check shapes
    assert logits.shape == (batch_size, config.message_length, config.vocabulary_size)
    assert tokens.shape == (batch_size, config.message_length)

    # Check that tokens are valid indices
    assert tokens.min() >= 0
    assert tokens.max() < config.vocabulary_size

    # Check that tokens are integers
    assert tokens.dtype == torch.long


def test_speaker_seq_autoregressive(
    speaker_seq: SpeakerSeq, config: CommunicationConfig
) -> None:
    """Test that SpeakerSeq generates tokens autoregressively."""
    batch_size = 2
    object_encoding = torch.randn(batch_size, TOTAL_ATTRIBUTES)

    # Generate with different temperatures
    _, tokens_cold = speaker_seq(object_encoding, temperature=0.1)
    _, tokens_hot = speaker_seq(object_encoding, temperature=2.0)

    # Both should have correct shapes
    assert tokens_cold.shape == (batch_size, config.message_length)
    assert tokens_hot.shape == (batch_size, config.message_length)

    # Tokens should be valid
    assert tokens_cold.min() >= 0
    assert tokens_cold.max() < config.vocabulary_size
    assert tokens_hot.min() >= 0
    assert tokens_hot.max() < config.vocabulary_size


def test_listener_seq_scores(
    listener_seq: ListenerSeq, config: CommunicationConfig
) -> None:
    """Test that ListenerSeq produces valid probability distributions.

    Validates that probabilities over K candidates sum to 1 and are finite.
    """
    batch_size = 3
    num_candidates = 6
    message_tokens = torch.randint(
        0, config.vocabulary_size, (batch_size, config.message_length)
    )
    candidate_objects = torch.randn(batch_size, num_candidates, TOTAL_ATTRIBUTES)

    probabilities = listener_seq(message_tokens, candidate_objects)

    # Check shape
    assert probabilities.shape == (batch_size, num_candidates)

    # Check that probabilities sum to 1 (within tolerance)
    prob_sums = probabilities.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)

    # Check that all probabilities are finite and non-negative
    assert torch.all(torch.isfinite(probabilities))
    assert torch.all(probabilities >= 0)

    # Check that probabilities are reasonable (not all zeros or ones)
    assert torch.all(probabilities > 1e-8)  # No exact zeros
    assert torch.all(probabilities < 1.0)  # No exact ones


def test_listener_seq_different_candidates(
    listener_seq: ListenerSeq, config: CommunicationConfig
) -> None:
    """Test ListenerSeq with different numbers of candidates."""
    batch_size = 2
    message_tokens = torch.randint(
        0, config.vocabulary_size, (batch_size, config.message_length)
    )

    # Test with different numbers of candidates
    for num_candidates in [3, 5, 8]:
        candidate_objects = torch.randn(batch_size, num_candidates, TOTAL_ATTRIBUTES)
        probabilities = listener_seq(message_tokens, candidate_objects)

        assert probabilities.shape == (batch_size, num_candidates)
        prob_sums = probabilities.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)


def test_sequence_models_gradient_flow(
    speaker_seq: SpeakerSeq, listener_seq: ListenerSeq, config: CommunicationConfig
) -> None:
    """Test that gradients flow properly through sequence models."""
    batch_size = 2
    num_candidates = 4

    # Create inputs
    object_encoding = torch.randn(batch_size, TOTAL_ATTRIBUTES, requires_grad=True)
    candidate_objects = torch.randn(batch_size, num_candidates, TOTAL_ATTRIBUTES)

    # Forward pass through SpeakerSeq (use logits directly for gradient flow)
    logits, generated_tokens = speaker_seq(object_encoding)

    # Use logits instead of generated tokens for gradient flow
    # Convert logits to probabilities for the listener
    message_probs = torch.nn.functional.softmax(logits, dim=-1)
    # Use weighted sum instead of discrete tokens
    message_features = torch.sum(
        message_probs.unsqueeze(-1)
        * torch.arange(config.vocabulary_size, dtype=torch.float)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0),
        dim=2,
    )
    message_features = message_features.mean(dim=1)  # Average across sequence length

    # Create dummy message tokens for listener (this is just for testing)
    dummy_tokens = torch.randint(
        0, config.vocabulary_size, (batch_size, config.message_length)
    )
    probabilities = listener_seq(dummy_tokens, candidate_objects)

    # Compute a simple loss using logits from speaker
    speaker_loss = logits.mean()  # Simple loss on logits
    listener_loss = torch.nn.functional.cross_entropy(
        probabilities, torch.randint(0, num_candidates, (batch_size,))
    )
    total_loss = speaker_loss + listener_loss

    # Backward pass
    total_loss.backward()

    # Check that gradients exist for model parameters
    speaker_has_grads = any(p.grad is not None for p in speaker_seq.parameters())
    listener_has_grads = any(p.grad is not None for p in listener_seq.parameters())

    assert speaker_has_grads, "SpeakerSeq should have gradients"
    assert listener_has_grads, "ListenerSeq should have gradients"
