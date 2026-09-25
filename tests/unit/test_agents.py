"""Tests for Speaker and Listener agents.

This module contains unit tests for the neural network agents used in
referential games, verifying correct output shapes and behavior.
"""

import pytest
import torch

from langlab.core.agents import Speaker, Listener
from langlab.core.config import CommunicationConfig
from langlab.data.world import TOTAL_ATTRIBUTES


@pytest.fixture
def config() -> CommunicationConfig:
    """Create a test communication configuration."""
    return CommunicationConfig(
        vocabulary_size=10, message_length=1, hidden_size=64, seed=42
    )


@pytest.fixture
def speaker(config: CommunicationConfig) -> Speaker:
    """Create a Speaker agent for testing."""
    return Speaker(config)


@pytest.fixture
def listener(config: CommunicationConfig) -> Listener:
    """Create a Listener agent for testing."""
    return Listener(config)


def test_speaker_output_shapes(speaker: Speaker, config: CommunicationConfig) -> None:
    """Test that Speaker produces correct output shapes."""
    batch_size = 3
    input_dim = TOTAL_ATTRIBUTES

    # Create test input
    object_encoding = torch.randn(batch_size, input_dim)

    # Forward pass
    output = speaker(object_encoding)

    # Check logits shape: (batch_size, message_length, vocabulary_size)
    expected_logits_shape = (batch_size, config.message_length, config.vocabulary_size)
    assert output.logits.shape == expected_logits_shape, (
        f"Expected {expected_logits_shape}, got {output.logits.shape}"
    )

    # Check tokens shape: (batch_size, message_length)
    expected_token_shape = (batch_size, config.message_length)
    assert output.tokens.shape == expected_token_shape, (
        f"Expected {expected_token_shape}, got {output.tokens.shape}"
    )

    # Check that tokens are integers
    assert output.tokens.dtype in [
        torch.int64,
        torch.int32,
        torch.long,
    ], f"Expected integer dtype, got {output.tokens.dtype}"


def test_listener_output_shapes(
    listener: Listener, config: CommunicationConfig
) -> None:
    """Test that Listener produces correct output shapes for K=5 candidates."""
    batch_size = 2
    num_candidates = 5
    message_length = config.message_length
    object_dim = TOTAL_ATTRIBUTES

    # Create test inputs
    tokens = torch.randint(0, config.vocabulary_size, (batch_size, message_length))
    candidate_objects = torch.randn(batch_size, num_candidates, object_dim)

    # Forward pass
    output = listener(tokens, candidate_objects)

    # Check probs shape: (batch_size, num_candidates)
    expected_shape = (batch_size, num_candidates)
    assert output.probs.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.probs.shape}"
    )

    # Check that probabilities sum to 1 for each batch
    prob_sums = output.probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6), (
        f"Probabilities should sum to 1, got sums: {prob_sums}"
    )


def test_sample_tokens_range_and_greedy() -> None:
    """Sampled tokens stay in [0, V-1]; greedy mode picks the argmax."""
    from langlab.core.agents import _sample_tokens

    logits = torch.randn(4, 2, 10)
    tokens = _sample_tokens(logits, 1.0, stochastic=True)
    assert tokens.min() >= 0 and tokens.max() < 10
    assert torch.equal(_sample_tokens(logits, 1.0, False), logits.argmax(-1))


def test_sample_tokens_matches_softmax() -> None:
    """Gumbel-max samples follow softmax(logits)."""
    from langlab.core.agents import _sample_tokens

    torch.manual_seed(0)
    logits = torch.tensor([0.0, 1.0, 2.0]).expand(20000, 3)
    freq = torch.bincount(_sample_tokens(logits, 1.0, True), minlength=3) / 20000
    assert torch.allclose(freq, torch.softmax(logits[0], -1), atol=0.02)


def test_speaker_training_mode(speaker: Speaker, config: CommunicationConfig) -> None:
    """Test Speaker behavior in training vs evaluation mode."""
    batch_size = 2
    input_dim = TOTAL_ATTRIBUTES
    object_encoding = torch.randn(batch_size, input_dim)

    # Training mode
    speaker.train()
    output_train = speaker(object_encoding)

    # Evaluation mode
    speaker.eval()
    output_eval = speaker(object_encoding)

    assert output_train.logits.shape == output_eval.logits.shape
    assert output_train.tokens.shape == output_eval.tokens.shape


def test_listener_probability_distribution(
    listener: Listener, config: CommunicationConfig
) -> None:
    """Test that Listener produces valid probability distributions."""
    batch_size = 3
    num_candidates = 4
    tokens = torch.randint(
        0, config.vocabulary_size, (batch_size, config.message_length)
    )
    candidate_objects = torch.randn(batch_size, num_candidates, TOTAL_ATTRIBUTES)

    output = listener(tokens, candidate_objects)

    assert torch.all(output.probs >= 0)
    assert torch.all(output.probs <= 1)
    assert torch.allclose(output.probs.sum(dim=-1), torch.ones(batch_size))


def test_agent_device_compatibility(
    speaker: Speaker, listener: Listener, config: CommunicationConfig
) -> None:
    """Test that agents work on both CPU and GPU (if available)."""
    batch_size = 2
    input_dim = TOTAL_ATTRIBUTES
    num_candidates = 3

    # CPU Test
    object_encoding = torch.randn(batch_size, input_dim)
    tokens = torch.randint(
        0, config.vocabulary_size, (batch_size, config.message_length)
    )
    candidates = torch.randn(batch_size, num_candidates, input_dim)

    speaker_out = speaker(object_encoding)
    listener_out = listener(tokens, candidates)

    assert speaker_out.tokens.device.type == "cpu"
    assert listener_out.probs.device.type == "cpu"


def test_dot_listener_is_additive() -> None:
    """DotListener scores decompose over (token, attribute) terms."""
    from langlab.core.agents import DotListener

    config = CommunicationConfig(vocabulary_size=5, message_length=2, hidden_size=16)
    listener = DotListener(config)
    tokens = torch.tensor([[1, 3]])
    objects = torch.eye(8)[[0, 3, 6]].sum(0).view(1, 1, 8)  # one object
    both = torch.cat([objects, torch.eye(8)[[1, 4, 7]].sum(0).view(1, 1, 8)], 1)
    out = listener(tokens, both)
    assert out.probs.shape == (1, 2)
    assert torch.allclose(out.probs.sum(-1), torch.ones(1))
    with pytest.raises(ValueError):
        CommunicationConfig(listener_type="attention")
