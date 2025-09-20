"""Tests for Speaker and Listener agents.

This module contains unit tests for the neural network agents used in
referential games, verifying correct output shapes and behavior.
"""

import pytest
import torch

from langlab.agents import Speaker, Listener
from langlab.channel import DiscreteChannel
from langlab.config import CommunicationConfig
from langlab.world import TOTAL_ATTRIBUTES


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


@pytest.fixture
def channel(config: CommunicationConfig) -> DiscreteChannel:
    """Create a DiscreteChannel for testing."""
    return DiscreteChannel(config)


def test_speaker_output_shapes(speaker: Speaker, config: CommunicationConfig) -> None:
    """Test that Speaker produces correct output shapes for A=7, V=10, L=1."""
    batch_size = 3
    input_dim = TOTAL_ATTRIBUTES  # A=7 (3 colors + 3 shapes + 2 sizes)

    # Create test input
    object_encoding = torch.randn(batch_size, input_dim)

    # Forward pass
    logits, token_ids = speaker(object_encoding)

    # Check logits shape: (batch_size, message_length, vocabulary_size)
    expected_logits_shape = (batch_size, config.message_length, config.vocabulary_size)
    assert (
        logits.shape == expected_logits_shape
    ), f"Expected {expected_logits_shape}, got {logits.shape}"

    # Check token_ids shape: (batch_size, message_length)
    expected_token_shape = (batch_size, config.message_length)
    assert (
        token_ids.shape == expected_token_shape
    ), f"Expected {expected_token_shape}, got {token_ids.shape}"

    # Check that token_ids are integers
    assert token_ids.dtype in [
        torch.int64,
        torch.int32,
        torch.long,
    ], f"Expected integer dtype, got {token_ids.dtype}"


def test_listener_output_shapes(
    listener: Listener, config: CommunicationConfig
) -> None:
    """Test that Listener produces correct output shapes for K=5 candidates."""
    batch_size = 2
    num_candidates = 5  # K=5
    message_length = config.message_length
    object_dim = TOTAL_ATTRIBUTES

    # Create test inputs
    message_tokens = torch.randint(
        0, config.vocabulary_size, (batch_size, message_length)
    )
    candidate_objects = torch.randn(batch_size, num_candidates, object_dim)

    # Forward pass
    probabilities = listener(message_tokens, candidate_objects)

    # Check probabilities shape: (batch_size, num_candidates)
    expected_shape = (batch_size, num_candidates)
    assert (
        probabilities.shape == expected_shape
    ), f"Expected {expected_shape}, got {probabilities.shape}"

    # Check that probabilities sum to 1 for each batch
    prob_sums = probabilities.sum(dim=-1)
    assert torch.allclose(
        prob_sums, torch.ones(batch_size), atol=1e-6
    ), f"Probabilities should sum to 1, got sums: {prob_sums}"

    # Check that all probabilities are non-negative
    assert torch.all(probabilities >= 0), "All probabilities should be non-negative"


def test_channel_token_range(
    channel: DiscreteChannel, config: CommunicationConfig
) -> None:
    """Test that channel enforces token range [0, V-1]."""
    batch_size = 4
    message_length = config.message_length
    vocab_size = config.vocabulary_size

    # Create test logits
    speaker_logits = torch.randn(batch_size, message_length, vocab_size)

    # Send through channel
    token_ids = channel.send(speaker_logits)

    # Check token range
    min_token = token_ids.min().item()
    max_token = token_ids.max().item()

    assert min_token >= 0, f"Minimum token should be >= 0, got {min_token}"
    assert (
        max_token < vocab_size
    ), f"Maximum token should be < {vocab_size}, got {max_token}"

    # Test validation method
    assert channel.validate_tokens(token_ids), "Channel should validate its own tokens"

    # Test invalid tokens
    invalid_tokens = torch.tensor([[-1], [vocab_size]])
    assert not channel.validate_tokens(
        invalid_tokens
    ), "Channel should reject invalid tokens"


def test_channel_token_range_edge_cases(
    channel: DiscreteChannel, config: CommunicationConfig
) -> None:
    """Test channel behavior with edge case logits."""
    # Test with extreme logits
    extreme_logits = torch.tensor(
        [
            [
                [
                    100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                ]
            ],
            [
                [
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    100.0,
                ]
            ],
        ]
    )

    token_ids = channel.send(extreme_logits)

    # Should select the highest logit (first and last tokens)
    expected_tokens = torch.tensor([[0], [config.vocabulary_size - 1]])
    assert torch.equal(
        token_ids, expected_tokens
    ), f"Expected {expected_tokens}, got {token_ids}"


def test_speaker_training_mode(speaker: Speaker, config: CommunicationConfig) -> None:
    """Test Speaker behavior in training vs evaluation mode."""
    batch_size = 2
    input_dim = TOTAL_ATTRIBUTES
    object_encoding = torch.randn(batch_size, input_dim)

    # Set seed for reproducible results
    torch.manual_seed(42)

    # Training mode
    speaker.train()
    logits_train, tokens_train = speaker(object_encoding)

    # Reset seed and evaluation mode
    torch.manual_seed(42)
    speaker.eval()
    logits_eval, tokens_eval = speaker(object_encoding)

    # Logits should be the same (deterministic forward pass)
    assert torch.allclose(
        logits_train, logits_eval
    ), "Logits should be identical in train/eval mode"

    # Tokens may differ due to Gumbel noise in training mode, but both should be valid
    assert (
        speaker.config.vocabulary_size > tokens_train.max().item()
    ), "Training tokens should be valid"
    assert (
        speaker.config.vocabulary_size > tokens_eval.max().item()
    ), "Evaluation tokens should be valid"
    assert tokens_train.min().item() >= 0, "Training tokens should be non-negative"
    assert tokens_eval.min().item() >= 0, "Evaluation tokens should be non-negative"


def test_listener_probability_distribution(
    listener: Listener, config: CommunicationConfig
) -> None:
    """Test that Listener produces valid probability distributions."""
    batch_size = 3
    num_candidates = 4
    message_length = config.message_length
    object_dim = TOTAL_ATTRIBUTES

    # Create test inputs
    message_tokens = torch.randint(
        0, config.vocabulary_size, (batch_size, message_length)
    )
    candidate_objects = torch.randn(batch_size, num_candidates, object_dim)

    # Forward pass
    probabilities = listener(message_tokens, candidate_objects)

    # Check probability properties
    assert torch.all(probabilities >= 0), "All probabilities must be non-negative"
    assert torch.all(probabilities <= 1), "All probabilities must be <= 1"

    # Check that each row sums to 1
    row_sums = probabilities.sum(dim=-1)
    assert torch.allclose(
        row_sums, torch.ones(batch_size), atol=1e-6
    ), f"Each row should sum to 1, got sums: {row_sums}"


def test_agent_device_compatibility(
    speaker: Speaker, listener: Listener, config: CommunicationConfig
) -> None:
    """Test that agents work on both CPU and GPU (if available)."""
    batch_size = 2
    input_dim = TOTAL_ATTRIBUTES
    num_candidates = 3

    # Test on CPU
    object_encoding_cpu = torch.randn(batch_size, input_dim)
    message_tokens_cpu = torch.randint(
        0, config.vocabulary_size, (batch_size, config.message_length)
    )
    candidates_cpu = torch.randn(batch_size, num_candidates, input_dim)

    # Speaker on CPU
    logits_cpu, tokens_cpu = speaker(object_encoding_cpu)
    assert logits_cpu.device.type == "cpu"
    assert tokens_cpu.device.type == "cpu"

    # Listener on CPU
    probs_cpu = listener(message_tokens_cpu, candidates_cpu)
    assert probs_cpu.device.type == "cpu"

    # Test on GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")

        # Move agents to GPU
        speaker_gpu = speaker.to(device)
        listener_gpu = listener.to(device)

        # Create GPU tensors
        object_encoding_gpu = object_encoding_cpu.to(device)
        message_tokens_gpu = message_tokens_cpu.to(device)
        candidates_gpu = candidates_cpu.to(device)

        # Speaker on GPU
        logits_gpu, tokens_gpu = speaker_gpu(object_encoding_gpu)
        assert logits_gpu.device.type == "cuda"
        assert tokens_gpu.device.type == "cuda"

        # Listener on GPU
        probs_gpu = listener_gpu(message_tokens_gpu, candidates_gpu)
        assert probs_gpu.device.type == "cuda"
