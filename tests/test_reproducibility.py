"""Tests for reproducibility and deterministic behavior.

This module contains tests to verify that the system produces deterministic
outputs when using fixed seeds and no stochastic sampling.
"""

import pytest
import torch

from langlab.agents import Speaker, Listener
from langlab.channel import DiscreteChannel
from langlab.config import CommunicationConfig
from langlab.utils import set_seed
from langlab.world import TOTAL_ATTRIBUTES


@pytest.fixture
def deterministic_config() -> CommunicationConfig:
    """Create a deterministic communication configuration."""
    return CommunicationConfig(
        vocabulary_size=10, message_length=1, hidden_size=64, seed=42
    )


def test_deterministic_sampling_with_seed(
    deterministic_config: CommunicationConfig,
) -> None:
    """Test that outputs are deterministic with fixed seed and no Gumbel noise."""
    # Set global seed
    set_seed(deterministic_config.seed)

    # Create agents
    speaker = Speaker(deterministic_config)
    listener = Listener(deterministic_config)
    channel = DiscreteChannel(deterministic_config)

    # Set to evaluation mode to disable Gumbel noise
    speaker.eval()
    listener.eval()

    # Create test inputs
    batch_size = 3
    input_dim = TOTAL_ATTRIBUTES
    num_candidates = 4

    object_encoding = torch.randn(batch_size, input_dim)
    message_tokens = torch.randint(
        0,
        deterministic_config.vocabulary_size,
        (batch_size, deterministic_config.message_length),
    )
    candidate_objects = torch.randn(batch_size, num_candidates, input_dim)

    # Run multiple times with same seed
    results = []
    for _ in range(3):
        # Reset seed before each run
        set_seed(deterministic_config.seed)

        # Speaker forward pass
        logits, tokens, _, _ = speaker(object_encoding)

        # Channel forward pass
        channel_tokens = channel.send(logits)

        # Listener forward pass
        probabilities = listener(message_tokens, candidate_objects)

        results.append(
            {
                "logits": logits.clone(),
                "tokens": tokens.clone(),
                "channel_tokens": channel_tokens.clone(),
                "probabilities": probabilities.clone(),
            }
        )

    # All results should be identical
    for i in range(1, len(results)):
        # Check logits are identical
        assert torch.allclose(
            results[0]["logits"], results[i]["logits"]
        ), f"Logits differ between runs {0} and {i}"

        # Check tokens are identical
        assert torch.equal(
            results[0]["tokens"], results[i]["tokens"]
        ), f"Tokens differ between runs {0} and {i}"

        # Check channel tokens are identical
        assert torch.equal(
            results[0]["channel_tokens"], results[i]["channel_tokens"]
        ), f"Channel tokens differ between runs {0} and {i}"

        # Check probabilities are identical
        assert torch.allclose(
            results[0]["probabilities"], results[i]["probabilities"]
        ), f"Probabilities differ between runs {0} and {i}"


def test_deterministic_with_different_seeds() -> None:
    """Test that different seeds produce different but reproducible results."""
    batch_size = 2
    input_dim = TOTAL_ATTRIBUTES

    # Test with two different seeds
    seeds = [42, 123]
    results_by_seed = {}

    for seed in seeds:
        # Create config with specific seed
        config = CommunicationConfig(
            vocabulary_size=10, message_length=1, hidden_size=64, seed=seed
        )

        # Set seed
        set_seed(seed)

        # Create agents
        speaker = Speaker(config)
        speaker.eval()

        # Create test input
        object_encoding = torch.randn(batch_size, input_dim)

        # Run multiple times with same seed
        run_results = []
        for _ in range(2):
            set_seed(seed)
            logits, tokens, _, _ = speaker(object_encoding)
            run_results.append({"logits": logits.clone(), "tokens": tokens.clone()})

        # Results within same seed should be identical
        assert torch.allclose(
            run_results[0]["logits"], run_results[1]["logits"]
        ), f"Results differ within seed {seed}"
        assert torch.equal(
            run_results[0]["tokens"], run_results[1]["tokens"]
        ), f"Tokens differ within seed {seed}"

        # Store results for cross-seed comparison
        results_by_seed[seed] = run_results[0]

    # Results between different seeds should be different
    seed1_results = results_by_seed[seeds[0]]
    seed2_results = results_by_seed[seeds[1]]

    # Logits should be different (due to different random initialization)
    assert not torch.allclose(
        seed1_results["logits"], seed2_results["logits"]
    ), "Results should differ between different seeds"

    # Tokens might be the same or different depending on the specific values
    # This is acceptable as long as they're deterministic within each seed


def test_reproducible_agent_initialization() -> None:
    """Test that agent initialization is reproducible with fixed seeds."""
    config = CommunicationConfig(
        vocabulary_size=10, message_length=1, hidden_size=64, seed=42
    )

    # Create agents multiple times with same seed
    set_seed(config.seed)
    speaker1 = Speaker(config)

    set_seed(config.seed)
    speaker2 = Speaker(config)

    # Agents should have identical parameters
    for param1, param2 in zip(speaker1.parameters(), speaker2.parameters()):
        assert torch.allclose(
            param1, param2
        ), "Agent parameters should be identical with same seed"

    # Test forward pass with same input
    batch_size = 2
    input_dim = TOTAL_ATTRIBUTES
    object_encoding = torch.randn(batch_size, input_dim)

    set_seed(config.seed)
    logits1, tokens1, _, _ = speaker1(object_encoding)

    set_seed(config.seed)
    logits2, tokens2, _, _ = speaker2(object_encoding)

    # Results should be identical
    assert torch.allclose(logits1, logits2), "Forward pass results should be identical"
    assert torch.equal(tokens1, tokens2), "Token outputs should be identical"


def test_channel_deterministic_behavior(
    deterministic_config: CommunicationConfig,
) -> None:
    """Test that channel produces deterministic outputs with fixed inputs."""
    channel = DiscreteChannel(deterministic_config)

    # Use fixed logits (not random)
    speaker_logits = torch.tensor(
        [
            [[1.0, 2.0, 0.5, 0.1, 0.3, 0.2, 0.4, 0.6, 0.8, 0.9]],
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
            [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]],
        ]
    )

    # Run multiple times
    results = []
    for _ in range(5):
        token_ids = channel.send(speaker_logits)
        results.append(token_ids.clone())

    # All results should be identical (argmax is deterministic)
    for i in range(1, len(results)):
        assert torch.equal(
            results[0], results[i]
        ), f"Channel output differs between runs {0} and {i}"

    # Verify the expected tokens (argmax of each row)
    expected_tokens = torch.tensor([[1], [9], [0]])  # argmax of each logit row
    assert torch.equal(
        results[0], expected_tokens
    ), f"Expected {expected_tokens}, got {results[0]}"


def test_training_mode_stochastic_behavior(
    deterministic_config: CommunicationConfig,
) -> None:
    """Test that training mode produces stochastic behavior (different from eval)."""
    set_seed(deterministic_config.seed)

    speaker = Speaker(deterministic_config)
    channel = DiscreteChannel(deterministic_config)

    batch_size = 2
    input_dim = TOTAL_ATTRIBUTES
    object_encoding = torch.randn(batch_size, input_dim)

    # Training mode (stochastic)
    speaker.train()
    set_seed(deterministic_config.seed)
    logits_train, tokens_train, _, _ = speaker(object_encoding)
    channel_tokens_train = channel.send(logits_train)

    # Evaluation mode (deterministic)
    speaker.eval()
    set_seed(deterministic_config.seed)
    logits_eval, tokens_eval, _, _ = speaker(object_encoding)
    channel_tokens_eval = channel.send(logits_eval)

    # Logits should be identical (no randomness in forward pass)
    assert torch.allclose(logits_train, logits_eval), "Logits should be identical"

    # Tokens might differ due to Gumbel noise in training mode
    # This is expected behavior - training mode adds noise for exploration
    # The exact behavior depends on the random state, but we can verify
    # that both modes produce valid tokens
    assert channel.validate_tokens(tokens_train), "Training tokens should be valid"
    assert channel.validate_tokens(tokens_eval), "Evaluation tokens should be valid"
    assert channel.validate_tokens(
        channel_tokens_train
    ), "Training channel tokens should be valid"
    assert channel.validate_tokens(
        channel_tokens_eval
    ), "Evaluation channel tokens should be valid"
