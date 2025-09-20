"""Tests for vocabulary statistics and usage tracking.

This module contains tests for vocabulary usage tracking and histogram
generation in the population-based cultural transmission system.
"""

import torch

from langlab.population import AgentPair
from langlab.agents import Speaker, Listener
from langlab.config import CommunicationConfig


class TestVocabStats:
    """Test cases for vocabulary statistics functionality."""

    def test_histogram_shape(self) -> None:
        """Test that vocab histogram has length V and sums to message count."""
        config = CommunicationConfig(
            vocabulary_size=10, message_length=1, hidden_size=32
        )
        speaker = Speaker(config)
        listener = Listener(config)

        pair = AgentPair(
            speaker, listener, lifespan=100, learning_rate=0.001, pair_id=0
        )

        # Add vocabulary usage
        pair.vocab_usage[0] = 5
        pair.vocab_usage[1] = 3
        pair.vocab_usage[2] = 0
        pair.vocab_usage[3] = 2
        pair.vocab_usage[4] = 1
        pair.vocab_usage[5] = 4
        pair.vocab_usage[6] = 0
        pair.vocab_usage[7] = 3
        pair.vocab_usage[8] = 1
        pair.vocab_usage[9] = 2

        histogram = pair.get_vocab_histogram(10)

        # Test histogram shape
        assert len(histogram) == 10  # Should have length V

        # Test histogram content
        expected_histogram = [5, 3, 0, 2, 1, 4, 0, 3, 1, 2]
        assert histogram == expected_histogram

        # Test that histogram sums to total message count
        total_messages = sum(pair.vocab_usage.values())
        histogram_sum = sum(histogram)
        assert histogram_sum == total_messages
        assert histogram_sum == 21  # 5+3+0+2+1+4+0+3+1+2

    def test_histogram_with_partial_vocab(self) -> None:
        """Test histogram when only some vocabulary tokens are used."""
        config = CommunicationConfig(
            vocabulary_size=8, message_length=1, hidden_size=32
        )
        speaker = Speaker(config)
        listener = Listener(config)

        pair = AgentPair(
            speaker, listener, lifespan=100, learning_rate=0.001, pair_id=0
        )

        # Add vocabulary usage for only some tokens
        pair.vocab_usage[0] = 10
        pair.vocab_usage[2] = 5
        pair.vocab_usage[4] = 3
        pair.vocab_usage[6] = 2

        histogram = pair.get_vocab_histogram(8)

        # Test histogram shape
        assert len(histogram) == 8

        # Test histogram content
        expected_histogram = [10, 0, 5, 0, 3, 0, 2, 0]
        assert histogram == expected_histogram

        # Test sum
        total_messages = sum(pair.vocab_usage.values())
        histogram_sum = sum(histogram)
        assert histogram_sum == total_messages
        assert histogram_sum == 20  # 10+0+5+0+3+0+2+0

    def test_histogram_with_out_of_bounds_tokens(self) -> None:
        """Test histogram when vocab_usage contains tokens outside vocabulary range."""
        config = CommunicationConfig(
            vocabulary_size=5, message_length=1, hidden_size=32
        )
        speaker = Speaker(config)
        listener = Listener(config)

        pair = AgentPair(
            speaker, listener, lifespan=100, learning_rate=0.001, pair_id=0
        )

        # Add vocabulary usage including out-of-bounds tokens
        pair.vocab_usage[0] = 3
        pair.vocab_usage[1] = 2
        pair.vocab_usage[2] = 1
        pair.vocab_usage[3] = 4
        pair.vocab_usage[4] = 1
        pair.vocab_usage[5] = 10  # Out of bounds
        pair.vocab_usage[6] = 5  # Out of bounds
        pair.vocab_usage[-1] = 2  # Out of bounds

        histogram = pair.get_vocab_histogram(5)

        # Test histogram shape
        assert len(histogram) == 5

        # Test histogram content (should ignore out-of-bounds tokens)
        expected_histogram = [3, 2, 1, 4, 1]
        assert histogram == expected_histogram

        # Test sum (should only include in-bounds tokens)
        histogram_sum = sum(histogram)
        assert histogram_sum == 11  # 3+2+1+4+1

    def test_empty_vocab_usage(self) -> None:
        """Test histogram when no vocabulary tokens have been used."""
        config = CommunicationConfig(
            vocabulary_size=7, message_length=1, hidden_size=32
        )
        speaker = Speaker(config)
        listener = Listener(config)

        pair = AgentPair(
            speaker, listener, lifespan=100, learning_rate=0.001, pair_id=0
        )

        # No vocabulary usage added

        histogram = pair.get_vocab_histogram(7)

        # Test histogram shape
        assert len(histogram) == 7

        # Test histogram content (should be all zeros)
        expected_histogram = [0, 0, 0, 0, 0, 0, 0]
        assert histogram == expected_histogram

        # Test sum
        histogram_sum = sum(histogram)
        assert histogram_sum == 0

    def test_vocab_usage_tracking(self) -> None:
        """Test that vocabulary usage is tracked correctly during training."""
        config = CommunicationConfig(
            vocabulary_size=6, message_length=2, hidden_size=32
        )
        speaker = Speaker(config)
        listener = Listener(config)

        pair = AgentPair(
            speaker, listener, lifespan=100, learning_rate=0.001, pair_id=0
        )

        # Simulate multiple training steps with different messages
        messages = [
            torch.tensor([[0, 1], [2, 3]]),  # Batch 1
            torch.tensor([[1, 2], [3, 4]]),  # Batch 2
            torch.tensor([[0, 5], [1, 0]]),  # Batch 3
        ]

        for message_tokens in messages:
            pair.update_metrics(0.8, message_tokens)

        # Check vocabulary usage
        assert pair.vocab_usage[0] == 3  # Appears in batch 1 and 3
        assert pair.vocab_usage[1] == 3  # Appears in batch 1, 2, and 3
        assert pair.vocab_usage[2] == 2  # Appears in batch 1 and 2
        assert pair.vocab_usage[3] == 2  # Appears in batch 1 and 2
        assert pair.vocab_usage[4] == 1  # Appears in batch 2
        assert pair.vocab_usage[5] == 1  # Appears in batch 3

        # Test histogram
        histogram = pair.get_vocab_histogram(6)
        expected_histogram = [3, 3, 2, 2, 1, 1]
        assert histogram == expected_histogram

        # Test sum
        total_tokens = sum(pair.vocab_usage.values())
        histogram_sum = sum(histogram)
        assert histogram_sum == total_tokens
        assert histogram_sum == 12  # 3+3+2+2+1+1

    def test_histogram_normalization(self) -> None:
        """Test that histogram can be normalized to probabilities."""
        config = CommunicationConfig(
            vocabulary_size=4, message_length=1, hidden_size=32
        )
        speaker = Speaker(config)
        listener = Listener(config)

        pair = AgentPair(
            speaker, listener, lifespan=100, learning_rate=0.001, pair_id=0
        )

        # Add vocabulary usage
        pair.vocab_usage[0] = 10
        pair.vocab_usage[1] = 5
        pair.vocab_usage[2] = 0
        pair.vocab_usage[3] = 5

        histogram = pair.get_vocab_histogram(4)

        # Test histogram
        expected_histogram = [10, 5, 0, 5]
        assert histogram == expected_histogram

        # Test normalization to probabilities
        total = sum(histogram)
        probabilities = [count / total for count in histogram]
        expected_probabilities = [0.5, 0.25, 0.0, 0.25]

        for i, (actual, expected) in enumerate(
            zip(probabilities, expected_probabilities)
        ):
            assert abs(actual - expected) < 1e-10, f"Probability mismatch at index {i}"

        # Test that probabilities sum to 1
        assert abs(sum(probabilities) - 1.0) < 1e-10
