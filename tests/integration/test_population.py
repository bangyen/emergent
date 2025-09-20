"""Tests for population management and cultural transmission.

This module contains tests for the population-based cultural transmission
system, including agent turnover, cross-play interactions, and vocabulary statistics.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from typing import Any

from langlab.experiments.population import (
    PopulationConfig,
    AgentPair,
    PopulationManager,
    train_population,
)
from langlab.core.agents import Speaker, Listener
from langlab.core.config import CommunicationConfig


class TestPopulationConfig:
    """Test cases for PopulationConfig validation."""

    def test_valid_config(self) -> None:
        """Test that valid configuration parameters are accepted."""
        config = PopulationConfig(
            n_pairs=3,
            lifespan=100,
            replacement_noise=0.05,
            crossplay_prob=0.2,
        )
        assert config.n_pairs == 3
        assert config.lifespan == 100
        assert config.replacement_noise == 0.05
        assert config.crossplay_prob == 0.2

    def test_invalid_n_pairs(self) -> None:
        """Test that invalid n_pairs raises ValueError."""
        with pytest.raises(ValueError, match="n_pairs must be positive"):
            PopulationConfig(n_pairs=0)

    def test_invalid_lifespan(self) -> None:
        """Test that invalid lifespan raises ValueError."""
        with pytest.raises(ValueError, match="lifespan must be positive"):
            PopulationConfig(lifespan=0)

    def test_invalid_replacement_noise(self) -> None:
        """Test that negative replacement_noise raises ValueError."""
        with pytest.raises(ValueError, match="replacement_noise must be non-negative"):
            PopulationConfig(replacement_noise=-0.1)

    def test_invalid_crossplay_prob(self) -> None:
        """Test that invalid crossplay_prob raises ValueError."""
        with pytest.raises(ValueError, match="crossplay_prob must be between 0 and 1"):
            PopulationConfig(crossplay_prob=1.5)


class TestAgentPair:
    """Test cases for AgentPair functionality."""

    def test_agent_pair_initialization(self) -> None:
        """Test that AgentPair initializes correctly."""
        config = CommunicationConfig(
            vocabulary_size=5, message_length=1, hidden_size=32
        )
        speaker = Speaker(config)
        listener = Listener(config)

        pair = AgentPair(speaker, listener, lifespan=50, learning_rate=0.001, pair_id=0)

        assert pair.age == 0
        assert pair.lifespan == 50
        assert pair.pair_id == 0
        assert not pair.is_expired()
        assert len(pair.accuracy_history) == 0
        assert len(pair.vocab_usage) == 0

    def test_age_up(self) -> None:
        """Test that age_up increments age correctly."""
        config = CommunicationConfig(
            vocabulary_size=5, message_length=1, hidden_size=32
        )
        speaker = Speaker(config)
        listener = Listener(config)

        pair = AgentPair(speaker, listener, lifespan=10, learning_rate=0.001, pair_id=0)

        assert pair.age == 0
        pair.age_up()
        assert pair.age == 1
        pair.age_up()
        assert pair.age == 2

    def test_is_expired(self) -> None:
        """Test that is_expired returns correct values."""
        config = CommunicationConfig(
            vocabulary_size=5, message_length=1, hidden_size=32
        )
        speaker = Speaker(config)
        listener = Listener(config)

        pair = AgentPair(speaker, listener, lifespan=2, learning_rate=0.001, pair_id=0)

        assert not pair.is_expired()
        pair.age_up()
        assert not pair.is_expired()
        pair.age_up()
        assert pair.is_expired()

    def test_update_metrics(self) -> None:
        """Test that update_metrics updates tracking correctly."""
        config = CommunicationConfig(
            vocabulary_size=5, message_length=1, hidden_size=32
        )
        speaker = Speaker(config)
        listener = Listener(config)

        pair = AgentPair(speaker, listener, lifespan=10, learning_rate=0.001, pair_id=0)

        # Mock message tokens
        message_tokens = torch.tensor([[1, 2, 3], [0, 4, 1]])

        pair.update_metrics(0.8, message_tokens)

        assert pair.accuracy_history == [0.8]
        assert pair.vocab_usage[1] == 2  # token 1 appears twice
        assert pair.vocab_usage[2] == 1
        assert pair.vocab_usage[3] == 1
        assert pair.vocab_usage[0] == 1
        assert pair.vocab_usage[4] == 1

    def test_get_vocab_histogram(self) -> None:
        """Test that get_vocab_histogram returns correct histogram."""
        config = CommunicationConfig(
            vocabulary_size=5, message_length=1, hidden_size=32
        )
        speaker = Speaker(config)
        listener = Listener(config)

        pair = AgentPair(speaker, listener, lifespan=10, learning_rate=0.001, pair_id=0)

        # Add some vocabulary usage
        pair.vocab_usage[0] = 5
        pair.vocab_usage[1] = 3
        pair.vocab_usage[2] = 0
        pair.vocab_usage[3] = 2
        pair.vocab_usage[4] = 1

        histogram = pair.get_vocab_histogram(5)

        assert histogram == [5, 3, 0, 2, 1]


class TestPopulationManager:
    """Test cases for PopulationManager functionality."""

    def test_population_initialization(self) -> None:
        """Test that PopulationManager initializes correctly."""
        config = PopulationConfig(n_pairs=3, lifespan=100, seed=42)

        with patch(
            "langlab.experiments.population.get_device",
            return_value=torch.device("cpu"),
        ):
            population = PopulationManager(config)

        assert len(population.pairs) == 3
        assert population.step == 0
        assert len(population.replacement_log) == 0

        # Check that all pairs are initialized correctly
        for i, pair in enumerate(population.pairs):
            assert pair.pair_id == i
            assert pair.age == 0
            assert pair.lifespan == 100

    def test_turnover_occurs(self) -> None:
        """Test that agent turnover occurs after lifespan T."""
        config = PopulationConfig(n_pairs=2, lifespan=5, seed=42)

        with patch(
            "langlab.experiments.population.get_device",
            return_value=torch.device("cpu"),
        ):
            population = PopulationManager(config)

        # Age up pairs to trigger replacement
        for _ in range(6):  # lifespan + 1
            for pair in population.pairs:
                pair.age_up()
            population._check_and_replace_expired_pairs()

        # Check that at least one replacement occurred
        assert len(population.replacement_log) > 0

        # Verify replacement log structure
        replacement = population.replacement_log[0]
        assert "step" in replacement
        assert "pair_id" in replacement
        assert "old_age" in replacement
        assert replacement["old_age"] >= config.lifespan

    def test_crossplay_flag(self) -> None:
        """Test that crossplay probability controls cross-pair interactions."""
        # Test with crossplay_prob = 0
        config_no_crossplay = PopulationConfig(n_pairs=3, crossplay_prob=0.0, seed=42)

        with patch(
            "langlab.experiments.population.get_device",
            return_value=torch.device("cpu"),
        ):
            population_no_crossplay = PopulationManager(config_no_crossplay)

        interactions_no_crossplay = population_no_crossplay._select_interaction_pairs()

        # With crossplay_prob=0, should only have self-play
        assert len(interactions_no_crossplay) == 3
        for interaction in interactions_no_crossplay:
            assert interaction[0] == interaction[1]  # self-play only

        # Test with crossplay_prob = 0.5 (mock random to ensure cross-play)
        config_crossplay = PopulationConfig(n_pairs=3, crossplay_prob=0.5, seed=42)

        with patch(
            "langlab.experiments.population.get_device",
            return_value=torch.device("cpu"),
        ):
            population_crossplay = PopulationManager(config_crossplay)

        # Mock random.random to return 0.3 (below 0.5 threshold) for cross-play
        with patch("random.random", return_value=0.3):
            interactions_crossplay = population_crossplay._select_interaction_pairs()

        # Should have self-play + some cross-play
        assert len(interactions_crossplay) >= 3  # At least self-play
        # Check that there's at least one cross-play interaction
        cross_play_found = any(
            interaction[0] != interaction[1] for interaction in interactions_crossplay
        )
        assert cross_play_found

    def test_get_population_stats(self) -> None:
        """Test that get_population_stats returns correct statistics."""
        config = PopulationConfig(n_pairs=2, lifespan=100, seed=42)

        with patch(
            "langlab.experiments.population.get_device",
            return_value=torch.device("cpu"),
        ):
            population = PopulationManager(config)

        # Add some mock data to pairs
        population.pairs[0].age = 10
        population.pairs[0].accuracy_history = [0.8, 0.9]
        population.pairs[0].vocab_usage = {0: 5, 1: 3}

        population.pairs[1].age = 20
        population.pairs[1].accuracy_history = [0.7, 0.85]
        population.pairs[1].vocab_usage = {1: 2, 2: 4}

        stats = population.get_population_stats()

        assert stats["step"] == 0
        assert stats["ages"] == [10, 20]
        assert stats["accuracies"] == [0.9, 0.85]  # Last accuracy values
        assert stats["avg_age"] == 15.0
        assert stats["avg_accuracy"] == 0.875
        assert len(stats["vocab_histograms"]) == 2
        assert stats["total_replacements"] == 0


class TestTrainPopulation:
    """Test cases for train_population function."""

    @patch("langlab.experiments.population.ReferentialGameDataset")
    @patch("langlab.experiments.population.DataLoader")
    @patch("langlab.experiments.population.PopulationManager")
    @patch("langlab.experiments.population.set_seed")
    @patch("langlab.experiments.population.os.makedirs")
    def test_train_population_basic(
        self,
        mock_makedirs: Any,
        mock_set_seed: Any,
        mock_pop_manager_class: Any,
        mock_dataloader_class: Any,
        mock_dataset_class: Any,
    ) -> None:
        """Test that train_population runs without errors."""
        # Mock the population manager
        mock_pop_manager = MagicMock()
        mock_pop_manager_class.return_value = mock_pop_manager

        # Mock training step to return metrics
        mock_pop_manager.train_step.return_value = {
            "total_loss": 0.5,
            "accuracy": 0.8,
            "interactions": 5,
            "replacements": 0,
        }

        # Mock population stats
        mock_pop_manager.get_population_stats.return_value = {
            "avg_age": 10.0,
            "avg_accuracy": 0.8,
            "total_replacements": 0,
        }

        # Mock dataset and dataloader
        mock_dataset = MagicMock()
        mock_dataset_class.return_value = mock_dataset

        mock_dataloader = MagicMock()
        mock_dataloader_class.return_value = mock_dataloader

        # Mock batch data
        mock_batch = (
            torch.randn(32, 5, 10),  # scene_tensor
            torch.randint(0, 5, (32,)),  # target_indices
            torch.randn(32, 5, 10),  # candidate_objects
        )

        # Mock dataloader iteration
        mock_dataloader.__iter__.return_value = iter(
            [mock_batch] * 3
        )  # 3 batches for 3 steps

        # Run training
        train_population(
            n_steps=3,
            n_pairs=2,
            lifespan=100,
            crossplay_prob=0.1,
            replacement_noise=0.1,
            k=5,
            v=10,
            message_length=1,
            seed=42,
        )

        # Verify that training was called
        assert mock_pop_manager.train_step.call_count == 3
        assert mock_pop_manager.get_population_stats.call_count == 3
        mock_pop_manager.save_logs.assert_called_once()
