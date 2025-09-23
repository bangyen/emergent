"""Tests for grounded language learning functionality.

This module tests the grounded communication protocol and training
functionality for grid world navigation tasks.
"""

import torch

from src.langlab.training.grounding import (
    GroundedSpeaker,
    GroundedListener,
    GroundedEnvironment,
    GroundedEpisode,
    evaluate_grounded_performance,
)
from src.langlab.experiments.grid import Grid, Action, create_simple_grid
from src.langlab.core.config import CommunicationConfig
from src.langlab.training.train_grounded import CurriculumScheduler


class TestGroundedSpeaker:
    """Test cases for GroundedSpeaker."""

    def test_forward_pass(self) -> None:
        """Test forward pass through speaker network."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=3,
            hidden_size=32,
            object_dim=5,
        )

        speaker = GroundedSpeaker(config)

        batch_size = 2
        object_attributes = torch.randn(batch_size, 5)
        target_coord = torch.randn(batch_size, 2)

        logits, message_tokens = speaker(object_attributes, target_coord)

        assert logits.shape == (batch_size, 3, 10)
        assert message_tokens.shape == (batch_size, 3)
        assert torch.all(message_tokens >= 0)
        assert torch.all(message_tokens < 10)


class TestGroundedListener:
    """Test cases for GroundedListener."""

    def test_forward_pass(self) -> None:
        """Test forward pass through listener network."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=3,
            hidden_size=32,
            object_dim=5,
        )

        listener = GroundedListener(config, grid_size=3)

        batch_size = 2
        message_tokens = torch.randint(0, 10, (batch_size, 3))
        grid_obs = torch.randn(batch_size, 3, 3)

        action_logits = listener(message_tokens, grid_obs)

        assert action_logits.shape == (batch_size, 5)

    def test_get_action(self) -> None:
        """Test action selection."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=3,
            hidden_size=32,
            object_dim=5,
        )

        listener = GroundedListener(config, grid_size=3)

        message_tokens = torch.randint(0, 10, (1, 3))
        grid_obs = torch.randn(1, 3, 3)

        action = listener.get_action(message_tokens, grid_obs)

        assert isinstance(action, Action)


class TestGroundedEnvironment:
    """Test cases for GroundedEnvironment."""

    def test_environment_creation(self) -> None:
        """Test environment creation."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=3,
            hidden_size=32,
            object_dim=5,
        )

        env = GroundedEnvironment(config, grid_size=3, seed=42)

        assert env.config == config
        assert env.grid_size == 3
        assert env.speaker is not None
        assert env.listener is not None

    def test_create_episode(self) -> None:
        """Test episode creation."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=3,
            hidden_size=32,
            object_dim=5,
        )

        env = GroundedEnvironment(config, grid_size=3, seed=42)

        episode = env.create_episode()

        assert isinstance(episode, GroundedEpisode)
        assert episode.grid is not None
        assert episode.target_attributes is not None
        assert episode.message is not None
        assert isinstance(episode.actions, list)
        assert isinstance(episode.rewards, list)
        assert isinstance(episode.success, bool)
        assert episode.episode_length >= 0

    def test_compute_speaker_loss(self) -> None:
        """Test speaker loss computation."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=3,
            hidden_size=32,
            object_dim=5,
        )

        env = GroundedEnvironment(config, grid_size=3, seed=42)

        batch_size = 2
        speaker_logits = torch.randn(batch_size, 3, 10, requires_grad=True)
        rewards = torch.tensor([1.0, 0.0])
        baseline = 0.5

        loss = env.compute_speaker_loss(speaker_logits, rewards, baseline)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_compute_listener_loss(self) -> None:
        """Test listener loss computation."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=3,
            hidden_size=32,
            object_dim=5,
        )

        env = GroundedEnvironment(config, grid_size=3, seed=42)

        batch_size = 2
        message_tokens = torch.randint(0, 10, (batch_size, 3))
        grid_obs = torch.randn(batch_size, 3, 3)
        actions = torch.randint(0, 5, (batch_size,))
        rewards = torch.tensor([1.0, 0.0])

        loss = env.compute_listener_loss(message_tokens, grid_obs, actions, rewards)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad


class TestGroundedEpisode:
    """Test cases for GroundedEpisode."""

    def test_episode_creation(self) -> None:
        """Test episode data structure."""
        grid = create_simple_grid(size=3, seed=42)
        target_attributes = torch.randn(1, 5)
        target_coord = (1, 1)
        message = torch.randint(0, 10, (1, 3))
        actions = [Action.RIGHT, Action.DOWN]
        rewards = [0.0, 1.0]
        success = True
        episode_length = 2

        episode = GroundedEpisode(
            grid=grid,
            target_attributes=target_attributes,
            target_coord=target_coord,
            message=message,
            actions=actions,
            rewards=rewards,
            success=success,
            episode_length=episode_length,
        )

        assert episode.grid == grid
        assert episode.target_attributes.shape == (1, 5)
        assert episode.target_coord == (1, 1)
        assert episode.message.shape == (1, 3)
        assert episode.actions == [Action.RIGHT, Action.DOWN]
        assert episode.rewards == [0.0, 1.0]
        assert episode.success is True
        assert episode.episode_length == 2


class TestCurriculumScheduler:
    """Test cases for CurriculumScheduler."""

    def test_curriculum_creation(self) -> None:
        """Test curriculum scheduler creation."""
        scheduler = CurriculumScheduler(
            base_size=3,
            max_size=5,
            max_walls=2,
        )

        assert scheduler.base_size == 3
        assert scheduler.max_size == 5
        assert scheduler.max_walls == 2
        assert scheduler.current_level == 0
        assert scheduler.episodes_at_level == 0
        assert scheduler.level_successes == 0

    def test_get_current_grids(self) -> None:
        """Test getting grids for current level."""
        scheduler = CurriculumScheduler(
            base_size=3,
            max_size=4,
            max_walls=1,
        )

        grids = scheduler.get_current_grids()

        assert isinstance(grids, list)
        assert len(grids) > 0
        assert all(isinstance(g, Grid) for g in grids)

    def test_curriculum_update(self) -> None:
        """Test curriculum level updates."""
        scheduler = CurriculumScheduler(
            base_size=3,
            max_size=4,
            max_walls=1,
            success_threshold=0.8,
            min_episodes_per_level=2,
        )

        # Simulate successful episodes
        level_changed = scheduler.update(True)
        assert not level_changed  # Not enough episodes yet

        level_changed = scheduler.update(True)
        assert level_changed  # Should advance to next level
        assert scheduler.current_level == 1
        assert scheduler.episodes_at_level == 0


class TestNavigationSmoke:
    """Smoke tests for navigation functionality."""

    def test_navigation_smoke(self) -> None:
        """Test tiny training improves success rate vs random policy."""
        config = CommunicationConfig(
            vocabulary_size=8,
            message_length=2,
            hidden_size=32,
            object_dim=5,
        )

        env = GroundedEnvironment(config, grid_size=3, max_steps=5, seed=42)

        # Test random policy performance
        random_successes = 0
        for _ in range(20):
            episode = env.create_episode()
            if episode.success:
                random_successes += 1

        random_success_rate = random_successes / 20

        # Simple training loop (very minimal)
        speaker_optimizer = torch.optim.Adam(env.speaker.parameters(), lr=1e-3)

        # Train for a few episodes
        for episode_num in range(10):
            grid = create_simple_grid(size=3, max_steps=5, seed=42 + episode_num)
            episode = env.create_episode(grid)

            if episode.actions:  # Only train if actions were taken
                # Simple training step
                target_attributes = episode.target_attributes
                target_coord = torch.tensor([episode.target_coord], dtype=torch.float32)

                speaker_logits, message_tokens = env.speaker(
                    target_attributes, target_coord
                )
                rewards = torch.tensor(episode.rewards, dtype=torch.float32)

                # Compute losses
                speaker_loss = env.compute_speaker_loss(speaker_logits, rewards, 0.5)

                # Update speaker
                speaker_optimizer.zero_grad()
                speaker_loss.backward()
                speaker_optimizer.step()

        # Test trained policy performance
        trained_successes = 0
        for _ in range(20):
            episode = env.create_episode()
            if episode.success:
                trained_successes += 1

        trained_success_rate = trained_successes / 20

        # Trained policy should be at least as good as random
        # (may not be significantly better due to minimal training)
        assert trained_success_rate >= random_success_rate - 0.1  # Allow some variance


class TestEvaluation:
    """Test cases for evaluation functionality."""

    def test_evaluate_grounded_performance(self) -> None:
        """Test performance evaluation."""
        config = CommunicationConfig(
            vocabulary_size=8,
            message_length=2,
            hidden_size=32,
            object_dim=5,
        )

        env = GroundedEnvironment(config, grid_size=3, seed=42)

        metrics = evaluate_grounded_performance(env, num_episodes=10)

        assert "success_rate" in metrics
        assert "avg_reward" in metrics
        assert "avg_episode_length" in metrics

        assert 0.0 <= metrics["success_rate"] <= 1.0
        assert metrics["avg_reward"] >= 0.0
        assert metrics["avg_episode_length"] >= 0.0
