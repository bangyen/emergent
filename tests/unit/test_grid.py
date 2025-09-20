"""Tests for grid world functionality.

This module tests the Grid class and related functionality for the
grounded language learning environment.
"""

import torch

from langlab.grid import Grid, Action, NavigationPolicy, create_simple_grid


class TestGrid:
    """Test cases for the Grid class."""

    def test_step_and_bounds(self) -> None:
        """Test that agent never leaves grid and walls block movement."""
        # Create a simple 3x3 grid with walls
        walls = [(1, 1)]  # Wall in the center
        grid = Grid(size=3, walls=walls, start_pos=(0, 0), target_pos=(2, 2))

        # Test that agent stays within bounds
        initial_pos = grid.agent_pos

        # Try to move up (should stay at (0,0) since already at top)
        state, reward, done, info = grid.step(Action.UP)
        assert grid.agent_pos == initial_pos  # Should not move

        # Try to move left (should stay at (0,0) since already at left edge)
        state, reward, done, info = grid.step(Action.LEFT)
        assert grid.agent_pos == initial_pos  # Should not move

        # Try to move right (should succeed)
        state, reward, done, info = grid.step(Action.RIGHT)
        assert grid.agent_pos == (1, 0)  # Should move right

        # Try to move down then up to test wall blocking
        state, reward, done, info = grid.step(Action.DOWN)
        assert grid.agent_pos == (1, 0)  # Should stay at (1,0) due to wall at (1,1)

        # Try to move up (should be blocked by wall at (1,1))
        state, reward, done, info = grid.step(Action.UP)
        assert grid.agent_pos == (1, 0)  # Should not move due to wall

    def test_episode_termination(self) -> None:
        """Test that success ends episode with reward=1."""
        # Create a simple 2x2 grid
        grid = Grid(size=2, start_pos=(0, 0), target_pos=(1, 1), max_steps=10)

        # Move to target
        state, reward, done, info = grid.step(Action.RIGHT)
        assert not done
        assert reward == 0.0

        state, reward, done, info = grid.step(Action.DOWN)
        assert done
        assert reward == 1.0
        assert grid.success
        assert info["target_reached"]

    def test_max_steps_termination(self) -> None:
        """Test that episode ends after max steps."""
        grid = Grid(size=3, start_pos=(0, 0), target_pos=(2, 2), max_steps=3)

        # Take max_steps actions without reaching target
        for _ in range(3):
            state, reward, done, info = grid.step(Action.STAY)

        assert done
        assert not grid.success
        assert reward == 0.0

    def test_reset(self) -> None:
        """Test that reset returns grid to initial state."""
        grid = Grid(size=3, start_pos=(0, 0), target_pos=(2, 2))

        # Take some actions
        grid.step(Action.RIGHT)
        grid.step(Action.DOWN)

        # Reset
        state = grid.reset()

        assert grid.agent_pos == (0, 0)  # Back to start
        assert grid.step_count == 0
        assert not grid.done
        assert not grid.success
        assert state["agent_pos"] == (0, 0)

    def test_render(self) -> None:
        """Test grid rendering."""
        grid = Grid(size=3, walls=[(1, 1)], start_pos=(0, 0), target_pos=(2, 2))

        rendered = grid.render()
        lines = rendered.split("\n")

        assert len(lines) == 3
        assert lines[0] == "A.."  # Agent at (0,0)
        assert lines[1] == ".#."  # Wall at (1,1)
        assert lines[2] == "..T"  # Target at (2,2)

    def test_get_observation(self) -> None:
        """Test observation tensor generation."""
        grid = Grid(size=3, walls=[(1, 1)], start_pos=(0, 0), target_pos=(2, 2))

        obs = grid.get_observation()

        assert obs.shape == (3, 3)
        assert obs[0, 0] == 2.0  # Agent
        assert obs[1, 1] == 1.0  # Wall
        assert obs[2, 2] == 3.0  # Target
        assert obs[0, 1] == 0.0  # Empty


class TestNavigationPolicy:
    """Test cases for the NavigationPolicy class."""

    def test_forward_pass(self) -> None:
        """Test forward pass through policy network."""
        policy = NavigationPolicy(
            grid_size=3,
            message_dim=10,
            hidden_size=32,
            num_actions=5,
        )

        batch_size = 2
        grid_obs = torch.randn(batch_size, 3, 3)
        message_embedding = torch.randn(batch_size, 10)

        action_logits = policy(grid_obs, message_embedding)

        assert action_logits.shape == (batch_size, 5)

    def test_get_action(self) -> None:
        """Test action selection."""
        policy = NavigationPolicy(
            grid_size=3,
            message_dim=10,
            hidden_size=32,
            num_actions=5,
        )

        grid_obs = torch.randn(1, 3, 3)
        message_embedding = torch.randn(1, 10)

        action = policy.get_action(grid_obs, message_embedding)

        assert isinstance(action, Action)
        assert action in [
            Action.UP,
            Action.DOWN,
            Action.LEFT,
            Action.RIGHT,
            Action.STAY,
        ]


class TestGridCreation:
    """Test cases for grid creation functions."""

    def test_create_simple_grid(self) -> None:
        """Test simple grid creation."""
        grid = create_simple_grid(size=4, num_walls=2, seed=42)

        assert grid.size == 4
        assert len(grid.walls) <= 2  # May be less if walls overlap
        assert grid.agent_pos != grid.target_pos
        assert not grid._is_wall(*grid.agent_pos)
        assert not grid._is_wall(*grid.target_pos)

    def test_create_simple_grid_reproducible(self) -> None:
        """Test that grid creation is reproducible with same seed."""
        grid1 = create_simple_grid(size=3, num_walls=1, seed=123)
        grid2 = create_simple_grid(size=3, num_walls=1, seed=123)

        # Should have same configuration
        assert grid1.size == grid2.size
        assert grid1.walls == grid2.walls
        assert grid1.start_pos == grid2.start_pos
        assert grid1.target_pos == grid2.target_pos


class TestActionEnum:
    """Test cases for Action enum."""

    def test_action_values(self) -> None:
        """Test that actions have correct values."""
        assert Action.UP.value == 0
        assert Action.DOWN.value == 1
        assert Action.LEFT.value == 2
        assert Action.RIGHT.value == 3
        assert Action.STAY.value == 4

    def test_action_from_value(self) -> None:
        """Test creating actions from values."""
        assert Action(0) == Action.UP
        assert Action(1) == Action.DOWN
        assert Action(2) == Action.LEFT
        assert Action(3) == Action.RIGHT
        assert Action(4) == Action.STAY
