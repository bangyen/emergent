"""2D Grid world environment for grounded language learning.

This module implements a simple 2D grid world where agents can navigate
using discrete actions to reach target objects. The grid supports walls
and provides a spatial grounding for language learning tasks.
"""

import random
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.utils import get_logger

logger = get_logger(__name__)


class Action(Enum):
    """Available actions in the grid world."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


class Grid:
    """2D Grid world environment for navigation tasks.

    The grid supports walls, start positions for agents, and target objects.
    Agents can navigate using discrete actions and receive rewards for
    reaching targets within episode limits.

    Args:
        size: Size of the square grid (default: 7).
        walls: List of wall coordinates (x, y) tuples.
        start_pos: Starting position for the listener agent (x, y).
        target_pos: Position of the target object (x, y).
        max_steps: Maximum steps per episode (default: 20).
        seed: Random seed for reproducible generation.
    """

    def __init__(
        self,
        size: int = 7,
        walls: Optional[List[Tuple[int, int]]] = None,
        start_pos: Optional[Tuple[int, int]] = None,
        target_pos: Optional[Tuple[int, int]] = None,
        max_steps: int = 20,
        seed: Optional[int] = None,
    ):
        self.size = size
        self.max_steps = max_steps

        if seed is not None:
            random.seed(seed)

        # Initialize grid state
        self.grid = [[0 for _ in range(size)] for _ in range(size)]

        # Set walls
        self.walls = walls or []
        for x, y in self.walls:
            if self._is_valid_position(x, y):
                self.grid[y][x] = 1  # 1 represents wall

        # Set start position
        if start_pos is None:
            self.start_pos = self._find_empty_position()
        else:
            self.start_pos = start_pos
            if not self._is_valid_position(*self.start_pos) or self._is_wall(
                *self.start_pos
            ):
                raise ValueError(f"Invalid start position: {self.start_pos}")

        # Set target position
        if target_pos is None:
            self.target_pos = self._find_empty_position(exclude=[self.start_pos])
        else:
            self.target_pos = target_pos
            if not self._is_valid_position(*self.target_pos) or self._is_wall(
                *self.target_pos
            ):
                raise ValueError(f"Invalid target position: {self.target_pos}")

        # Initialize agent state
        self.reset()

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= x < self.size and 0 <= y < self.size

    def _is_wall(self, x: int, y: int) -> bool:
        """Check if position contains a wall."""
        return self.grid[y][x] == 1

    def _is_empty(self, x: int, y: int) -> bool:
        """Check if position is empty (no wall, agent, or target)."""
        return self.grid[y][x] == 0

    def _find_empty_position(
        self, exclude: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[int, int]:
        """Find a random empty position on the grid."""
        exclude = exclude or []
        empty_positions = []

        for y in range(self.size):
            for x in range(self.size):
                if self._is_empty(x, y) and (x, y) not in exclude:
                    empty_positions.append((x, y))

        if not empty_positions:
            raise ValueError("No empty positions available on the grid")

        return random.choice(empty_positions)

    def reset(self) -> Dict[str, Any]:
        """Reset the grid to initial state.

        Returns:
            Dictionary containing initial state information.
        """
        self.agent_pos = self.start_pos
        self.step_count = 0
        self.done = False
        self.success = False

        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the grid.

        Returns:
            Dictionary containing current state information.
        """
        return {
            "agent_pos": self.agent_pos,
            "target_pos": self.target_pos,
            "walls": self.walls,
            "step_count": self.step_count,
            "done": self.done,
            "success": self.success,
            "grid_size": self.size,
        }

    def step(
        self, action: Action
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute an action and return new state, reward, done flag, and info.

        Args:
            action: Action to execute.

        Returns:
            Tuple containing:
            - state: New state dictionary
            - reward: Reward for this step (1.0 if target reached, 0.0 otherwise)
            - done: Whether episode is finished
            - info: Additional information
        """
        if self.done:
            logger.warning("Episode already finished, ignoring action")
            return self.get_state(), 0.0, True, {"message": "Episode already finished"}

        # Calculate new position
        x, y = self.agent_pos

        if action == Action.UP:
            new_pos = (x, y - 1)
        elif action == Action.DOWN:
            new_pos = (x, y + 1)
        elif action == Action.LEFT:
            new_pos = (x - 1, y)
        elif action == Action.RIGHT:
            new_pos = (x + 1, y)
        elif action == Action.STAY:
            new_pos = (x, y)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check if new position is valid and not a wall
        if self._is_valid_position(*new_pos) and not self._is_wall(*new_pos):
            self.agent_pos = new_pos

        # Update step count
        self.step_count += 1

        # Check if target reached
        reward = 0.0
        if self.agent_pos == self.target_pos:
            reward = 1.0
            self.success = True
            self.done = True
        else:
            # Add distance-based reward to guide learning
            distance = abs(self.agent_pos[0] - self.target_pos[0]) + abs(
                self.agent_pos[1] - self.target_pos[1]
            )
            max_distance = self.size * 2  # Maximum possible distance
            reward = 0.1 * (
                1.0 - distance / max_distance
            )  # Small positive reward for getting closer

        if self.step_count >= self.max_steps:
            self.done = True

        # Prepare info
        info = {
            "step_count": self.step_count,
            "success": self.success,
            "target_reached": self.agent_pos == self.target_pos,
        }

        return self.get_state(), reward, self.done, info

    def render(self) -> str:
        """Render the grid as a string representation.

        Returns:
            String representation of the current grid state.
        """
        lines = []
        for y in range(self.size):
            line = ""
            for x in range(self.size):
                if (x, y) == self.agent_pos:
                    line += "A"  # Agent
                elif (x, y) == self.target_pos:
                    line += "T"  # Target
                elif self._is_wall(x, y):
                    line += "#"  # Wall
                else:
                    line += "."  # Empty
            lines.append(line)

        return "\n".join(lines)

    def get_observation(self) -> torch.Tensor:
        """Get observation tensor for neural network input.

        Returns:
            Tensor of shape (grid_size, grid_size) representing the grid state.
        """
        obs = torch.zeros(self.size, self.size)

        # Mark walls
        for x, y in self.walls:
            obs[y, x] = 1.0

        # Mark agent position
        obs[self.agent_pos[1], self.agent_pos[0]] = 2.0

        # Mark target position
        obs[self.target_pos[1], self.target_pos[0]] = 3.0

        return obs


class NavigationPolicy(nn.Module):
    """Simple policy network for navigation in grid world.

    This network takes grid observations and message embeddings as input
    and outputs action probabilities for navigation.

    Args:
        grid_size: Size of the grid (assumed square).
        message_dim: Dimension of message embeddings.
        hidden_size: Hidden layer size.
        num_actions: Number of possible actions (default: 5).
    """

    def __init__(
        self,
        grid_size: int,
        message_dim: int,
        hidden_size: int = 64,
        num_actions: int = 5,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.message_dim = message_dim
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        # Grid encoder (convolutional layers)
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Fixed size output
            nn.Flatten(),
        )

        # Message encoder
        self.message_encoder = nn.Sequential(
            nn.Linear(message_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Combined feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(32 * 4 * 4 + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Action head
        self.action_head = nn.Linear(hidden_size, num_actions)

    def forward(
        self, grid_obs: torch.Tensor, message_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the policy network.

        Args:
            grid_obs: Grid observation tensor of shape (batch_size, grid_size, grid_size).
            message_embedding: Message embedding tensor of shape (batch_size, message_dim).

        Returns:
            Action logits tensor of shape (batch_size, num_actions).
        """
        # Encode grid observation
        grid_input = grid_obs.unsqueeze(1)  # Add channel dimension
        grid_features = self.grid_encoder(grid_input)  # (batch_size, 32*4*4)

        # Encode message
        message_features = self.message_encoder(
            message_embedding
        )  # (batch_size, hidden_size)

        # Combine features
        combined_features = torch.cat([grid_features, message_features], dim=-1)
        processed_features = self.feature_processor(combined_features)

        # Output action logits
        action_logits: torch.Tensor = self.action_head(processed_features)

        return action_logits

    def get_action(
        self, grid_obs: torch.Tensor, message_embedding: torch.Tensor
    ) -> Action:
        """Get action from policy network.

        Args:
            grid_obs: Grid observation tensor.
            message_embedding: Message embedding tensor.

        Returns:
            Selected action.
        """
        with torch.no_grad():
            logits = self.forward(grid_obs, message_embedding)
            action_probs = F.softmax(logits, dim=-1)
            action_idx = torch.argmax(action_probs, dim=-1).item()
            return Action(action_idx)


def create_simple_grid(
    size: int = 5,
    num_walls: int = 0,
    max_steps: int = 15,
    seed: Optional[int] = None,
) -> Grid:
    """Create a simple grid with random walls and positions.

    Args:
        size: Size of the grid.
        num_walls: Number of walls to place randomly.
        max_steps: Maximum steps per episode.
        seed: Random seed for reproducibility.

    Returns:
        Configured Grid instance.
    """
    if seed is not None:
        random.seed(seed)

    # Generate random walls
    walls = []
    for _ in range(num_walls):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        walls.append((x, y))

    return Grid(
        size=size,
        walls=walls,
        max_steps=max_steps,
        seed=seed,
    )


def create_curriculum_grids(
    base_size: int = 3,
    max_size: int = 7,
    max_walls: int = 3,
    max_steps: int = 20,
    seed: Optional[int] = None,
) -> List[Grid]:
    """Create a curriculum of grids with increasing difficulty.

    Args:
        base_size: Starting grid size.
        max_size: Maximum grid size.
        max_walls: Maximum number of walls.
        max_steps: Maximum steps per episode.
        seed: Random seed for reproducibility.

    Returns:
        List of Grid instances with increasing difficulty.
    """
    if seed is not None:
        random.seed(seed)

    grids = []

    for size in range(base_size, max_size + 1):
        for num_walls in range(min(max_walls + 1, size * size // 4)):
            grid = create_simple_grid(
                size=size,
                num_walls=num_walls,
                max_steps=max_steps,
                seed=seed,
            )
            grids.append(grid)

    return grids
