"""Grounded language learning protocol for grid world navigation.

This module implements a grounded communication protocol where a Speaker
observes object attributes and target coordinates, sends a message, and
a Listener must navigate to the target using the message and a policy network.
"""

import random
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..experiments.grid import (
    Grid,
    Action,
    NavigationPolicy,
    create_simple_grid,
)
from ..core.config import CommunicationConfig
from ..utils.utils import get_logger, get_device, set_seed

logger = get_logger(__name__)


@dataclass
class GroundedEpisode:
    """Data structure for a grounded navigation episode."""

    grid: Grid
    target_attributes: torch.Tensor  # Object attributes for the target
    target_coord: Tuple[int, int]  # Target coordinates
    message: torch.Tensor  # Generated message
    actions: List[Action]  # Actions taken by listener
    rewards: List[float]  # Rewards received
    success: bool  # Whether target was reached
    episode_length: int  # Number of steps taken


class GroundedSpeaker(nn.Module):
    """Speaker agent for grounded navigation tasks.

    The Speaker observes object attributes and target coordinates,
    then generates messages to guide the Listener to the target.

    Args:
        config: Communication configuration.
        coord_dim: Dimension for coordinate encoding (default: 2).
    """

    def __init__(self, config: CommunicationConfig, coord_dim: int = 2):
        super().__init__()
        self.config = config
        self.coord_dim = coord_dim

        # Input dimensions: object attributes + coordinates
        self.input_dim = config.object_dim + coord_dim

        # Encoder for combined input
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )

        # Output layers for each message position
        self.output_layers = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.vocabulary_size)
                for _ in range(config.message_length)
            ]
        )

    def forward(
        self,
        object_attributes: torch.Tensor,
        target_coord: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate message for the given object and target coordinates.

        Args:
            object_attributes: Tensor of shape (batch_size, object_dim) with object attributes.
            target_coord: Tensor of shape (batch_size, coord_dim) with target coordinates.
            temperature: Temperature for sampling (default: 1.0).

        Returns:
            Tuple containing:
            - logits: Tensor of shape (batch_size, message_length, vocabulary_size)
            - message_tokens: Tensor of shape (batch_size, message_length)
        """
        # Combine object attributes and coordinates
        combined_input = torch.cat([object_attributes, target_coord], dim=-1)

        # Encode combined input
        hidden = self.encoder(combined_input)

        # Generate logits for each message position
        logits = []
        message_tokens = []

        for i in range(self.config.message_length):
            pos_logits = self.output_layers[i](hidden)
            logits.append(pos_logits)

            # Sample tokens using Gumbel-Softmax
            if self.training:
                gumbel_noise = -torch.log(
                    -torch.log(torch.rand_like(pos_logits) + 1e-20) + 1e-20
                )
                pos_logits_with_noise = pos_logits + gumbel_noise
            else:
                pos_logits_with_noise = pos_logits

            pos_logits_scaled = pos_logits_with_noise / temperature
            pos_probs = F.softmax(pos_logits_scaled, dim=-1)
            pos_tokens = torch.argmax(pos_probs, dim=-1)

            message_tokens.append(pos_tokens)

        # Stack outputs
        logits_tensor = torch.stack(logits, dim=1)
        message_tokens_tensor = torch.stack(message_tokens, dim=1)

        return logits_tensor, message_tokens_tensor


class GroundedListener(nn.Module):
    """Listener agent for grounded navigation tasks.

    The Listener receives messages and uses them to navigate in the grid world
    using a policy network that combines message understanding with spatial reasoning.

    Args:
        config: Communication configuration.
        grid_size: Size of the grid world.
    """

    def __init__(self, config: CommunicationConfig, grid_size: int):
        super().__init__()
        self.config = config
        self.grid_size = grid_size

        # Message encoder
        self.message_encoder = nn.Sequential(
            nn.Linear(
                config.message_length * config.vocabulary_size, config.hidden_size
            ),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )

        # Navigation policy
        self.navigation_policy = NavigationPolicy(
            grid_size=grid_size,
            message_dim=config.hidden_size,
            hidden_size=config.hidden_size,
            num_actions=5,  # Number of actions
        )

    def forward(
        self, message_tokens: torch.Tensor, grid_obs: torch.Tensor
    ) -> torch.Tensor:
        """Process message and grid observation to get action logits.

        Args:
            message_tokens: Tensor of shape (batch_size, message_length) with message tokens.
            grid_obs: Tensor of shape (batch_size, grid_size, grid_size) with grid observations.

        Returns:
            Action logits tensor of shape (batch_size, num_actions).
        """
        # One-hot encode message tokens
        message_onehot = F.one_hot(
            message_tokens, num_classes=self.config.vocabulary_size
        )
        message_onehot = message_onehot.view(message_tokens.size(0), -1).float()

        # Encode message
        message_embedding = self.message_encoder(message_onehot)

        # Get action logits from navigation policy
        action_logits: torch.Tensor = self.navigation_policy(
            grid_obs, message_embedding
        )

        return action_logits

    def get_action(
        self, message_tokens: torch.Tensor, grid_obs: torch.Tensor
    ) -> Action:
        """Get action from the listener.

        Args:
            message_tokens: Message tokens.
            grid_obs: Grid observation.

        Returns:
            Selected action.
        """
        with torch.no_grad():
            action_logits = self.forward(message_tokens, grid_obs)
            action_probs = F.softmax(action_logits, dim=-1)
            # Use sampling instead of argmax for exploration
            action_idx = torch.multinomial(action_probs, 1).item()
            return Action(action_idx)


class GroundedEnvironment:
    """Environment for grounded language learning in grid world.

    This environment manages the interaction between Speaker and Listener
    agents in a grid world navigation task.
    """

    def __init__(
        self,
        config: CommunicationConfig,
        grid_size: int = 5,
        max_steps: int = 15,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.seed = seed

        if seed is not None:
            set_seed(seed)

        # Create agents
        self.speaker = GroundedSpeaker(config).to(get_device())
        self.listener = GroundedListener(config, grid_size).to(get_device())

        # Object attributes (simplified for grid world)
        self.object_attributes = torch.randn(1, config.object_dim)

    def create_episode(self, grid: Optional[Grid] = None) -> GroundedEpisode:
        """Create a new grounded navigation episode.

        Args:
            grid: Optional grid to use. If None, creates a random grid.

        Returns:
            GroundedEpisode containing the episode data.
        """
        if grid is None:
            grid = create_simple_grid(
                size=self.grid_size,
                max_steps=self.max_steps,
                seed=self.seed,
            )

        # Reset grid
        grid.reset()

        # Generate target attributes (simplified)
        target_attributes = self.object_attributes.clone()

        # Speaker generates message
        target_coord_tensor = torch.tensor(
            [grid.target_pos], dtype=torch.float32
        )  # (1, 2)

        with torch.no_grad():
            speaker_logits, message_tokens = self.speaker(
                target_attributes, target_coord_tensor
            )

        # Listener navigates using the message
        actions = []
        rewards = []

        for step in range(self.max_steps):
            # Get current grid observation
            grid_obs = grid.get_observation().unsqueeze(0)  # Add batch dimension

            # Listener chooses action
            action = self.listener.get_action(message_tokens, grid_obs)
            actions.append(action)

            # Execute action
            state, reward, done, info = grid.step(action)
            rewards.append(reward)

            if done:
                break

        return GroundedEpisode(
            grid=grid,
            target_attributes=target_attributes,
            target_coord=grid.target_pos,
            message=message_tokens,
            actions=actions,
            rewards=rewards,
            success=grid.success,
            episode_length=len(actions),
        )

    def compute_speaker_loss(
        self,
        speaker_logits: torch.Tensor,
        rewards: torch.Tensor,
        baseline: float,
        entropy_weight: float = 0.01,
    ) -> torch.Tensor:
        """Compute REINFORCE loss for the Speaker.

        Args:
            speaker_logits: Speaker logits tensor.
            rewards: Rewards tensor.
            baseline: Baseline value for variance reduction.
            entropy_weight: Weight for entropy bonus.

        Returns:
            Speaker loss tensor.
        """
        batch_size, message_length, vocab_size = speaker_logits.shape

        # Compute log probabilities
        log_probs = F.log_softmax(speaker_logits, dim=-1)

        # Get sampled tokens (argmax)
        sampled_tokens = torch.argmax(speaker_logits, dim=-1)

        # Compute log probabilities of sampled actions
        log_probs_sampled = []
        for i in range(message_length):
            pos_log_probs = log_probs[:, i, :]
            pos_sampled = sampled_tokens[:, i]
            pos_log_probs_sampled = pos_log_probs.gather(
                1, pos_sampled.unsqueeze(1)
            ).squeeze(1)
            log_probs_sampled.append(pos_log_probs_sampled)

        # Sum log probabilities across message positions
        total_log_probs = torch.stack(log_probs_sampled, dim=1).sum(dim=1)

        # Compute REINFORCE loss with baseline
        advantages = rewards - baseline
        reinforce_loss = -(total_log_probs * advantages).mean()

        # Add entropy bonus
        probs = F.softmax(speaker_logits, dim=-1)
        log_probs_entropy = F.log_softmax(speaker_logits, dim=-1)
        entropy = -(probs * log_probs_entropy).sum(dim=-1).mean()

        total_loss = reinforce_loss - entropy_weight * entropy

        return total_loss

    def compute_listener_loss(
        self,
        message_tokens: torch.Tensor,
        grid_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute REINFORCE loss for the Listener.

        Args:
            message_tokens: Message tokens tensor.
            grid_obs: Grid observations tensor.
            actions: Actions taken tensor.
            rewards: Rewards tensor.

        Returns:
            Listener loss tensor.
        """
        # Get action logits
        action_logits = self.listener(message_tokens, grid_obs)

        # Compute log probabilities
        log_probs = F.log_softmax(action_logits, dim=-1)

        # Get log probabilities of taken actions
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute REINFORCE loss (negative log probability weighted by rewards)
        loss = -(action_log_probs * rewards).mean()

        return loss


def run_grounded_episode(
    environment: GroundedEnvironment,
    grid: Optional[Grid] = None,
) -> GroundedEpisode:
    """Run a single grounded navigation episode.

    Args:
        environment: GroundedEnvironment instance.
        grid: Optional grid to use.

    Returns:
        GroundedEpisode containing episode results.
    """
    return environment.create_episode(grid)


def evaluate_grounded_performance(
    environment: GroundedEnvironment,
    num_episodes: int = 100,
    grids: Optional[List[Grid]] = None,
) -> Dict[str, float]:
    """Evaluate performance of grounded agents.

    Args:
        environment: GroundedEnvironment instance.
        num_episodes: Number of episodes to evaluate.
        grids: Optional list of grids to use.

    Returns:
        Dictionary containing performance metrics.
    """
    successes = 0
    total_rewards = 0.0
    episode_lengths = []

    for i in range(num_episodes):
        if grids is not None:
            grid = random.choice(grids)
        else:
            grid = None

        episode = run_grounded_episode(environment, grid)

        if episode.success:
            successes += 1

        total_rewards += sum(episode.rewards)
        episode_lengths.append(episode.episode_length)

    success_rate = successes / num_episodes
    avg_reward = total_rewards / num_episodes
    avg_length = sum(episode_lengths) / len(episode_lengths)

    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_episode_length": avg_length,
    }
