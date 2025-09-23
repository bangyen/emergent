"""Training module for grounded language learning in grid world.

This module implements the training loop for grounded navigation tasks
where Speaker and Listener agents learn to communicate about spatial
targets through curriculum learning.
"""

import os
import csv
import random
from typing import Dict, List, Any
from collections import deque

import torch

from .grounding import (
    GroundedEnvironment,
    run_grounded_episode,
    evaluate_grounded_performance,
)
from ..experiments.grid import Grid, create_curriculum_grids
from ..core.config import CommunicationConfig
from ..utils.utils import get_logger, get_device, set_seed

logger = get_logger(__name__)


class MovingAverage:
    """Moving average baseline for REINFORCE training."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rewards: deque = deque(maxlen=window_size)
        self._average = 0.0

    def update(self, reward: float) -> None:
        self.rewards.append(reward)
        self._average = sum(self.rewards) / len(self.rewards)

    @property
    def average(self) -> float:
        return self._average


class CurriculumScheduler:
    """Curriculum scheduler for progressive difficulty increase.

    This scheduler manages the progression through different grid configurations
    based on performance metrics.
    """

    def __init__(
        self,
        base_size: int = 3,
        max_size: int = 7,
        max_walls: int = 3,
        success_threshold: float = 0.8,
        min_episodes_per_level: int = 50,
    ):
        self.base_size = base_size
        self.max_size = max_size
        self.max_walls = max_walls
        self.success_threshold = success_threshold
        self.min_episodes_per_level = min_episodes_per_level

        # Generate curriculum grids
        self.curriculum_grids = create_curriculum_grids(
            base_size=base_size,
            max_size=max_size,
            max_walls=max_walls,
        )

        self.current_level = 0
        self.episodes_at_level = 0
        self.level_successes = 0

    def get_current_grids(self) -> List[Grid]:
        """Get grids for current curriculum level."""
        if self.current_level >= len(self.curriculum_grids):
            return self.curriculum_grids[-1:]  # Use hardest level

        return self.curriculum_grids[: self.current_level + 1]

    def update(self, success: bool) -> bool:
        """Update curriculum based on episode result.

        Args:
            success: Whether the episode was successful.

        Returns:
            True if curriculum level changed.
        """
        self.episodes_at_level += 1
        if success:
            self.level_successes += 1

        # Check if we should advance to next level
        if (
            self.episodes_at_level >= self.min_episodes_per_level
            and self.level_successes / self.episodes_at_level >= self.success_threshold
            and self.current_level < len(self.curriculum_grids) - 1
        ):
            logger.info(f"Advancing to curriculum level {self.current_level + 1}")
            self.current_level += 1
            self.episodes_at_level = 0
            self.level_successes = 0
            return True

        return False

    def get_level_info(self) -> Dict[str, Any]:
        """Get information about current curriculum level."""
        if self.current_level >= len(self.curriculum_grids):
            return {"level": self.current_level, "description": "Maximum level reached"}

        current_grids = self.get_current_grids()
        if not current_grids:
            return {"level": 0, "description": "No grids available"}

        # Get characteristics of current level
        sizes = [g.size for g in current_grids]
        wall_counts = [len(g.walls) for g in current_grids]

        return {
            "level": self.current_level,
            "num_grids": len(current_grids),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "min_walls": min(wall_counts),
            "max_walls": max(wall_counts),
            "success_rate": self.level_successes / max(1, self.episodes_at_level),
        }


def train_grounded_step(
    environment: GroundedEnvironment,
    speaker_optimizer: torch.optim.Optimizer,
    listener_optimizer: torch.optim.Optimizer,
    speaker_baseline: MovingAverage,
    grid: Grid,
    entropy_weight: float = 0.01,
) -> Dict[str, float]:
    """Perform one training step for grounded agents.

    Args:
        environment: GroundedEnvironment instance.
        speaker_optimizer: Optimizer for Speaker.
        listener_optimizer: Optimizer for Listener.
        speaker_baseline: Moving average baseline for Speaker rewards.
        grid: Grid to use for this episode.
        entropy_weight: Weight for entropy bonus.

    Returns:
        Dictionary containing loss values and metrics.
    """
    device = get_device()

    # Run episode
    episode = run_grounded_episode(environment, grid)

    # Prepare data for training
    target_attributes = episode.target_attributes.to(device)
    target_coord = torch.tensor([episode.target_coord], dtype=torch.float32).to(device)

    # Speaker forward pass
    speaker_logits, message_tokens = environment.speaker(
        target_attributes, target_coord
    )

    # Prepare listener training data
    actions_tensor = torch.tensor(
        [action.value for action in episode.actions], dtype=torch.long
    ).to(device)

    rewards_tensor = torch.tensor(episode.rewards, dtype=torch.float32).to(device)

    # Get grid observations for each step
    grid_obs_list = []
    for i, action in enumerate(episode.actions):
        # Recreate grid state up to step i
        temp_grid = Grid(
            size=grid.size,
            walls=grid.walls,
            start_pos=grid.start_pos,
            target_pos=grid.target_pos,
            max_steps=grid.max_steps,
        )
        temp_grid.reset()

        for j in range(i):
            temp_grid.step(episode.actions[j])

        grid_obs = temp_grid.get_observation().unsqueeze(0)
        grid_obs_list.append(grid_obs)

    if grid_obs_list:
        grid_obs_tensor = torch.stack(grid_obs_list).squeeze(1).to(device)

        # Listener forward pass
        # Repeat message tokens for each step
        message_tokens_repeated = message_tokens.repeat(len(episode.actions), 1)
        environment.listener(message_tokens_repeated, grid_obs_tensor)

        # Compute losses
        speaker_loss = environment.compute_speaker_loss(
            speaker_logits,
            rewards_tensor,
            speaker_baseline.average,
            entropy_weight,
        )

        listener_loss = environment.compute_listener_loss(
            message_tokens_repeated,
            grid_obs_tensor,
            actions_tensor,
            rewards_tensor,
        )

        # Update baselines
        avg_reward = rewards_tensor.mean().item()
        speaker_baseline.update(avg_reward)

        # Backward pass
        speaker_optimizer.zero_grad()
        listener_optimizer.zero_grad()

        total_loss = speaker_loss + listener_loss
        total_loss.backward()

        speaker_optimizer.step()
        listener_optimizer.step()

        # Compute metrics
        success_rate = 1.0 if episode.success else 0.0
        avg_reward = rewards_tensor.mean().item()

        return {
            "total_loss": total_loss.item(),
            "speaker_loss": speaker_loss.item(),
            "listener_loss": listener_loss.item(),
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "episode_length": episode.episode_length,
            "baseline": speaker_baseline.average,
        }
    else:
        # No actions taken (shouldn't happen)
        return {
            "total_loss": 0.0,
            "speaker_loss": 0.0,
            "listener_loss": 0.0,
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "episode_length": 0,
            "baseline": speaker_baseline.average,
        }


def train_grounded(
    episodes: int = 500,
    grid_size: int = 5,
    max_steps: int = 15,
    vocabulary_size: int = 10,
    message_length: int = 3,
    hidden_size: int = 64,
    learning_rate: float = 1e-3,
    entropy_weight: float = 0.01,
    log_every: int = 50,
    eval_every: int = 100,
    seed: int = 3,
    use_curriculum: bool = True,
) -> None:
    """Train grounded Speaker and Listener agents.

    Args:
        episodes: Number of training episodes.
        grid_size: Size of the grid world.
        max_steps: Maximum steps per episode.
        vocabulary_size: Size of the vocabulary.
        message_length: Length of messages.
        hidden_size: Hidden layer size.
        learning_rate: Learning rate for optimizers.
        entropy_weight: Weight for entropy bonus.
        log_every: Frequency of logging.
        eval_every: Frequency of evaluation.
        seed: Random seed.
        use_curriculum: Whether to use curriculum learning.
    """
    # Set seed
    set_seed(seed)

    # Create output directories
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    device = get_device()
    logger.info(f"Training grounded agents on device: {device}")

    # Create configuration
    config = CommunicationConfig(
        vocabulary_size=vocabulary_size,
        message_length=message_length,
        hidden_size=hidden_size,
        seed=seed,
    )

    # Create environment
    environment = GroundedEnvironment(
        config=config,
        grid_size=grid_size,
        max_steps=max_steps,
        seed=seed,
    )

    # Create optimizers
    speaker_optimizer = torch.optim.Adam(
        environment.speaker.parameters(), lr=learning_rate
    )
    listener_optimizer = torch.optim.Adam(
        environment.listener.parameters(), lr=learning_rate
    )

    # Create baseline
    speaker_baseline = MovingAverage(window_size=100)

    # Create curriculum scheduler
    if use_curriculum:
        curriculum = CurriculumScheduler(
            base_size=3,
            max_size=grid_size,
            max_walls=min(3, grid_size // 2),
        )
    else:
        curriculum = None

    # Initialize metrics logging
    metrics_file = "outputs/logs/grid_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "total_loss",
                "speaker_loss",
                "listener_loss",
                "success_rate",
                "avg_reward",
                "episode_length",
                "baseline",
                "curriculum_level",
            ]
        )

    logger.info(f"Starting grounded training for {episodes} episodes")
    logger.info(
        f"Configuration: grid_size={grid_size}, vocab={vocabulary_size}, "
        f"message_length={message_length}, seed={seed}"
    )

    # Training loop
    for episode in range(episodes):
        # Select grid for this episode
        if curriculum is not None:
            available_grids = curriculum.get_current_grids()
            grid = random.choice(available_grids)
        else:
            from .grid import create_simple_grid

            grid = create_simple_grid(
                size=grid_size,
                max_steps=max_steps,
                seed=seed + episode,
            )

        # Training step
        metrics = train_grounded_step(
            environment,
            speaker_optimizer,
            listener_optimizer,
            speaker_baseline,
            grid,
            entropy_weight,
        )

        # Update curriculum
        if curriculum is not None:
            curriculum.update(metrics["success_rate"] > 0.5)
            metrics["curriculum_level"] = curriculum.current_level
        else:
            metrics["curriculum_level"] = 0

        # Logging
        if episode % log_every == 0:
            level_info = ""
            if curriculum is not None:
                level_info = f", Level={curriculum.current_level}"

            logger.info(
                f"Episode {episode}: Loss={metrics['total_loss']:.4f}, "
                f"Success={metrics['success_rate']:.2f}, "
                f"Reward={metrics['avg_reward']:.3f}{level_info}"
            )

        # Save metrics
        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    episode,  # Use 'episode' for grid training (different from 'step' in regular training)
                    metrics["total_loss"],
                    metrics["speaker_loss"],
                    metrics["listener_loss"],
                    metrics["success_rate"],
                    metrics["avg_reward"],
                    metrics["episode_length"],
                    metrics["baseline"],
                    metrics["curriculum_level"],
                ]
            )

        # Evaluation
        if episode % eval_every == 0 and episode > 0:
            # Evaluate on current curriculum level
            if curriculum is not None:
                eval_grids = curriculum.get_current_grids()
            else:
                eval_grids = None

            eval_metrics = evaluate_grounded_performance(
                environment,
                num_episodes=20,
                grids=eval_grids,
            )

            logger.info(
                f"Evaluation at episode {episode}: "
                f"Success rate={eval_metrics['success_rate']:.3f}, "
                f"Avg reward={eval_metrics['avg_reward']:.3f}, "
                f"Avg length={eval_metrics['avg_episode_length']:.1f}"
            )

            # Save checkpoint
            checkpoint_path = (
                f"outputs/checkpoints/grounded_checkpoint_episode_{episode}.pt"
            )
            torch.save(
                {
                    "episode": episode,
                    "speaker_state_dict": environment.speaker.state_dict(),
                    "listener_state_dict": environment.listener.state_dict(),
                    "speaker_optimizer_state_dict": speaker_optimizer.state_dict(),
                    "listener_optimizer_state_dict": listener_optimizer.state_dict(),
                    "config": config,
                    "metrics": metrics,
                    "eval_metrics": eval_metrics,
                },
                checkpoint_path,
            )

    # Save final checkpoint
    final_checkpoint_path = "outputs/checkpoints/grounded_checkpoint_final.pt"
    torch.save(
        {
            "episode": episodes,
            "speaker_state_dict": environment.speaker.state_dict(),
            "listener_state_dict": environment.listener.state_dict(),
            "speaker_optimizer_state_dict": speaker_optimizer.state_dict(),
            "listener_optimizer_state_dict": listener_optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        final_checkpoint_path,
    )

    logger.info(f"Grounded training completed after {episodes} episodes")
    logger.info(f"Metrics saved to: {metrics_file}")
    logger.info(f"Final checkpoint saved to: {final_checkpoint_path}")

    # Final evaluation
    if curriculum is not None:
        eval_grids = curriculum.get_current_grids()
    else:
        eval_grids = None

    final_eval = evaluate_grounded_performance(
        environment,
        num_episodes=100,
        grids=eval_grids,
    )

    logger.info(
        f"Final evaluation: Success rate={final_eval['success_rate']:.3f}, "
        f"Avg reward={final_eval['avg_reward']:.3f}"
    )
