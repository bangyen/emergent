"""Population management for cultural transmission experiments.

This module implements a population manager that maintains N pairs of Speaker/Listener
agents with lifespans to study cultural drift and transmission in emergent language.
"""

import os
import csv
import random
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..core.agents import Speaker, Listener, SpeakerSeq, ListenerSeq
from ..core.config import CommunicationConfig
from ..data.data import ReferentialGameDataset
from ..training.train import MovingAverage
from ..utils.utils import get_logger, get_device, set_seed

logger = get_logger(__name__)


@dataclass
class PopulationConfig:
    """Configuration for population-based cultural transmission experiments.

    This dataclass encapsulates parameters for managing populations of agent pairs
    with lifespans, replacement strategies, and cultural transmission dynamics.

    Attributes:
        n_pairs: Number of Speaker/Listener pairs in the population.
        lifespan: Maximum age before agent replacement.
        replacement_noise: Standard deviation of Gaussian noise for new agents.
        crossplay_prob: Probability of cross-pair interactions.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizers.
        hidden_size: Hidden dimension for neural networks.
        vocabulary_size: Size of communication vocabulary.
        message_length: Length of messages in tokens.
        use_sequence_models: Whether to use sequence-aware models.
        entropy_weight: Weight for entropy bonus regularization.
        seed: Random seed for reproducibility.
    """

    n_pairs: int = 5
    lifespan: int = 1000
    replacement_noise: float = 0.1
    crossplay_prob: float = 0.1
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_size: int = 64
    vocabulary_size: int = 10
    message_length: int = 1
    use_sequence_models: bool = False
    entropy_weight: float = 0.01
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.n_pairs <= 0:
            raise ValueError("n_pairs must be positive")
        if self.lifespan <= 0:
            raise ValueError("lifespan must be positive")
        if self.replacement_noise < 0:
            raise ValueError("replacement_noise must be non-negative")
        if not 0 <= self.crossplay_prob <= 1:
            raise ValueError("crossplay_prob must be between 0 and 1")


class AgentPair:
    """A pair of Speaker and Listener agents with age tracking.

    This class represents a single agent pair in the population, tracking
    their age, lifespan, and performance metrics for cultural transmission studies.

    Attributes:
        speaker: The Speaker agent.
        listener: The Listener agent.
        age: Current age of the agent pair.
        lifespan: Maximum age before replacement.
        speaker_optimizer: Optimizer for the Speaker.
        listener_optimizer: Optimizer for the Listener.
        speaker_baseline: Moving average baseline for Speaker rewards.
        pair_id: Unique identifier for this pair.
        accuracy_history: History of accuracy scores.
        vocab_usage: Vocabulary usage statistics.
    """

    def __init__(
        self,
        speaker: Union[Speaker, SpeakerSeq],
        listener: Union[Listener, ListenerSeq],
        lifespan: int,
        learning_rate: float,
        pair_id: int,
    ):
        """Initialize an agent pair with optimizers and tracking.

        Args:
            speaker: The Speaker agent.
            listener: The Listener agent.
            lifespan: Maximum age before replacement.
            learning_rate: Learning rate for optimizers.
            pair_id: Unique identifier for this pair.
        """
        self.speaker = speaker
        self.listener = listener
        self.age = 0
        self.lifespan = lifespan
        self.pair_id = pair_id

        # Create optimizers
        self.speaker_optimizer = torch.optim.Adam(
            speaker.parameters(), lr=learning_rate
        )
        self.listener_optimizer = torch.optim.Adam(
            listener.parameters(), lr=learning_rate
        )

        # Create baseline
        self.speaker_baseline = MovingAverage(window_size=100)

        # Tracking
        self.accuracy_history: List[float] = []
        self.vocab_usage: Dict[int, int] = defaultdict(int)

    def is_expired(self) -> bool:
        """Check if this agent pair has reached its lifespan."""
        return self.age >= self.lifespan

    def age_up(self) -> None:
        """Increment the age of this agent pair."""
        self.age += 1

    def update_metrics(self, accuracy: float, message_tokens: torch.Tensor) -> None:
        """Update performance metrics for this pair.

        Args:
            accuracy: Accuracy score for this training step.
            message_tokens: Generated message tokens for vocabulary tracking.
        """
        self.accuracy_history.append(accuracy)

        # Track vocabulary usage
        for token in message_tokens.flatten():
            self.vocab_usage[token.item()] += 1

    def get_vocab_histogram(self, vocab_size: int) -> List[int]:
        """Get vocabulary usage histogram.

        Args:
            vocab_size: Size of vocabulary.

        Returns:
            List of usage counts for each vocabulary token.
        """
        histogram = [0] * vocab_size
        for token_id, count in self.vocab_usage.items():
            if 0 <= token_id < vocab_size:
                histogram[token_id] = count
        return histogram


class PopulationManager:
    """Manages a population of agent pairs for cultural transmission studies.

    This class maintains N pairs of Speaker/Listener agents, handles agent
    replacement when they reach their lifespan, and coordinates training
    with both self-play and cross-play interactions.

    Attributes:
        config: Population configuration parameters.
        pairs: List of AgentPair instances.
        device: Device for computations.
        step: Current training step.
        replacement_log: Log of agent replacements.
    """

    def __init__(self, config: PopulationConfig):
        """Initialize the population manager.

        Args:
            config: Population configuration parameters.
        """
        self.config = config
        self.device = get_device()

        # Set seed for reproducibility
        if config.seed is not None:
            set_seed(config.seed)

        # Create communication config
        self.comm_config = CommunicationConfig(
            vocabulary_size=config.vocabulary_size,
            message_length=config.message_length,
            hidden_size=config.hidden_size,
            seed=config.seed,
        )

        # Initialize population
        self.pairs: List[AgentPair] = []
        self._initialize_population()

        # Tracking
        self.step = 0
        self.replacement_log: List[Dict] = []

        logger.info(f"Initialized population with {config.n_pairs} agent pairs")
        logger.info(
            f"Lifespan: {config.lifespan}, Crossplay probability: {config.crossplay_prob}"
        )

    def _initialize_population(self) -> None:
        """Initialize the population with random agent pairs."""
        for i in range(self.config.n_pairs):
            # Create agents
            if self.config.use_sequence_models:
                speaker: Union[Speaker, SpeakerSeq] = SpeakerSeq(self.comm_config).to(
                    self.device
                )
                listener: Union[Listener, ListenerSeq] = ListenerSeq(
                    self.comm_config
                ).to(self.device)
            else:
                speaker = Speaker(self.comm_config).to(self.device)
                listener = Listener(self.comm_config).to(self.device)

            # Create agent pair
            pair = AgentPair(
                speaker=speaker,
                listener=listener,
                lifespan=self.config.lifespan,
                learning_rate=self.config.learning_rate,
                pair_id=i,
            )

            self.pairs.append(pair)

    def _replace_agent_pair(
        self, pair_idx: int, parent_pair: Optional[AgentPair] = None
    ) -> None:
        """Replace an agent pair with a new one, optionally inheriting from parent.

        Args:
            pair_idx: Index of the pair to replace.
            parent_pair: Optional parent pair to inherit weights from.
        """
        old_pair = self.pairs[pair_idx]

        # Create new agents
        if self.config.use_sequence_models:
            speaker: Union[Speaker, SpeakerSeq] = SpeakerSeq(self.comm_config).to(
                self.device
            )
            listener: Union[Listener, ListenerSeq] = ListenerSeq(self.comm_config).to(
                self.device
            )
        else:
            speaker = Speaker(self.comm_config).to(self.device)
            listener = Listener(self.comm_config).to(self.device)

        # Inherit weights from parent with noise
        if parent_pair is not None:
            # Copy parent weights
            speaker.load_state_dict(parent_pair.speaker.state_dict())
            listener.load_state_dict(parent_pair.listener.state_dict())

            # Add Gaussian noise to weights
            with torch.no_grad():
                for param in speaker.parameters():
                    noise = torch.randn_like(param) * self.config.replacement_noise
                    param.add_(noise)

                for param in listener.parameters():
                    noise = torch.randn_like(param) * self.config.replacement_noise
                    param.add_(noise)

        # Create new agent pair
        new_pair = AgentPair(
            speaker=speaker,
            listener=listener,
            lifespan=self.config.lifespan,
            learning_rate=self.config.learning_rate,
            pair_id=pair_idx,
        )

        # Log replacement
        replacement_info = {
            "step": self.step,
            "pair_id": pair_idx,
            "old_age": old_pair.age,
            "parent_pair_id": parent_pair.pair_id if parent_pair else None,
            "replacement_noise": self.config.replacement_noise,
        }
        self.replacement_log.append(replacement_info)

        # Replace the pair
        self.pairs[pair_idx] = new_pair

        logger.info(
            f"Replaced pair {pair_idx} at step {self.step} (age {old_pair.age})"
        )

    def _check_and_replace_expired_pairs(self) -> None:
        """Check for expired pairs and replace them."""
        for i, pair in enumerate(self.pairs):
            if pair.is_expired():
                # Select a random parent pair for inheritance
                parent_pair = random.choice(self.pairs) if len(self.pairs) > 1 else None
                self._replace_agent_pair(i, parent_pair)

    def _select_interaction_pairs(self) -> List[Tuple[int, int]]:
        """Select pairs for interactions (self-play and cross-play).

        Returns:
            List of (speaker_pair_idx, listener_pair_idx) tuples.
        """
        interactions = []

        for i in range(self.config.n_pairs):
            # Self-play within each pair
            interactions.append((i, i))

            # Cross-play with probability crossplay_prob
            if random.random() < self.config.crossplay_prob:
                # Select a random different pair
                other_pairs = [j for j in range(self.config.n_pairs) if j != i]
                if other_pairs:
                    other_pair = random.choice(other_pairs)
                    interactions.append((i, other_pair))

        return interactions

    def train_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """Perform one training step for the population.

        Args:
            batch: Training batch containing (scene_tensor, target_indices, candidate_objects).

        Returns:
            Dictionary containing aggregated metrics across all pairs.
        """
        scene_tensor, target_indices, candidate_objects = batch

        # Move tensors to device
        scene_tensor = scene_tensor.to(self.device)
        target_indices = target_indices.to(self.device)
        candidate_objects = candidate_objects.to(self.device)

        # Select interaction pairs
        interactions = self._select_interaction_pairs()

        # Track metrics
        total_accuracy = 0.0
        total_loss = 0.0
        interaction_count = 0

        # Process each interaction
        for speaker_idx, listener_idx in interactions:
            speaker_pair = self.pairs[speaker_idx]
            listener_pair = self.pairs[listener_idx]

            # Extract target objects
            batch_size = scene_tensor.size(0)
            target_objects = scene_tensor[torch.arange(batch_size), target_indices]

            # Speaker generates messages
            speaker_output = speaker_pair.speaker(target_objects)
            if len(speaker_output) == 4:
                speaker_logits, message_tokens, _, _ = speaker_output
            else:
                speaker_logits, message_tokens = speaker_output

            # Listener makes predictions
            listener_probs = listener_pair.listener(message_tokens, candidate_objects)
            listener_predictions = torch.argmax(listener_probs, dim=1)

            # Compute rewards
            rewards = (listener_predictions == target_indices).float()
            accuracy = rewards.mean().item()

            # Update baselines
            speaker_pair.speaker_baseline.update(accuracy)

            # Compute losses
            listener_loss = F.cross_entropy(listener_probs, target_indices)

            # Speaker loss (REINFORCE)
            log_probs = F.log_softmax(speaker_logits, dim=-1)
            sampled_tokens = torch.argmax(speaker_logits, dim=-1)

            log_probs_sampled = []
            for pos in range(self.config.message_length):
                pos_log_probs = log_probs[:, pos, :]
                pos_sampled = sampled_tokens[:, pos]
                pos_log_probs_sampled = pos_log_probs.gather(
                    1, pos_sampled.unsqueeze(1)
                ).squeeze(1)
                log_probs_sampled.append(pos_log_probs_sampled)

            total_log_probs = torch.stack(log_probs_sampled, dim=1).sum(dim=1)
            advantages = rewards - speaker_pair.speaker_baseline.average
            speaker_loss = -(total_log_probs * advantages).mean()

            # Add entropy bonus
            probs = F.softmax(speaker_logits, dim=-1)
            log_probs_entropy = F.log_softmax(speaker_logits, dim=-1)
            entropy = -(probs * log_probs_entropy).sum(dim=-1).mean()
            speaker_loss = speaker_loss - self.config.entropy_weight * entropy

            # Combined loss
            total_loss_step = listener_loss + speaker_loss

            # Backward pass
            speaker_pair.speaker_optimizer.zero_grad()
            listener_pair.listener_optimizer.zero_grad()
            total_loss_step.backward()
            speaker_pair.speaker_optimizer.step()
            listener_pair.listener_optimizer.step()

            # Update metrics
            speaker_pair.update_metrics(accuracy, message_tokens)

            # Accumulate metrics
            total_accuracy += accuracy
            total_loss += total_loss_step.item()
            interaction_count += 1

        # Age up all pairs
        for pair in self.pairs:
            pair.age_up()

        # Check for replacements
        self._check_and_replace_expired_pairs()

        # Increment step
        self.step += 1

        return {
            "total_loss": (
                total_loss / interaction_count if interaction_count > 0 else 0.0
            ),
            "accuracy": (
                total_accuracy / interaction_count if interaction_count > 0 else 0.0
            ),
            "interactions": interaction_count,
            "replacements": len(
                [r for r in self.replacement_log if r["step"] == self.step - 1]
            ),
        }

    def get_population_stats(self) -> Dict:
        """Get current population statistics.

        Returns:
            Dictionary containing population statistics.
        """
        ages = [pair.age for pair in self.pairs]
        accuracies = [
            pair.accuracy_history[-1] if pair.accuracy_history else 0.0
            for pair in self.pairs
        ]

        # Vocabulary usage histograms
        vocab_histograms = []
        for pair in self.pairs:
            histogram = pair.get_vocab_histogram(self.config.vocabulary_size)
            vocab_histograms.append(histogram)

        return {
            "step": self.step,
            "ages": ages,
            "accuracies": accuracies,
            "avg_age": sum(ages) / len(ages),
            "avg_accuracy": sum(accuracies) / len(accuracies),
            "vocab_histograms": vocab_histograms,
            "total_replacements": len(self.replacement_log),
        }

    def save_logs(self, log_dir: str = "outputs/logs") -> None:
        """Save population logs to CSV files.

        Args:
            log_dir: Directory to save log files.
        """
        os.makedirs(log_dir, exist_ok=True)

        # Save population logs
        population_log_file = os.path.join(log_dir, "population_logs.csv")
        with open(population_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "pair_id", "age", "accuracy", "vocab_usage_total"])

            for pair in self.pairs:
                vocab_total = sum(pair.vocab_usage.values())
                accuracy = pair.accuracy_history[-1] if pair.accuracy_history else 0.0
                writer.writerow(
                    [
                        self.step,
                        pair.pair_id,
                        pair.age,
                        accuracy,
                        vocab_total,
                    ]
                )

        # Save replacement logs
        replacement_log_file = os.path.join(log_dir, "replacement_logs.csv")
        with open(replacement_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["step", "pair_id", "old_age", "parent_pair_id", "replacement_noise"]
            )

            for replacement in self.replacement_log:
                writer.writerow(
                    [
                        replacement["step"],
                        replacement["pair_id"],
                        replacement["old_age"],
                        replacement["parent_pair_id"],
                        replacement["replacement_noise"],
                    ]
                )

        logger.info(f"Saved population logs to {population_log_file}")
        logger.info(f"Saved replacement logs to {replacement_log_file}")


def train_population(
    n_steps: int,
    n_pairs: int = 5,
    lifespan: int = 1000,
    crossplay_prob: float = 0.1,
    replacement_noise: float = 0.1,
    k: int = 5,
    v: int = 10,
    message_length: int = 1,
    seed: int = 42,
    log_every: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    use_sequence_models: bool = False,
    entropy_weight: float = 0.01,
) -> None:
    """Train a population of agent pairs for cultural transmission studies.

    This function runs the main training loop for population-based cultural
    transmission experiments, managing agent lifespans and replacements.

    Args:
        n_steps: Number of training steps to perform.
        n_pairs: Number of agent pairs in the population.
        lifespan: Maximum age before agent replacement.
        crossplay_prob: Probability of cross-pair interactions.
        replacement_noise: Standard deviation of Gaussian noise for new agents.
        k: Number of objects per scene.
        v: Vocabulary size.
        message_length: Message length.
        seed: Random seed for reproducibility.
        log_every: Frequency of logging training metrics.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizers.
        hidden_size: Hidden dimension for neural networks.
        use_sequence_models: Whether to use sequence-aware models.
        entropy_weight: Weight for entropy bonus regularization.
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Create output directories
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    # Create population configuration
    config = PopulationConfig(
        n_pairs=n_pairs,
        lifespan=lifespan,
        replacement_noise=replacement_noise,
        crossplay_prob=crossplay_prob,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        vocabulary_size=v,
        message_length=message_length,
        use_sequence_models=use_sequence_models,
        entropy_weight=entropy_weight,
        seed=seed,
    )

    # Create population manager
    population = PopulationManager(config)

    # Create dataset
    dataset = ReferentialGameDataset(n_scenes=n_steps * batch_size, k=k, seed=seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    logger.info(f"Starting population training for {n_steps} steps")
    logger.info(
        f"Population: {n_pairs} pairs, lifespan: {lifespan}, crossplay: {crossplay_prob}"
    )

    # Initialize metrics logging
    metrics_file = "outputs/logs/population_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "total_loss",
                "accuracy",
                "interactions",
                "replacements",
                "avg_age",
                "avg_accuracy",
                "total_replacements",
            ]
        )

    step = 0
    dataloader_iter = iter(dataloader)

    while step < n_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Restart dataloader if we run out of data
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # Training step
        metrics = population.train_step(batch)

        # Get population stats
        stats = population.get_population_stats()

        # Logging
        if step % log_every == 0:
            logger.info(
                f"Step {step}: Loss={metrics['total_loss']:.4f}, "
                f"Acc={metrics['accuracy']:.4f}, Interactions={metrics['interactions']}, "
                f"Replacements={metrics['replacements']}, Avg Age={stats['avg_age']:.1f}"
            )

        # Save metrics
        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    step,
                    metrics["total_loss"],
                    metrics["accuracy"],
                    metrics["interactions"],
                    metrics["replacements"],
                    stats["avg_age"],
                    stats["avg_accuracy"],
                    stats["total_replacements"],
                ]
            )

        step += 1

    # Save final logs
    population.save_logs()

    logger.info(
        f"Population training completed. Final accuracy: {metrics['accuracy']:.4f}"
    )
    logger.info(f"Total replacements: {stats['total_replacements']}")
    logger.info(f"Metrics saved to: {metrics_file}")
