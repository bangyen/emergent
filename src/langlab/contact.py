"""Contact experiments for measuring mutual intelligibility between populations.

This module implements two-stage training experiments where two populations
are trained separately and then brought into contact to measure their mutual
intelligibility through cross-population interactions.
"""

import os
import json
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from .agents import Speaker, Listener, SpeakerSeq, ListenerSeq
from .data import ReferentialGameDataset
from .population import PopulationManager, PopulationConfig
from .utils import get_logger, get_device, set_seed

logger = get_logger(__name__)


@dataclass
class ContactConfig:
    """Configuration for contact experiments.

    This dataclass encapsulates parameters for two-stage contact experiments
    where populations are trained separately and then brought into contact.

    Attributes:
        n_pairs: Number of agent pairs per population.
        steps_a: Number of training steps for Stage A (separate training).
        steps_b: Number of training steps for Stage B (contact phase).
        contact_steps: Number of contact phase steps.
        p_contact: Probability of cross-population interactions.
        k: Number of objects per scene.
        v: Vocabulary size.
        message_length: Message length.
        seed_a: Random seed for Population A.
        seed_b: Random seed for Population B.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizers.
        hidden_size: Hidden dimension for neural networks.
        use_sequence_models: Whether to use sequence-aware models.
        entropy_weight: Weight for entropy bonus regularization.
        heldout_pairs_a: Held-out attribute pairs for Population A.
        heldout_pairs_b: Held-out attribute pairs for Population B.
    """

    n_pairs: int = 4
    steps_a: int = 4000
    steps_b: int = 4000
    contact_steps: int = 2000
    p_contact: float = 0.3
    k: int = 5
    v: int = 10
    message_length: int = 1
    seed_a: int = 42
    seed_b: int = 123
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_size: int = 64
    use_sequence_models: bool = False
    entropy_weight: float = 0.01
    heldout_pairs_a: Optional[List[Tuple[str, str]]] = None
    heldout_pairs_b: Optional[List[Tuple[str, str]]] = None


class ContactExperiment:
    """Manages contact experiments between two populations.

    This class coordinates the two-stage training process and measures
    mutual intelligibility between populations through cross-interactions.

    Attributes:
        config: Contact experiment configuration.
        device: Device for computations.
        population_a: First population manager.
        population_b: Second population manager.
        step: Current training step.
        intelligibility_matrix: Mutual intelligibility matrix.
        jsd_score: Jensen-Shannon divergence score.
    """

    def __init__(self, config: ContactConfig):
        """Initialize the contact experiment.

        Args:
            config: Contact experiment configuration.
        """
        self.config = config
        self.device = get_device()

        # Create output directories
        os.makedirs("outputs/logs", exist_ok=True)
        os.makedirs("outputs/figures", exist_ok=True)

        # Initialize populations (will be set in train_stage_a)
        self.population_a: PopulationManager = None  # type: ignore
        self.population_b: PopulationManager = None  # type: ignore

        # Tracking
        self.step = 0
        self.intelligibility_matrix: Optional[np.ndarray] = None
        self.jsd_score: Optional[float] = None

        logger.info(
            f"Initialized contact experiment with {config.n_pairs} pairs per population"
        )
        logger.info(f"Stage A steps: {config.steps_a}, Stage B steps: {config.steps_b}")
        logger.info(f"Contact probability: {config.p_contact}")

    def _create_population(
        self,
        seed: int,
        heldout_pairs: Optional[List[Tuple[str, str]]],
        population_id: str,
    ) -> PopulationManager:
        """Create a population manager with specified configuration.

        Args:
            seed: Random seed for the population.
            heldout_pairs: Held-out attribute pairs for compositional splits.
            population_id: Identifier for the population.

        Returns:
            Configured PopulationManager instance.
        """
        # Set seed for this population
        set_seed(seed)

        # Create population configuration
        pop_config = PopulationConfig(
            n_pairs=self.config.n_pairs,
            lifespan=self.config.steps_a
            + self.config.steps_b
            + self.config.contact_steps,
            replacement_noise=0.1,
            crossplay_prob=0.0,  # No crossplay during separate training
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            vocabulary_size=self.config.v,
            message_length=self.config.message_length,
            use_sequence_models=self.config.use_sequence_models,
            entropy_weight=self.config.entropy_weight,
            seed=seed,
        )

        # Create population manager
        population = PopulationManager(pop_config)

        logger.info(
            f"Created {population_id} with {self.config.n_pairs} pairs (seed={seed})"
        )
        if heldout_pairs:
            logger.info(f"{population_id} heldout pairs: {heldout_pairs}")

        return population

    def train_stage_a(self) -> None:
        """Train both populations separately in Stage A."""
        logger.info("Starting Stage A: Separate population training")

        # Create Population A
        self.population_a = self._create_population(
            self.config.seed_a, self.config.heldout_pairs_a, "Population A"
        )

        # Create Population B
        self.population_b = self._create_population(
            self.config.seed_b, self.config.heldout_pairs_b, "Population B"
        )

        # Train Population A
        logger.info(f"Training Population A for {self.config.steps_a} steps")
        self._train_population(self.population_a, self.config.steps_a, "A")

        # Train Population B
        logger.info(f"Training Population B for {self.config.steps_b} steps")
        self._train_population(self.population_b, self.config.steps_b, "B")

        logger.info("Stage A completed: Both populations trained separately")

    def _train_population(
        self, population: PopulationManager, n_steps: int, population_id: str
    ) -> None:
        """Train a single population for specified number of steps.

        Args:
            population: Population manager to train.
            n_steps: Number of training steps.
            population_id: Identifier for logging.
        """
        # Create dataset
        dataset = ReferentialGameDataset(
            n_scenes=n_steps * self.config.batch_size,
            k=self.config.k,
            seed=population.config.seed,
        )
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # Training loop
        step = 0
        dataloader_iter = iter(dataloader)

        while step < n_steps:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            # Training step
            metrics = population.train_step(batch)

            # Logging
            if step % 100 == 0:
                logger.info(
                    f"Population {population_id} Step {step}: "
                    f"Loss={metrics['total_loss']:.4f}, Acc={metrics['accuracy']:.4f}"
                )

            step += 1

    def train_stage_b(self) -> None:
        """Train populations in contact phase (Stage B)."""
        logger.info("Starting Stage B: Contact phase training")

        # Enable cross-population interactions
        self.population_a.config.crossplay_prob = self.config.p_contact
        self.population_b.config.crossplay_prob = self.config.p_contact

        # Train both populations with contact
        logger.info(f"Training with contact for {self.config.contact_steps} steps")

        # Create combined dataset
        dataset = ReferentialGameDataset(
            n_scenes=self.config.contact_steps * self.config.batch_size,
            k=self.config.k,
            seed=42,
        )
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # Training loop
        step = 0
        dataloader_iter = iter(dataloader)

        while step < self.config.contact_steps:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            # Train both populations
            metrics_a = self.population_a.train_step(batch)
            metrics_b = self.population_b.train_step(batch)

            # Logging
            if step % 100 == 0:
                logger.info(
                    f"Contact Step {step}: "
                    f"Pop A Acc={metrics_a['accuracy']:.4f}, "
                    f"Pop B Acc={metrics_b['accuracy']:.4f}"
                )

            step += 1

        logger.info("Stage B completed: Contact phase training finished")

    def measure_intelligibility(self) -> None:
        """Measure mutual intelligibility between populations."""
        logger.info("Measuring mutual intelligibility")

        # Create evaluation dataset
        eval_dataset = ReferentialGameDataset(
            n_scenes=1000, k=self.config.k, seed=42  # Fixed size for evaluation
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        # Initialize intelligibility matrix
        n_pairs_a = len(self.population_a.pairs)
        n_pairs_b = len(self.population_b.pairs)
        self.intelligibility_matrix = np.zeros((n_pairs_a, n_pairs_b))

        # Measure intelligibility for each pair combination
        for i, speaker_pair in enumerate(self.population_a.pairs):
            for j, listener_pair in enumerate(self.population_b.pairs):
                accuracy = self._measure_pair_intelligibility(
                    speaker_pair.speaker, listener_pair.listener, eval_dataloader
                )
                self.intelligibility_matrix[i, j] = accuracy

        # Compute Jensen-Shannon divergence
        self.jsd_score = self._compute_jsd()

        if self.intelligibility_matrix is not None:
            logger.info(
                f"Intelligibility matrix shape: {self.intelligibility_matrix.shape}"
            )
            logger.info(
                f"Average intelligibility: {np.mean(self.intelligibility_matrix):.4f}"
            )
        logger.info(f"Jensen-Shannon divergence: {self.jsd_score:.4f}")

    def _measure_pair_intelligibility(
        self,
        speaker: Union[Speaker, SpeakerSeq],
        listener: Union[Listener, ListenerSeq],
        dataloader: DataLoader,
    ) -> float:
        """Measure intelligibility between a specific speaker-listener pair.

        Args:
            speaker: Speaker agent.
            listener: Listener agent.
            dataloader: Data loader for evaluation.

        Returns:
            Accuracy score for this speaker-listener pair.
        """
        speaker.eval()
        listener.eval()

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                scene_tensor, target_indices, candidate_objects = batch

                # Move to device
                scene_tensor = scene_tensor.to(self.device)
                target_indices = target_indices.to(self.device)
                candidate_objects = candidate_objects.to(self.device)

                # Extract target objects
                batch_size = scene_tensor.size(0)
                target_objects = scene_tensor[torch.arange(batch_size), target_indices]

                # Speaker generates messages
                _, message_tokens = speaker(target_objects)

                # Listener makes predictions
                listener_probs = listener(message_tokens, candidate_objects)
                listener_predictions = torch.argmax(listener_probs, dim=1)

                # Count correct predictions
                correct = (listener_predictions == target_indices).sum().item()
                total_correct += correct
                total_samples += batch_size

        return total_correct / total_samples if total_samples > 0 else 0.0

    def _compute_jsd(self) -> float:
        """Compute Jensen-Shannon divergence between message distributions.

        Returns:
            JSD score between populations A and B.
        """
        # Collect message distributions from both populations
        messages_a = self._collect_message_distribution(self.population_a)
        messages_b = self._collect_message_distribution(self.population_b)

        # Convert to probability distributions
        vocab_size = self.config.v
        dist_a = self._messages_to_distribution(messages_a, vocab_size)
        dist_b = self._messages_to_distribution(messages_b, vocab_size)

        # Compute JSD
        jsd = self._jensen_shannon_divergence(dist_a, dist_b)
        return jsd

    def _collect_message_distribution(
        self, population: PopulationManager
    ) -> List[torch.Tensor]:
        """Collect message samples from a population.

        Args:
            population: Population manager to sample from.

        Returns:
            List of message tensors from the population.
        """
        messages = []

        # Create a small dataset for sampling
        dataset = ReferentialGameDataset(n_scenes=100, k=self.config.k, seed=42)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        for pair in population.pairs:
            pair.speaker.eval()

            with torch.no_grad():
                for batch in dataloader:
                    scene_tensor, target_indices, _ = batch
                    scene_tensor = scene_tensor.to(self.device)
                    target_indices = target_indices.to(self.device)

                    # Extract target objects
                    batch_size = scene_tensor.size(0)
                    target_objects = scene_tensor[
                        torch.arange(batch_size), target_indices
                    ]

                    # Generate messages
                    _, message_tokens = pair.speaker(target_objects)
                    messages.append(message_tokens)

        return messages

    def _messages_to_distribution(
        self, messages: List[torch.Tensor], vocab_size: int
    ) -> np.ndarray:
        """Convert message tensors to probability distribution.

        Args:
            messages: List of message tensors.
            vocab_size: Size of vocabulary.

        Returns:
            Probability distribution over vocabulary.
        """
        counts = np.zeros(vocab_size)
        total_tokens = 0

        for message_batch in messages:
            for message in message_batch:
                for token in message:
                    token_id = token.item()
                    if 0 <= token_id < vocab_size:
                        counts[token_id] += 1
                        total_tokens += 1

        # Normalize to probabilities
        if total_tokens > 0:
            distribution = counts / total_tokens
        else:
            distribution = np.ones(vocab_size) / vocab_size

        return np.asarray(distribution, dtype=np.float64)  # type: ignore

    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence between two distributions.

        Args:
            p: First probability distribution.
            q: Second probability distribution.

        Returns:
            JSD score between 0 and 1.
        """
        # Ensure distributions are normalized
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Compute average distribution
        m = (p + q) / 2

        # Compute KL divergences with proper handling of zeros
        kl_pm: float = np.sum(np.where(p > 0, p * np.log(p / m), 0))
        kl_qm: float = np.sum(np.where(q > 0, q * np.log(q / m), 0))

        # JSD is the average of the KL divergences
        jsd: float = (kl_pm + kl_qm) / 2

        return jsd

    def save_results(self) -> None:
        """Save intelligibility results and visualizations."""
        logger.info("Saving intelligibility results")

        # Save intelligibility matrix as CSV
        if self.intelligibility_matrix is not None:
            matrix_file = "outputs/M.csv"
            np.savetxt(
                matrix_file, self.intelligibility_matrix, delimiter=",", fmt="%.4f"
            )
            logger.info(f"Intelligibility matrix saved to {matrix_file}")

        # Save JSD score as JSON
        jsd_file = "outputs/jsd.json"
        with open(jsd_file, "w") as f:
            json.dump({"jsd": self.jsd_score}, f, indent=2)
        logger.info(f"JSD score saved to {jsd_file}")

        # Create and save heatmap
        if self.intelligibility_matrix is not None:
            self._create_heatmap()

    def _create_heatmap(self) -> None:
        """Create and save intelligibility heatmap visualization."""
        plt.figure(figsize=(10, 8))

        # Create heatmap
        if self.intelligibility_matrix is not None:
            sns.heatmap(
                self.intelligibility_matrix,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                cbar_kws={"label": "Intelligibility"},
                xticklabels=[
                    f"Listener_B{i}"
                    for i in range(self.intelligibility_matrix.shape[1])
                ],
                yticklabels=[
                    f"Speaker_A{i}" for i in range(self.intelligibility_matrix.shape[0])
                ],
            )

        plt.title("Mutual Intelligibility Matrix")
        plt.xlabel("Population B Listeners")
        plt.ylabel("Population A Speakers")
        plt.tight_layout()

        # Save heatmap
        heatmap_file = "outputs/figures/intelligibility_heatmap.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Heatmap saved to {heatmap_file}")


def train_contact_experiment(
    n_pairs: int = 4,
    steps_a: int = 4000,
    steps_b: int = 4000,
    contact_steps: int = 2000,
    p_contact: float = 0.3,
    k: int = 5,
    v: int = 10,
    message_length: int = 1,
    seed_a: int = 42,
    seed_b: int = 123,
    log_every: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    use_sequence_models: bool = False,
    entropy_weight: float = 0.01,
    heldout_pairs_a: Optional[List[Tuple[str, str]]] = None,
    heldout_pairs_b: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Train two populations separately, then bring them into contact and measure intelligibility.

    This function runs the complete contact experiment pipeline:
    1. Stage A: Train both populations separately
    2. Stage B: Train populations in contact phase
    3. Measure mutual intelligibility and Jensen-Shannon divergence
    4. Save results and visualizations

    Args:
        n_pairs: Number of agent pairs per population.
        steps_a: Number of training steps for Population A.
        steps_b: Number of training steps for Population B.
        contact_steps: Number of contact phase steps.
        p_contact: Probability of cross-population interactions.
        k: Number of objects per scene.
        v: Vocabulary size.
        message_length: Message length.
        seed_a: Random seed for Population A.
        seed_b: Random seed for Population B.
        log_every: Frequency of logging.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizers.
        hidden_size: Hidden dimension for neural networks.
        use_sequence_models: Whether to use sequence-aware models.
        entropy_weight: Weight for entropy bonus regularization.
        heldout_pairs_a: Held-out attribute pairs for Population A.
        heldout_pairs_b: Held-out attribute pairs for Population B.
    """
    # Create configuration
    config = ContactConfig(
        n_pairs=n_pairs,
        steps_a=steps_a,
        steps_b=steps_b,
        contact_steps=contact_steps,
        p_contact=p_contact,
        k=k,
        v=v,
        message_length=message_length,
        seed_a=seed_a,
        seed_b=seed_b,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        use_sequence_models=use_sequence_models,
        entropy_weight=entropy_weight,
        heldout_pairs_a=heldout_pairs_a,
        heldout_pairs_b=heldout_pairs_b,
    )

    # Create experiment
    experiment = ContactExperiment(config)

    # Run experiment
    experiment.train_stage_a()
    experiment.train_stage_b()
    experiment.measure_intelligibility()
    experiment.save_results()

    logger.info("Contact experiment completed successfully!")
