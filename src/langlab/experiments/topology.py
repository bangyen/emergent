"""Social topology experiments for studying networked agent populations.

This module implements a SocialPopulationManager that restricts agent interactions
to a specified social network topology (e.g., ring, small-world, scale-free).
"""

import os
import random
import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

from torch.utils.data import DataLoader

from .population import PopulationManager, PopulationConfig
from ..data.data import ReferentialGameDataset
from ..utils.utils import get_logger, set_seed

logger = get_logger(__name__)


@dataclass
class TopologyConfig(PopulationConfig):
    """Configuration for networked population experiments.

    Attributes:
        topology_type: Type of network structure ('ring', 'small-world', 'scale-free', 'fully-connected').
        p_rewire: Rewiring probability for small-world networks.
        m_attach: Number of edges to attach for scale-free networks.
    """

    topology_type: str = "ring"
    p_rewire: float = 0.1
    m_attach: int = 2


class SocialPopulationManager(PopulationManager):
    """Manages a population where interactions are constrained by a social network.

    This class extends PopulationManager by maintaining an adjacency list
    representing the social ties between agent pairs.
    """

    def __init__(self, config: TopologyConfig):
        """Initialize the social population manager.

        Args:
            config: Topology configuration parameters.
        """
        super().__init__(config)
        self.topology_config = config
        self.adj_list: Dict[int, Set[int]] = {}
        self._initialize_topology()

    def _initialize_topology(self) -> None:
        """Construct the social network based on the specified topology."""
        n = self.config.n_pairs
        self.adj_list = {i: set() for i in range(n)}

        if self.topology_config.topology_type == "fully-connected":
            for i in range(n):
                for j in range(i + 1, n):
                    self._add_edge(i, j)

        elif self.topology_config.topology_type == "ring":
            for i in range(n):
                self._add_edge(i, (i + 1) % n)

        elif self.topology_config.topology_type == "small-world":
            # Start with a ring (k=2 neighbors)
            for i in range(n):
                self._add_edge(i, (i + 1) % n)

            # Rewire edges with probability p_rewire
            for i in range(n):
                if random.random() < self.topology_config.p_rewire:
                    # Remove the regular neighbor link (simple version: just add a new link)
                    target = random.randint(0, n - 1)
                    if target != i:
                        self._add_edge(i, target)

        elif self.topology_config.topology_type == "scale-free":
            # Barabási-Albert model (simplified)
            # Start with a small clique
            m = min(self.topology_config.m_attach, n - 1)
            for i in range(m + 1):
                for j in range(i + 1, m + 1):
                    self._add_edge(i, j)

            # Growth with preferential attachment
            for i in range(m + 1, n):
                # Calculate attachment probabilities based on degrees
                degrees = [len(self.adj_list[j]) for j in range(i)]
                total_degree = sum(degrees)
                if total_degree == 0:
                    targets = random.sample(range(i), m)
                else:
                    probs = [d / total_degree for d in degrees]
                    targets = [
                        int(t)
                        for t in np.random.choice(
                            range(i), size=m, replace=False, p=probs
                        )
                    ]

                for target in targets:
                    self._add_edge(i, target)

        else:
            logger.warning(
                f"Unknown topology '{self.topology_config.topology_type}'. Defaulting to fully-connected."
            )
            for i in range(n):
                for j in range(i + 1, n):
                    self._add_edge(i, j)

        logger.info(
            f"Social network initialized: type={self.topology_config.topology_type}, "
            f"avg_degree={np.mean([len(v) for v in self.adj_list.values()]):.2f}"
        )

    def _add_edge(self, u: int, v: int) -> None:
        """Add an undirected edge between two agent pairs."""
        if u != v:
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)

    def _select_interaction_pairs(self) -> List[Tuple[int, int]]:
        """Select interaction pairs constrained by the social network.

        Returns:
            List of (speaker_pair_idx, listener_pair_idx) tuples.
        """
        interactions = []

        for i in range(self.config.n_pairs):
            # Self-play (always permitted)
            interactions.append((i, i))

            # Cross-play with probability crossplay_prob, only if connected in the network
            if random.random() < self.config.crossplay_prob:
                neighbors = list(self.adj_list[i])
                if neighbors:
                    partner = random.choice(neighbors)
                    interactions.append((i, partner))

        return interactions

    def get_dialect_distance_matrix(self) -> np.ndarray:
        """Compute the linguistic distance between all pairs in the population.

        This uses the Jaccard distance between vocabulary usage histograms as a proxy
        for dialectal distance.

        Returns:
            N x N distance matrix.
        """
        n = self.config.n_pairs
        matrix = np.zeros((n, n))

        hist_lists = [
            pair.get_vocab_histogram(self.config.vocabulary_size) for pair in self.pairs
        ]
        norm_hists = [np.array(h) / (sum(h) + 1e-10) for h in hist_lists]

        for i in range(n):
            for j in range(i + 1, n):
                # Total Variation Distance as a simple dialect distance
                dist = 0.5 * np.sum(np.abs(norm_hists[i] - norm_hists[j]))
                matrix[i, j] = matrix[j, i] = dist

        return matrix


def train_topology_experiment(
    n_steps: int,
    n_pairs: int = 10,
    topology_type: str = "ring",
    p_rewire: float = 0.1,
    m_attach: int = 2,
    lifespan: int = 1000,
    crossplay_prob: float = 0.5,
    use_fitness_selection: bool = True,
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
    """Train a networked population of agent pairs.

    This function runs the main training loop for social topology experiments.
    """
    set_seed(seed)
    os.makedirs("outputs/logs", exist_ok=True)

    config = TopologyConfig(
        n_pairs=n_pairs,
        topology_type=topology_type,
        p_rewire=p_rewire,
        m_attach=m_attach,
        lifespan=lifespan,
        replacement_noise=replacement_noise,
        crossplay_prob=crossplay_prob,
        use_fitness_selection=use_fitness_selection,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        vocabulary_size=v,
        message_length=message_length,
        use_sequence_models=use_sequence_models,
        entropy_weight=entropy_weight,
        seed=seed,
    )

    population = SocialPopulationManager(config)
    dataset = ReferentialGameDataset(n_scenes=n_steps * batch_size, k=k, seed=seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(
        f"Starting topology training: {topology_type}, {n_pairs} pairs, {n_steps} steps"
    )

    step = 0
    dataloader_iter = iter(dataloader)

    import csv

    metrics_file = "outputs/logs/topology_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "accuracy", "avg_age", "avg_accuracy"])

    while step < n_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        metrics = population.train_step(batch)
        stats = population.get_population_stats()

        if step % log_every == 0:
            logger.info(
                f"Step {step}: Loss={metrics['total_loss']:.4f}, "
                f"Acc={metrics['accuracy']:.4f}, Avg Age={stats['avg_age']:.1f}"
            )
            with open(metrics_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        step,
                        metrics["total_loss"],
                        metrics["accuracy"],
                        stats["avg_age"],
                        stats["avg_accuracy"],
                    ]
                )

        step += 1

    population.save_logs()

    # Save distance matrix
    dist_matrix = population.get_dialect_distance_matrix()
    np.savetxt("outputs/dialect_distances.csv", dist_matrix, delimiter=",")
    logger.info("Saved dialect distance matrix to outputs/dialect_distances.csv")
