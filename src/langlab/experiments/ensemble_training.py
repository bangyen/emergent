"""Ensemble training for emergent language models.

This module implements ensemble training where multiple models are trained
with different random seeds and their predictions are combined for improved accuracy.
"""

import os
import torch
import numpy as np
from typing import List, Tuple
from torch.utils.data import DataLoader

from ..core.agents import Speaker, Listener
from ..core.config import CommunicationConfig
from ..data.data import ReferentialGameDataset
from ..training.train import train
from ..utils.utils import get_logger, get_device, set_seed

logger = get_logger(__name__)


class EnsembleSpeaker:
    """Ensemble of Speaker models for improved message generation."""

    def __init__(self, speakers: List[Speaker]):
        self.speakers = speakers
        self.config = speakers[0].config

    def eval(self) -> None:
        """Set all speakers to evaluation mode."""
        for speaker in self.speakers:
            speaker.eval()

    def train(self) -> None:
        """Set all speakers to training mode."""
        for speaker in self.speakers:
            speaker.train()

    def __call__(
        self, object_encoding: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """Make the ensemble callable."""
        return self.forward(object_encoding, temperature)

    def forward(
        self, object_encoding: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """Generate ensemble predictions by averaging logits from all speakers."""
        all_logits = []
        all_token_ids = []

        for speaker in self.speakers:
            logits, token_ids, _, _ = speaker(object_encoding, temperature)
            all_logits.append(logits)
            all_token_ids.append(token_ids)

        # Average logits across ensemble
        ensemble_logits = torch.stack(all_logits).mean(dim=0)

        # Sample from ensemble logits
        ensemble_token_ids = torch.argmax(ensemble_logits, dim=-1)

        return ensemble_logits, ensemble_token_ids, None, None


class EnsembleListener:
    """Ensemble of Listener models for improved object identification."""

    def __init__(self, listeners: List[Listener]):
        self.listeners = listeners
        self.config = listeners[0].config

    def eval(self) -> None:
        """Set all listeners to evaluation mode."""
        for listener in self.listeners:
            listener.eval()

    def train(self) -> None:
        """Set all listeners to training mode."""
        for listener in self.listeners:
            listener.train()

    def __call__(
        self, message_tokens: torch.Tensor, candidate_objects: torch.Tensor
    ) -> torch.Tensor:
        """Make the ensemble callable."""
        return self.forward(message_tokens, candidate_objects)

    def forward(
        self, message_tokens: torch.Tensor, candidate_objects: torch.Tensor
    ) -> torch.Tensor:
        """Generate ensemble predictions by averaging probabilities from all listeners."""
        all_probabilities = []

        for listener in self.listeners:
            probs = listener(message_tokens, candidate_objects)
            all_probabilities.append(probs)

        # Average probabilities across ensemble
        ensemble_probabilities = torch.stack(all_probabilities).mean(dim=0)

        return ensemble_probabilities


def train_ensemble(
    n_models: int = 3,
    n_steps: int = 1000,
    k: int = 5,
    v: int = 16,
    message_length: int = 2,
    hidden_size: int = 64,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    base_seed: int = 42,
) -> Tuple[EnsembleSpeaker, EnsembleListener, float]:
    """Train an ensemble of models with different random seeds.

    Args:
        n_models: Number of models in the ensemble.
        n_steps: Number of training steps per model.
        k: Number of objects per scene.
        v: Vocabulary size.
        message_length: Message length.
        hidden_size: Hidden layer size.
        batch_size: Batch size.
        learning_rate: Learning rate.
        base_seed: Base random seed.

    Returns:
        Tuple of (ensemble_speaker, ensemble_listener, best_accuracy).
    """
    speakers = []
    listeners = []
    accuracies = []

    logger.info(f"Training ensemble of {n_models} models...")

    for i in range(n_models):
        seed = base_seed + i * 1000
        logger.info(f"Training model {i+1}/{n_models} with seed {seed}")

        # Train individual model
        train(
            n_steps=n_steps,
            k=k,
            v=v,
            message_length=message_length,
            seed=seed,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            use_sequence_models=False,
            entropy_weight=0.01,
            length_weight=0.0,
            heldout_pairs=None,
            multimodal=False,
            distractors=0,
            temperature_start=2.0,
            temperature_end=0.5,
            use_curriculum=True,
            use_warmup=True,
            use_ema=True,
            use_early_stopping=True,
            early_stopping_patience=20,
            early_stopping_min_delta=0.001,
            log_every=100,
            eval_every=500,
            lambda_speaker=1.0,
        )

        # Read accuracy from metrics file
        accuracy = 0.0
        metrics_path = "outputs/logs/metrics.csv"
        if os.path.exists(metrics_path):
            import csv

            with open(metrics_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    accuracy = float(rows[-1]["accuracy"])

        accuracies.append(accuracy)

        # Load the trained models
        config = CommunicationConfig(
            vocabulary_size=v,
            message_length=message_length,
            hidden_size=hidden_size,
            multimodal=False,
            gesture_size=1,  # Set to 1 when multimodal=False
        )

        speaker = Speaker(config)
        listener = Listener(config)

        # Load checkpoint
        checkpoint_path = "outputs/checkpoints/checkpoint.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=get_device(), weights_only=False
            )
            speaker.load_state_dict(checkpoint["speaker_state_dict"])
            listener.load_state_dict(checkpoint["listener_state_dict"])
            logger.info(f"Loaded checkpoint for model {i+1}")
        else:
            logger.warning(f"No checkpoint found for model {i+1}")

        speakers.append(speaker)
        listeners.append(listener)

    # Create ensemble models
    ensemble_speaker = EnsembleSpeaker(speakers)
    ensemble_listener = EnsembleListener(listeners)

    avg_accuracy = np.mean(accuracies)
    logger.info(f"Ensemble training completed. Individual accuracies: {accuracies}")
    logger.info(f"Average accuracy: {avg_accuracy:.4f}")

    return ensemble_speaker, ensemble_listener, float(avg_accuracy)


def evaluate_ensemble(
    ensemble_speaker: EnsembleSpeaker,
    ensemble_listener: EnsembleListener,
    k: int = 5,
    n_samples: int = 1000,
    seed: int = 42,
) -> float:
    """Evaluate the ensemble model on a test set.

    Args:
        ensemble_speaker: Ensemble speaker model.
        ensemble_listener: Ensemble listener model.
        k: Number of objects per scene.
        n_samples: Number of test samples.
        seed: Random seed for evaluation.

    Returns:
        Test accuracy.
    """
    set_seed(seed)
    device = get_device()

    # Create test dataset
    test_dataset = ReferentialGameDataset(
        n_scenes=n_samples,
        k=k,
        seed=seed + 10000,  # Different seed for test data
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    ensemble_speaker.eval()
    ensemble_listener.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            target_objects, candidate_objects, target_indices = batch
            target_objects = target_objects.to(device)
            candidate_objects = candidate_objects.to(device)
            target_indices = target_indices.to(device)

            # Generate messages using ensemble speaker
            speaker_output = ensemble_speaker(target_objects)
            if len(speaker_output) == 4:
                _, message_tokens, _, _ = speaker_output
            else:
                _, message_tokens = speaker_output

            # Get predictions using ensemble listener
            probabilities = ensemble_listener(message_tokens, candidate_objects)
            predictions = torch.argmax(probabilities, dim=1)

            # Calculate accuracy
            correct += (predictions == target_indices).sum().item()
            total += target_indices.size(0)

    accuracy = correct / total
    logger.info(f"Ensemble test accuracy: {accuracy:.4f}")

    return accuracy


if __name__ == "__main__":
    # Train ensemble
    ensemble_speaker, ensemble_listener, train_accuracy = train_ensemble(
        n_models=3,
        n_steps=1000,
        k=5,
        v=16,
        message_length=2,
        hidden_size=64,
        batch_size=16,
        learning_rate=2e-4,
        base_seed=42,
    )

    # Evaluate ensemble
    test_accuracy = evaluate_ensemble(
        ensemble_speaker,
        ensemble_listener,
        k=5,
        n_samples=1000,
        seed=42,
    )

    print("Ensemble Results:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
