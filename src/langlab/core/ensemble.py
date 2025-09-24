"""Ensemble methods for emergent language models.

This module implements ensemble techniques to improve model performance
by combining multiple trained models for better predictions.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn

from .agents import Speaker, Listener
from .config import CommunicationConfig


class EnsembleListener(nn.Module):
    """Ensemble of multiple Listener models for improved predictions.

    This class combines predictions from multiple trained Listener models
    to achieve better accuracy through ensemble averaging.
    """

    def __init__(
        self, listeners: List[Listener], weights: Optional[List[float]] = None
    ):
        """Initialize ensemble listener.

        Args:
            listeners: List of trained Listener models.
            weights: Optional weights for each model (default: equal weights).
        """
        super().__init__()
        self.listeners = nn.ModuleList(listeners)
        self.num_models = len(listeners)

        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            assert (
                len(weights) == self.num_models
            ), "Number of weights must match number of models"
            # Normalize weights
            weight_sum = sum(weights)
            self.weights = [w / weight_sum for w in weights]

    def forward(
        self,
        message_tokens: torch.Tensor,
        candidate_objects: torch.Tensor,
        gesture_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ensemble predictions.

        Args:
            message_tokens: Message tokens tensor.
            candidate_objects: Candidate objects tensor.
            gesture_tokens: Optional gesture tokens tensor.

        Returns:
            Ensemble-averaged probability distribution over candidates.
        """
        predictions = []

        for i, listener in enumerate(self.listeners):
            if gesture_tokens is not None:
                pred = listener(message_tokens, candidate_objects, gesture_tokens)
            else:
                pred = listener(message_tokens, candidate_objects)
            predictions.append(pred * self.weights[i])

        # Average predictions
        ensemble_pred = torch.stack(predictions, dim=0).sum(dim=0)

        return ensemble_pred


class EnsembleSpeaker(nn.Module):
    """Ensemble of multiple Speaker models for improved message generation.

    This class combines predictions from multiple trained Speaker models
    to generate more diverse and accurate messages.
    """

    def __init__(self, speakers: List[Speaker], weights: Optional[List[float]] = None):
        """Initialize ensemble speaker.

        Args:
            speakers: List of trained Speaker models.
            weights: Optional weights for each model (default: equal weights).
        """
        super().__init__()
        self.speakers = nn.ModuleList(speakers)
        self.num_models = len(speakers)

        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            assert (
                len(weights) == self.num_models
            ), "Number of weights must match number of models"
            # Normalize weights
            weight_sum = sum(weights)
            self.weights = [w / weight_sum for w in weights]

    def forward(
        self, object_encoding: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Generate ensemble messages.

        Args:
            object_encoding: Object encoding tensor.
            temperature: Temperature for sampling.

        Returns:
            Tuple of (logits, tokens, gesture_logits, gesture_tokens).
        """
        all_logits = []
        all_tokens = []
        all_gesture_logits = []
        all_gesture_tokens = []

        for i, speaker in enumerate(self.speakers):
            logits, tokens, gesture_logits, gesture_tokens = speaker(
                object_encoding, temperature
            )

            # Weight the predictions
            all_logits.append(logits * self.weights[i])
            all_tokens.append(tokens)
            if gesture_logits is not None:
                all_gesture_logits.append(gesture_logits * self.weights[i])
            if gesture_tokens is not None:
                all_gesture_tokens.append(gesture_tokens)

        # Average logits and select most common tokens
        ensemble_logits = torch.stack(all_logits, dim=0).sum(dim=0)

        # Handle gesture logits if available
        if all_gesture_logits:
            ensemble_gesture_logits = torch.stack(all_gesture_logits, dim=0).sum(dim=0)
        else:
            ensemble_gesture_logits = None

        # For tokens, use majority voting or select from best model
        ensemble_tokens = all_tokens[0]  # Simple: use first model's tokens
        ensemble_gesture_tokens = all_gesture_tokens[0] if all_gesture_tokens else None

        return (
            ensemble_logits,
            ensemble_tokens,
            ensemble_gesture_logits,
            ensemble_gesture_tokens,
        )


def create_ensemble_from_checkpoints(
    checkpoint_paths: List[str],
    config: CommunicationConfig,
    device: Optional[torch.device] = None,
) -> Tuple[EnsembleSpeaker, EnsembleListener]:
    """Create ensemble models from multiple checkpoints.

    Args:
        checkpoint_paths: List of paths to model checkpoints.
        config: Communication configuration.
        device: Device to load models on.

    Returns:
        Tuple of (ensemble_speaker, ensemble_listener).
    """
    if device is None:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    speakers = []
    listeners = []

    for checkpoint_path in checkpoint_paths:
        import torch

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # Create models
        speaker = Speaker(config).to(device)
        listener = Listener(config).to(device)

        # Load state dicts
        speaker.load_state_dict(checkpoint["speaker_state_dict"])
        listener.load_state_dict(checkpoint["listener_state_dict"])

        speakers.append(speaker)
        listeners.append(listener)

    # Create ensemble models
    ensemble_speaker = EnsembleSpeaker(speakers)
    ensemble_listener = EnsembleListener(listeners)

    return ensemble_speaker, ensemble_listener
