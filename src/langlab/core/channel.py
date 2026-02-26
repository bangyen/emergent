"""Discrete communication channel for agent interactions.

This module implements a discrete communication channel that handles message
transmission between Speaker and Listener agents, ensuring proper token
constraints and supporting differentiable training. Supports multimodal
communication with parallel gesture streams.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

from .config import CommunicationConfig


class DiscreteChannel:
    """Discrete communication channel for agent message transmission.

    The DiscreteChannel handles the transmission of messages between agents,
    enforcing token range constraints and supporting straight-through gradients
    during training for differentiable discrete communication. Supports both
    unimodal (tokens only) and multimodal (tokens + gestures) communication.

    Args:
        config: Communication configuration containing vocabulary and multimodal parameters.
    """

    def __init__(self, config: CommunicationConfig):
        self.config = config
        # Initialize default costs (can be modified for experimental purposes)
        self.token_costs = torch.ones(config.vocabulary_size)

    def send(
        self, speaker_logits: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """Transmit message through the discrete channel.

        This method handles the conversion of raw logits to discrete tokens.
        During training, it maintains differentiability using straight-through
        gradients.

        Args:
            speaker_logits: Tensor of shape (batch_size, message_length, vocabulary_size)
                          containing raw logits from the speaker.
            temperature: Temperature for sampling (default: 1.0).

        Returns:
            Tensor of shape (batch_size, message_length) containing discrete token indices.
        """
        batch_size, message_length, vocab_size = speaker_logits.shape

        if message_length != self.config.message_length:
            raise ValueError(
                f"Expected message_length={self.config.message_length}, "
                f"got {message_length}"
            )
        if vocab_size != self.config.vocabulary_size:
            raise ValueError(
                f"Expected vocabulary_size={self.config.vocabulary_size}, "
                f"got {vocab_size}"
            )

        # Apply temperature scaling
        scaled_logits = speaker_logits / temperature

        if speaker_logits.requires_grad:
            # Training mode: differentiable discrete sampling
            probs = F.gumbel_softmax(scaled_logits, tau=1.0, hard=True, dim=-1)
            token_ids = torch.argmax(probs, dim=-1)
        else:
            # Inference mode: greedy sampling
            token_ids = torch.argmax(scaled_logits, dim=-1)

        return token_ids

    def send_multimodal(
        self,
        speaker_logits: torch.Tensor,
        gesture_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transmit multimodal message through the discrete channel.

        Args:
            speaker_logits: Tensor of shape (batch_size, message_length, vocabulary_size).
            gesture_logits: Tensor of shape (batch_size, message_length, gesture_size).
            temperature: Temperature for sampling (default: 1.0).

        Returns:
            A tuple of (tokens, gestures).
        """
        tokens = self.send(speaker_logits, temperature)

        # Sample gestures using similar logic
        if gesture_logits.requires_grad:
            gesture_probs = F.gumbel_softmax(
                gesture_logits / temperature, tau=1.0, hard=True, dim=-1
            )
            gestures = torch.argmax(gesture_probs, dim=-1)
        else:
            gestures = torch.argmax(gesture_logits / temperature, dim=-1)

        return tokens, gestures

    def compute_message_cost(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute the total cost of a message based on token-specific costs.

        Args:
            token_ids: Tensor of shape (batch_size, message_length) containing token IDs.

        Returns:
            Tensor of shape (batch_size,) containing the total cost for each message.
        """
        # Ensure token_costs is on the same device as token_ids
        costs = self.token_costs.to(token_ids.device)
        return costs[token_ids].sum(dim=1)

    def set_token_costs(self, costs: torch.Tensor) -> None:
        """Set custom costs for tokens.

        Args:
            costs: Tensor of shape (vocabulary_size,) containing costs for each token.
        """
        if costs.shape[0] != self.config.vocabulary_size:
            raise ValueError(
                f"Cost tensor must have size {self.config.vocabulary_size}"
            )
        self.token_costs = costs
