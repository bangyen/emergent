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

        This method takes speaker logits and converts them to discrete tokens,
        enforcing the vocabulary range [0, vocabulary_size-1]. During training,
        it supports straight-through gradients for differentiable discrete sampling.

        Args:
            speaker_logits: Tensor of shape (batch_size, message_length, vocabulary_size)
                          containing raw logits from the speaker.
            temperature: Temperature for sampling (default: 1.0).

        Returns:
            Tensor of shape (batch_size, message_length) containing discrete token IDs
            in the range [0, vocabulary_size-1].

        Raises:
            ValueError: If logits tensor has incorrect shape or vocabulary size mismatch.
        """
        batch_size, message_length, vocab_size = speaker_logits.shape

        # Validate input shape
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

        # Sample tokens using Gumbel-Softmax for differentiable discrete sampling
        if speaker_logits.requires_grad:
            # Training mode: use Gumbel-Softmax with straight-through
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(scaled_logits) + 1e-20) + 1e-20
            )
            noisy_logits = scaled_logits + gumbel_noise
            probs = F.softmax(noisy_logits, dim=-1)

            # Straight-through: use hard samples in forward pass
            token_ids = torch.argmax(probs, dim=-1)
        else:
            # Inference mode: use argmax for discrete tokens
            token_ids = torch.argmax(scaled_logits, dim=-1)

        # Enforce token range constraint
        token_ids = torch.clamp(token_ids, 0, self.config.vocabulary_size - 1)

        return token_ids

    def send_multimodal(
        self,
        speaker_logits: torch.Tensor,
        gesture_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transmit multimodal message through the discrete channel.

        This method takes speaker logits for both tokens and gestures and converts
        them to discrete outputs, enforcing vocabulary and gesture range constraints.
        During training, it supports straight-through gradients for differentiable
        discrete sampling.

        Args:
            speaker_logits: Tensor of shape (batch_size, message_length, vocabulary_size)
                          containing raw logits from the speaker for tokens.
            gesture_logits: Tensor of shape (batch_size, message_length, gesture_size)
                          containing raw logits from the speaker for gestures.
            temperature: Temperature for sampling (default: 1.0).

        Returns:
            A tuple containing:
            - token_ids: Tensor of shape (batch_size, message_length) with discrete token IDs
            - gesture_ids: Tensor of shape (batch_size, message_length) with discrete gesture IDs

        Raises:
            ValueError: If logits tensors have incorrect shapes or size mismatches.
        """
        batch_size, message_length, vocab_size = speaker_logits.shape
        _, _, gesture_size = gesture_logits.shape

        # Validate input shapes
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
        if gesture_size != self.config.gesture_size:
            raise ValueError(
                f"Expected gesture_size={self.config.gesture_size}, got {gesture_size}"
            )

        # Sample tokens
        token_ids = self.send(speaker_logits, temperature)

        # Sample gestures using the same logic
        scaled_gesture_logits = gesture_logits / temperature

        if gesture_logits.requires_grad:
            # Training mode: use Gumbel-Softmax with straight-through
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(scaled_gesture_logits) + 1e-20) + 1e-20
            )
            noisy_gesture_logits = scaled_gesture_logits + gumbel_noise
            gesture_probs = F.softmax(noisy_gesture_logits, dim=-1)
            gesture_ids = torch.argmax(gesture_probs, dim=-1)
        else:
            # Inference mode: use argmax for discrete gestures
            gesture_ids = torch.argmax(scaled_gesture_logits, dim=-1)

        # Enforce gesture range constraint
        gesture_ids = torch.clamp(gesture_ids, 0, self.config.gesture_size - 1)

        return token_ids, gesture_ids

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
