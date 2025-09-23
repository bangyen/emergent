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

    def validate_tokens(self, token_ids: torch.Tensor) -> bool:
        """Validate that token IDs are within the allowed range.

        Args:
            token_ids: Tensor containing token IDs to validate.

        Returns:
            True if all tokens are in range [0, vocabulary_size-1], False otherwise.
        """
        min_token = token_ids.min().item()
        max_token = token_ids.max().item()

        return (
            min_token >= 0
            and max_token < self.config.vocabulary_size
            and token_ids.dtype in [torch.int64, torch.int32, torch.long]
        )

    def validate_gestures(self, gesture_ids: torch.Tensor) -> bool:
        """Validate that gesture IDs are within the allowed range.

        Args:
            gesture_ids: Tensor containing gesture IDs to validate.

        Returns:
            True if all gestures are in range [0, gesture_size-1], False otherwise.
        """
        min_gesture = gesture_ids.min().item()
        max_gesture = gesture_ids.max().item()

        return (
            min_gesture >= 0
            and max_gesture < self.config.gesture_size
            and gesture_ids.dtype in [torch.int64, torch.int32, torch.long]
        )

    def get_token_range(self) -> tuple[int, int]:
        """Get the valid token range for this channel.

        Returns:
            Tuple of (min_token, max_token) representing the valid range [0, vocabulary_size-1].
        """
        return (0, self.config.vocabulary_size - 1)

    def get_gesture_range(self) -> tuple[int, int]:
        """Get the valid gesture range for this channel.

        Returns:
            Tuple of (min_gesture, max_gesture) representing the valid range [0, gesture_size-1].
        """
        return (0, self.config.gesture_size - 1)
