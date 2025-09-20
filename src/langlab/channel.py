"""Discrete communication channel for agent interactions.

This module implements a discrete communication channel that handles message
transmission between Speaker and Listener agents, ensuring proper token
constraints and supporting differentiable training.
"""

import torch
import torch.nn.functional as F

from .config import CommunicationConfig


class DiscreteChannel:
    """Discrete communication channel for agent message transmission.

    The DiscreteChannel handles the transmission of messages between agents,
    enforcing token range constraints and supporting straight-through gradients
    during training for differentiable discrete communication.

    Args:
        config: Communication configuration containing vocabulary parameters.
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

    def get_token_range(self) -> tuple[int, int]:
        """Get the valid token range for this channel.

        Returns:
            Tuple of (min_token, max_token) representing the valid range [0, vocabulary_size-1].
        """
        return (0, self.config.vocabulary_size - 1)
