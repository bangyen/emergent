"""Configuration dataclasses for communication parameters.

This module defines the configuration structures used to control communication
parameters in referential games, including vocabulary size, message length,
and neural network architecture settings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CommunicationConfig:
    """Configuration for communication parameters in referential games.

    This dataclass encapsulates all the key parameters that control how agents
    communicate in referential games, including vocabulary size, message length,
    neural network architecture, and reproducibility settings.

    Attributes:
        vocabulary_size: Number of tokens in the communication vocabulary (default: 10).
        message_length: Length of messages in tokens (default: 1).
        hidden_size: Hidden dimension size for neural networks (default: 64).
        seed: Random seed for reproducibility (default: None).
    """

    vocabulary_size: int = 10
    message_length: int = 1
    hidden_size: int = 64
    object_dim: int = 5
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        if self.message_length <= 0:
            raise ValueError("message_length must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.object_dim <= 0:
            raise ValueError("object_dim must be positive")
