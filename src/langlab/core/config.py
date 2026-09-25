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

    Attributes:
        vocabulary_size: Number of tokens in the communication vocabulary (default: 16).
        message_length: Length of messages in tokens (default: 2).
        hidden_size: Hidden dimension size for neural networks (default: 128).
        object_dim: Size of the one-hot object encoding (default: 8, the default world).
        dropout: Dropout rate in the MLP Speaker/Listener (default: 0.0). Dropout
            makes the speaker's policy noisy and tends to collapse its messages.
        listener_type: "mlp" (default) or "dot" for the additive DotListener
            (ignored by sequence models).
        use_sequence_models: Whether to use sequence-aware models (SpeakerSeq/ListenerSeq) (default: False).
        seed: Random seed for reproducibility (default: None).
    """

    vocabulary_size: int = 16
    message_length: int = 2
    hidden_size: int = 128
    object_dim: int = 8
    dropout: float = 0.0
    listener_type: str = "mlp"
    use_sequence_models: bool = False
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
        if self.listener_type not in ("mlp", "dot"):
            raise ValueError("listener_type must be 'mlp' or 'dot'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
