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
    neural network architecture, multimodal support, and reproducibility settings.

    Attributes:
        vocabulary_size: Number of tokens in the communication vocabulary (default: 10).
        message_length: Length of messages in tokens (default: 1).
        hidden_size: Hidden dimension size for neural networks (default: 64).
        object_dim: Dimension of object encodings (default: 5).
        gesture_size: Number of discrete gestures in multimodal communication (default: 5).
        multimodal: Whether to enable multimodal communication with gestures (default: False).
        distractors: Number of distractor objects in pragmatic scenarios (default: 0).
        pragmatic: Whether to enable pragmatic inference (default: False).
        use_sequence_models: Whether to use sequence-aware models (SpeakerSeq/ListenerSeq) (default: False).
        seed: Random seed for reproducibility (default: None).
        use_attention: Whether to use attention mechanisms in Listener (default: True).
        use_residual: Whether to use residual connections (default: True).
        dropout_rate: Dropout rate for regularization (default: 0.1).
    """

    vocabulary_size: int = 16
    message_length: int = 2
    hidden_size: int = 128
    object_dim: int = 8
    gesture_size: int = 5
    multimodal: bool = False
    distractors: int = 0
    pragmatic: bool = False
    use_sequence_models: bool = False
    seed: Optional[int] = None
    use_attention: bool = True
    use_residual: bool = True
    dropout_rate: float = 0.1

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
        if self.gesture_size <= 0:
            raise ValueError("gesture_size must be positive")
        if self.distractors < 0:
            raise ValueError("distractors must be non-negative")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be between 0.0 and 1.0")
