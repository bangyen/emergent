"""Contrastive learning agents for referential games.

This module implements Speaker and Listener agents with contrastive learning
to improve representation quality and model performance.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CommunicationConfig
from ..data.world import TOTAL_ATTRIBUTES


class ContrastiveSpeaker(nn.Module):
    """Speaker agent with contrastive learning for better representations.

    This speaker uses contrastive learning to learn better object representations
    by contrasting positive and negative pairs in the representation space.
    """

    def __init__(self, config: CommunicationConfig):
        super().__init__()
        self.config = config
        self.input_dim = TOTAL_ATTRIBUTES

        # Object encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(self.input_dim, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Contrastive learning head
        self.contrastive_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
        )

        # Message generator
        self.message_generator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Output layers for each message position
        self.output_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size // 2),
                    nn.LayerNorm(config.hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(config.hidden_size // 2, config.vocabulary_size),
                )
                for _ in range(config.message_length)
            ]
        )

        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def encode_object(self, object_encoding: torch.Tensor) -> torch.Tensor:
        """Encode object to hidden representation."""
        return self.object_encoder(object_encoding)  # type: ignore[no-any-return]

    def get_contrastive_representation(
        self, object_encoding: torch.Tensor
    ) -> torch.Tensor:
        """Get contrastive representation for contrastive learning."""
        hidden = self.encode_object(object_encoding)
        return F.normalize(self.contrastive_head(hidden), dim=1)

    def forward(
        self, object_encoding: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Forward pass with contrastive learning."""
        # Encode object
        object_features = self.encode_object(object_encoding)

        # Generate message features
        message_features = self.message_generator(object_features)

        # Generate outputs for each position
        logits_list = []
        token_ids_list = []

        for i, output_layer in enumerate(self.output_layers):
            logits = output_layer(message_features)
            logits = logits / temperature

            # Gumbel-Softmax for differentiable sampling
            if self.training:
                token_ids = F.gumbel_softmax(logits, tau=temperature, hard=True)
            else:
                token_ids = F.one_hot(
                    torch.argmax(logits, dim=1), num_classes=self.config.vocabulary_size
                ).float()

            logits_list.append(logits)
            token_ids_list.append(token_ids)

        # Stack outputs
        logits_tensor = torch.stack(logits_list, dim=1)
        token_ids_tensor = torch.stack(token_ids_list, dim=1)

        return logits_tensor, token_ids_tensor, None, None


class ContrastiveListener(nn.Module):
    """Listener agent with contrastive learning for better representations.

    This listener uses contrastive learning to learn better message and object
    representations by contrasting positive and negative pairs.
    """

    def __init__(self, config: CommunicationConfig):
        super().__init__()
        self.config = config

        # Input dimensions
        self.message_dim = config.vocabulary_size
        self.object_dim = TOTAL_ATTRIBUTES

        # Calculate message input dimension
        message_input_dim = config.message_length * config.vocabulary_size
        if config.multimodal:
            message_input_dim += config.message_length * config.gesture_size

        # Message encoder
        self.message_encoder = nn.Sequential(
            nn.Linear(message_input_dim, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Object encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(self.object_dim, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Contrastive learning heads
        self.message_contrastive_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
        )

        self.object_contrastive_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(config.hidden_size)

        # Scorer
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1),
        )

        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def encode_message(self, message_tokens: torch.Tensor) -> torch.Tensor:
        """Encode message to hidden representation."""
        batch_size = message_tokens.size(0)
        message_flat = message_tokens.view(batch_size, -1)
        return self.message_encoder(message_flat)  # type: ignore[no-any-return]

    def encode_object(self, object_encoding: torch.Tensor) -> torch.Tensor:
        """Encode object to hidden representation."""
        return self.object_encoder(object_encoding)  # type: ignore[no-any-return]

    def get_contrastive_representations(
        self, message_tokens: torch.Tensor, object_encoding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get contrastive representations for contrastive learning."""
        message_hidden = self.encode_message(message_tokens)
        object_hidden = self.encode_object(object_encoding)

        message_contrastive = F.normalize(
            self.message_contrastive_head(message_hidden), dim=1
        )
        object_contrastive = F.normalize(
            self.object_contrastive_head(object_hidden), dim=1
        )

        return message_contrastive, object_contrastive

    def forward(
        self,
        message_tokens: torch.Tensor,
        candidate_objects: torch.Tensor,
        gesture_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with contrastive learning."""
        batch_size, num_candidates = candidate_objects.shape[:2]

        # Encode message
        if gesture_tokens is not None and self.config.multimodal:
            message_input = torch.cat([message_tokens, gesture_tokens], dim=-1)
        else:
            message_input = message_tokens

        message_features = self.encode_message(message_input)

        # Encode all candidate objects
        object_features = self.encode_object(
            candidate_objects.view(-1, self.object_dim)
        )
        object_features = object_features.view(batch_size, num_candidates, -1)

        # Apply cross-attention
        message_expanded = message_features.unsqueeze(1).repeat(1, num_candidates, 1)
        attended_features, _ = self.cross_attention(
            message_expanded, object_features, object_features
        )
        attended_features = self.attention_norm(attended_features + message_expanded)

        # Combine features and score
        combined_features = torch.cat([attended_features, object_features], dim=-1)
        scores = self.scorer(combined_features).squeeze(-1)

        # Apply softmax
        probabilities = F.softmax(scores, dim=1)

        return probabilities


def contrastive_loss(
    message_repr: torch.Tensor, object_repr: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Compute contrastive loss between message and object representations.

    Args:
        message_repr: Message representations (batch_size, repr_dim)
        object_repr: Object representations (batch_size, repr_dim)
        temperature: Temperature parameter for contrastive learning

    Returns:
        Contrastive loss value
    """
    batch_size = message_repr.size(0)

    # Normalize representations
    message_repr = F.normalize(message_repr, dim=1)
    object_repr = F.normalize(object_repr, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(message_repr, object_repr.T) / temperature

    # Create labels (diagonal elements are positive pairs)
    labels = torch.arange(batch_size, device=message_repr.device)

    # Compute cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss


def compute_contrastive_loss(
    speaker: ContrastiveSpeaker,
    listener: ContrastiveListener,
    target_objects: torch.Tensor,
    message_tokens: torch.Tensor,
    candidate_objects: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute contrastive loss for the speaker-listener pair.

    Args:
        speaker: Contrastive speaker model
        listener: Contrastive listener model
        target_objects: Target object encodings
        message_tokens: Generated message tokens
        candidate_objects: Candidate object encodings
        temperature: Temperature parameter

    Returns:
        Total contrastive loss
    """
    # Get contrastive representations
    speaker_object_repr = speaker.get_contrastive_representation(target_objects)
    listener_message_repr, listener_object_repr = (
        listener.get_contrastive_representations(message_tokens, target_objects)
    )

    # Compute contrastive losses
    speaker_loss = contrastive_loss(
        speaker_object_repr, listener_message_repr, temperature
    )
    listener_loss = contrastive_loss(
        listener_message_repr, listener_object_repr, temperature
    )

    # Total contrastive loss
    total_loss = speaker_loss + listener_loss

    return total_loss
