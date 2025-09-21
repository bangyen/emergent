"""Improved neural network agents for referential games.

This module implements enhanced Speaker and Listener agents with advanced
architectures and training techniques to improve model performance.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CommunicationConfig
from ..data.world import TOTAL_ATTRIBUTES


class ImprovedSpeaker(nn.Module):
    """Enhanced Speaker agent with advanced architecture and training techniques.

    This improved speaker uses:
    - Transformer-based architecture for better sequence modeling
    - Attention mechanisms for better object understanding
    - Advanced regularization techniques
    - Better initialization and optimization
    """

    def __init__(self, config: CommunicationConfig):
        super().__init__()
        self.config = config
        self.input_dim = TOTAL_ATTRIBUTES

        # Object encoder with attention mechanism
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

        # Self-attention for object representation
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(config.hidden_size)

        # Transformer-based message generator
        self.message_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=2,
        )

        # Positional encoding for message positions
        self.positional_encoding = nn.Parameter(
            torch.randn(config.message_length, config.hidden_size) * 0.1
        )

        # Output layers with improved architecture
        self.output_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.LayerNorm(config.hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(config.hidden_size, config.vocabulary_size),
                )
                for _ in range(config.message_length)
            ]
        )

        # Gesture output layers for multimodal communication
        if config.multimodal:
            self.gesture_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(config.hidden_size, config.hidden_size // 2),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(config.hidden_size // 2, config.gesture_size),
                    )
                    for _ in range(config.message_length)
                ]
            )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using advanced initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize attention weights
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)

    def forward(
        self, object_encoding: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Forward pass with improved architecture."""
        # Encode object with attention
        object_features = self.object_encoder(object_encoding)

        # Apply self-attention to object features
        object_features_expanded = object_features.unsqueeze(1)  # (batch, 1, hidden)
        attended_features, _ = self.self_attention(
            object_features_expanded, object_features_expanded, object_features_expanded
        )
        object_features = self.attention_norm(
            attended_features.squeeze(1) + object_features
        )

        # Create message sequence with positional encoding
        message_features = object_features.unsqueeze(1).repeat(
            1, self.config.message_length, 1
        )
        message_features = message_features + self.positional_encoding.unsqueeze(0)

        # Apply transformer
        message_features = self.message_transformer(message_features)

        # Generate outputs for each position
        logits_list = []
        token_ids_list = []

        for i, output_layer in enumerate(self.output_layers):
            logits = output_layer(message_features[:, i, :])
            logits = logits / temperature  # Temperature scaling

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
        logits_tensor = torch.stack(
            logits_list, dim=1
        )  # (batch, message_length, vocab_size)
        token_ids_tensor = torch.stack(
            token_ids_list, dim=1
        )  # (batch, message_length, vocab_size)

        # Handle gesture outputs if multimodal
        gesture_logits_tensor = None
        gesture_ids_tensor = None

        if self.config.multimodal:
            gesture_logits_list = []
            gesture_ids_list = []

            for i, gesture_layer in enumerate(self.gesture_layers):
                gesture_logits = gesture_layer(message_features[:, i, :])
                gesture_logits = gesture_logits / temperature

                if self.training:
                    gesture_ids = F.gumbel_softmax(
                        gesture_logits, tau=temperature, hard=True
                    )
                else:
                    gesture_ids = F.one_hot(
                        torch.argmax(gesture_logits, dim=1),
                        num_classes=self.config.gesture_size,
                    ).float()

                gesture_logits_list.append(gesture_logits)
                gesture_ids_list.append(gesture_ids)

            gesture_logits_tensor = torch.stack(gesture_logits_list, dim=1)
            gesture_ids_tensor = torch.stack(gesture_ids_list, dim=1)

        return (
            logits_tensor,
            token_ids_tensor,
            gesture_logits_tensor,
            gesture_ids_tensor,
        )


class ImprovedListener(nn.Module):
    """Enhanced Listener agent with advanced architecture and training techniques.

    This improved listener uses:
    - Cross-attention between messages and objects
    - Transformer-based architecture
    - Advanced scoring mechanisms
    - Better regularization and optimization
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

        # Message encoder with transformer
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

        # Cross-attention between message and objects
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(config.hidden_size)

        # Advanced scorer with multiple heads
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
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

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using advanced initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.MultiheadAttention):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)

    def forward(
        self,
        message_tokens: torch.Tensor,
        candidate_objects: torch.Tensor,
        gesture_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with improved architecture."""
        batch_size, num_candidates = candidate_objects.shape[:2]

        # Encode message
        if gesture_tokens is not None and self.config.multimodal:
            # Concatenate message and gesture tokens
            message_input = torch.cat([message_tokens, gesture_tokens], dim=-1)
        else:
            message_input = message_tokens

        # Flatten message for encoding
        message_flat = message_input.view(batch_size, -1)
        message_features = self.message_encoder(message_flat)

        # Encode all candidate objects
        object_features = self.object_encoder(
            candidate_objects.view(-1, self.object_dim)
        )
        object_features = object_features.view(batch_size, num_candidates, -1)

        # Apply cross-attention
        message_expanded = message_features.unsqueeze(1).repeat(1, num_candidates, 1)
        attended_features, _ = self.cross_attention(
            message_expanded, object_features, object_features
        )
        attended_features = self.cross_attention_norm(
            attended_features + message_expanded
        )

        # Combine message and object features
        combined_features = torch.cat([attended_features, object_features], dim=-1)

        # Score each candidate
        scores = self.scorer(combined_features).squeeze(
            -1
        )  # (batch_size, num_candidates)

        # Apply softmax to get probabilities
        probabilities = F.softmax(scores, dim=1)

        return probabilities


class ImprovedSpeakerSeq(nn.Module):
    """Enhanced sequence-aware Speaker with advanced architecture."""

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

        # Transformer-based sequence generator
        self.sequence_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=2,
        )

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.hidden_size)
        self.positional_encoding = nn.Parameter(
            torch.randn(config.message_length, config.hidden_size) * 0.1
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.vocabulary_size),
        )

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
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(
        self,
        object_encoding: torch.Tensor,
        temperature: float = 1.0,
        teacher_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with improved sequence generation."""
        batch_size = object_encoding.size(0)

        # Encode object
        object_features = self.object_encoder(object_encoding)

        # Create target sequence
        if teacher_tokens is not None and self.training:
            # Teacher forcing during training
            target_sequence = teacher_tokens
        else:
            # Generate sequence autoregressively
            target_sequence = torch.zeros(
                batch_size,
                self.config.message_length,
                dtype=torch.long,
                device=object_encoding.device,
            )

        # Embed tokens and add positional encoding
        target_embedded = self.token_embedding(target_sequence)
        target_embedded = target_embedded + self.positional_encoding.unsqueeze(0)

        # Use object features as memory for transformer
        memory = object_features.unsqueeze(1)  # (batch, 1, hidden)

        # Apply transformer decoder
        output_features = self.sequence_transformer(target_embedded, memory)

        # Generate logits
        logits = self.output_layer(output_features)
        logits = logits / temperature

        # Generate tokens
        if self.training:
            token_ids = F.gumbel_softmax(logits, tau=temperature, hard=True)
        else:
            token_ids = F.one_hot(
                torch.argmax(logits, dim=-1), num_classes=self.config.vocabulary_size
            ).float()

        return logits, token_ids


class ImprovedListenerSeq(nn.Module):
    """Enhanced sequence-aware Listener with advanced architecture."""

    def __init__(self, config: CommunicationConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocabulary_size
        self.object_dim = TOTAL_ATTRIBUTES
        self.hidden_size = config.hidden_size

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.hidden_size)

        # Transformer-based message encoder
        self.message_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=2,
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

        # Cross-attention for message-object interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(config.hidden_size)

        # Advanced scorer
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 1),
        )

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
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(
        self, message_tokens: torch.Tensor, candidate_objects: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with improved sequence understanding."""
        batch_size, num_candidates = candidate_objects.shape[:2]

        # Embed message tokens (convert one-hot to indices if needed)
        if message_tokens.dtype == torch.float:
            # Convert one-hot to indices
            message_indices = torch.argmax(message_tokens, dim=-1)
        else:
            message_indices = message_tokens
        message_embedded = self.token_embedding(message_indices)

        # Encode message sequence
        message_features = self.message_encoder(message_embedded)
        message_pooled = message_features.mean(dim=1)  # Pool over sequence length

        # Encode candidate objects
        object_features = self.object_encoder(
            candidate_objects.view(-1, self.object_dim)
        )
        object_features = object_features.view(batch_size, num_candidates, -1)

        # Apply cross-attention
        message_expanded = message_pooled.unsqueeze(1).repeat(1, num_candidates, 1)
        attended_features, _ = self.cross_attention(
            message_expanded, object_features, object_features
        )
        attended_features = self.cross_attention_norm(
            attended_features + message_expanded
        )

        # Combine features and score
        combined_features = torch.cat([attended_features, object_features], dim=-1)
        scores = self.scorer(combined_features).squeeze(-1)

        # Apply softmax
        probabilities = F.softmax(scores, dim=1)

        return probabilities
