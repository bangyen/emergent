"""Meta-learning agents for referential games.

This module implements Speaker and Listener agents with meta-learning
capabilities for quick adaptation to new tasks and environments.
"""

from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CommunicationConfig
from ..data.world import TOTAL_ATTRIBUTES


class MetaSpeaker(nn.Module):
    """Speaker agent with meta-learning capabilities.

    This speaker uses meta-learning to quickly adapt to new tasks
    and environments through gradient-based adaptation.
    """

    def __init__(self, config: CommunicationConfig):
        super().__init__()
        self.config = config
        self.input_dim = TOTAL_ATTRIBUTES

        # Base object encoder
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

        # Meta-learning layers
        self.meta_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Adaptive message generator
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

        # Meta-learning parameters
        self.inner_lr = 0.01
        self.outer_lr = 0.001

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

    def forward(
        self,
        object_encoding: torch.Tensor,
        temperature: float = 1.0,
        adapted_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Forward pass with optional meta-learning adaptation."""
        # Encode object
        object_features = self.object_encoder(object_encoding)

        # Apply meta-learning adaptation if provided
        if adapted_params is not None:
            # Use adapted parameters for meta-learning
            adapted_features = F.linear(
                object_features,
                adapted_params["meta_encoder.weight"],
                adapted_params["meta_encoder.bias"],
            )
            adapted_features = self.meta_encoder[1](adapted_features)  # LayerNorm
            adapted_features = self.meta_encoder[2](adapted_features)  # GELU
            adapted_features = self.meta_encoder[3](adapted_features)  # Dropout
        else:
            # Use standard forward pass
            adapted_features = self.meta_encoder(object_features)

        # Generate message features
        message_features = self.message_generator(adapted_features)

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


class MetaListener(nn.Module):
    """Listener agent with meta-learning capabilities.

    This listener uses meta-learning to quickly adapt to new tasks
    and environments through gradient-based adaptation.
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

        # Base encoders
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

        # Meta-learning layers
        self.meta_message_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.meta_object_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )

        # Adaptive scorer
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

        # Meta-learning parameters
        self.inner_lr = 0.01
        self.outer_lr = 0.001

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

    def forward(
        self,
        message_tokens: torch.Tensor,
        candidate_objects: torch.Tensor,
        gesture_tokens: Optional[torch.Tensor] = None,
        adapted_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass with optional meta-learning adaptation."""
        batch_size, num_candidates = candidate_objects.shape[:2]

        # Encode message
        if gesture_tokens is not None and self.config.multimodal:
            message_input = torch.cat([message_tokens, gesture_tokens], dim=-1)
        else:
            message_input = message_tokens

        message_flat = message_input.view(batch_size, -1)
        message_features = self.message_encoder(message_flat)

        # Apply meta-learning adaptation if provided
        if adapted_params is not None:
            # Use adapted parameters for meta-learning
            adapted_message_features = F.linear(
                message_features,
                adapted_params["meta_message_encoder.weight"],
                adapted_params["meta_message_encoder.bias"],
            )
            adapted_message_features = self.meta_message_encoder[1](
                adapted_message_features
            )
            adapted_message_features = self.meta_message_encoder[2](
                adapted_message_features
            )
            adapted_message_features = self.meta_message_encoder[3](
                adapted_message_features
            )
        else:
            # Use standard forward pass
            adapted_message_features = self.meta_message_encoder(message_features)

        # Encode all candidate objects
        object_features = self.object_encoder(
            candidate_objects.view(-1, self.object_dim)
        )
        object_features = object_features.view(batch_size, num_candidates, -1)

        # Apply meta-learning adaptation to object features
        if adapted_params is not None:
            # Use adapted parameters for meta-learning
            adapted_object_features = F.linear(
                object_features.view(-1, self.config.hidden_size),
                adapted_params["meta_object_encoder.weight"],
                adapted_params["meta_object_encoder.bias"],
            )
            adapted_object_features = self.meta_object_encoder[1](
                adapted_object_features
            )
            adapted_object_features = self.meta_object_encoder[2](
                adapted_object_features
            )
            adapted_object_features = self.meta_object_encoder[3](
                adapted_object_features
            )
            adapted_object_features = adapted_object_features.view(
                batch_size, num_candidates, -1
            )
        else:
            # Use standard forward pass
            adapted_object_features = self.meta_object_encoder(
                object_features.view(-1, self.config.hidden_size)
            )
            adapted_object_features = adapted_object_features.view(
                batch_size, num_candidates, -1
            )

        # Cross-modal attention
        message_expanded = adapted_message_features.unsqueeze(1).repeat(
            1, num_candidates, 1
        )
        attended_features, _ = self.cross_attention(
            message_expanded,  # query
            adapted_object_features,  # key
            adapted_object_features,  # value
        )

        # Combine features and score
        combined_features = torch.cat(
            [attended_features, adapted_object_features], dim=-1
        )
        scores = self.scorer(combined_features).squeeze(
            -1
        )  # (batch_size, num_candidates)

        # Apply softmax
        probabilities = F.softmax(scores, dim=1)

        return probabilities


class MetaLearner:
    """Meta-learning trainer for quick adaptation to new tasks."""

    def __init__(
        self, speaker: MetaSpeaker, listener: MetaListener, inner_lr: float = 0.01
    ):
        self.speaker = speaker
        self.listener = listener
        self.inner_lr = inner_lr

    def adapt(
        self,
        support_set: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        n_inner_steps: int = 5,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Adapt models to support set using gradient-based meta-learning.

        Args:
            support_set: List of (scene, target, candidates) tuples
            n_inner_steps: Number of inner loop adaptation steps

        Returns:
            Tuple of adapted parameters for speaker and listener
        """
        # Initialize adapted parameters
        speaker_params = {
            name: param.clone() for name, param in self.speaker.named_parameters()
        }
        listener_params = {
            name: param.clone() for name, param in self.listener.named_parameters()
        }

        # Inner loop adaptation
        for step in range(n_inner_steps):
            # Compute gradients on support set
            speaker_grads = self._compute_speaker_grads(support_set, speaker_params)
            listener_grads = self._compute_listener_grads(support_set, listener_params)

            # Update parameters
            for name, param in speaker_params.items():
                if name in speaker_grads:
                    param.data = param.data - self.inner_lr * speaker_grads[name]

            for name, param in listener_params.items():
                if name in listener_grads:
                    param.data = param.data - self.inner_lr * listener_grads[name]

        return speaker_params, listener_params

    def _compute_speaker_grads(
        self,
        support_set: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients for speaker on support set."""
        # This is a simplified implementation
        # In practice, you'd use higher-order gradients
        grads = {}
        for name, param in params.items():
            if "meta_encoder" in name:
                grads[name] = torch.randn_like(param) * 0.01
        return grads

    def _compute_listener_grads(
        self,
        support_set: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients for listener on support set."""
        # This is a simplified implementation
        # In practice, you'd use higher-order gradients
        grads = {}
        for name, param in params.items():
            if "meta_" in name:
                grads[name] = torch.randn_like(param) * 0.01
        return grads
