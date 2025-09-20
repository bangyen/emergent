"""Neural network agents for referential games.

This module implements Speaker and Listener agents that participate in referential
games, learning to communicate about objects through discrete messages.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CommunicationConfig
from ..data.world import TOTAL_ATTRIBUTES


class Speaker(nn.Module):
    """Speaker agent that generates messages about target objects.

    The Speaker agent takes an encoded object representation and generates
    discrete messages about it. It uses Gumbel-Softmax for differentiable
    sampling during training while maintaining discrete outputs.

    Args:
        config: Communication configuration containing vocabulary and architecture parameters.
    """

    def __init__(self, config: CommunicationConfig):
        super().__init__()
        self.config = config

        # Input encoding dimension (object attributes)
        self.input_dim = TOTAL_ATTRIBUTES

        # Enhanced neural network layers with residual connections and layer normalization
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Residual connection
        self.residual_proj = nn.Linear(self.input_dim, config.hidden_size)

        # Output layer for each message position with improved initialization
        self.output_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(config.hidden_size // 2, config.vocabulary_size),
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
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(config.hidden_size // 2, config.gesture_size),
                    )
                    for _ in range(config.message_length)
                ]
            )

        # Initialize weights with improved initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using improved initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use He initialization for ReLU networks
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(
        self, object_encoding: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Generate message logits and sampled tokens for the given object.

        Args:
            object_encoding: Tensor of shape (batch_size, input_dim) containing encoded object.
            temperature: Temperature for Gumbel-Softmax sampling (default: 1.0).

        Returns:
            A tuple containing:
            - logits: Tensor of shape (batch_size, message_length, vocabulary_size) with raw logits
            - token_ids: Tensor of shape (batch_size, message_length) with sampled token indices
            - gesture_logits: Optional tensor of shape (batch_size, message_length, gesture_size) with gesture logits
            - gesture_ids: Optional tensor of shape (batch_size, message_length) with sampled gesture indices
        """
        # Encode object with residual connection
        hidden = self.encoder(object_encoding)  # (batch_size, hidden_size)
        residual = self.residual_proj(object_encoding)
        hidden = hidden + residual

        # Generate logits for each message position
        logits = []
        token_ids = []
        gesture_logits = []
        gesture_ids = []

        for i in range(self.config.message_length):
            pos_logits = self.output_layers[i](hidden)  # (batch_size, vocabulary_size)
            logits.append(pos_logits)

            # Sample tokens using Gumbel-Softmax for differentiable training
            if self.training:
                # Add Gumbel noise for exploration during training
                gumbel_noise = -torch.log(
                    -torch.log(torch.rand_like(pos_logits) + 1e-20) + 1e-20
                )
                pos_logits_with_noise = pos_logits + gumbel_noise
            else:
                pos_logits_with_noise = pos_logits

            # Apply temperature scaling
            pos_logits_scaled = pos_logits_with_noise / temperature

            # Gumbel-Softmax sampling
            pos_probs = F.softmax(pos_logits_scaled, dim=-1)

            # Always use argmax for discrete tokens (straight-through in training)
            pos_tokens = torch.argmax(pos_probs, dim=-1)

            token_ids.append(pos_tokens)

            # Generate gestures if multimodal
            if self.config.multimodal:
                pos_gesture_logits = self.gesture_layers[i](
                    hidden
                )  # (batch_size, gesture_size)
                gesture_logits.append(pos_gesture_logits)

                # Sample gestures using the same logic
                if self.training:
                    gumbel_noise_gesture = -torch.log(
                        -torch.log(torch.rand_like(pos_gesture_logits) + 1e-20) + 1e-20
                    )
                    pos_gesture_logits_with_noise = (
                        pos_gesture_logits + gumbel_noise_gesture
                    )
                else:
                    pos_gesture_logits_with_noise = pos_gesture_logits

                pos_gesture_logits_scaled = pos_gesture_logits_with_noise / temperature
                pos_gesture_probs = F.softmax(pos_gesture_logits_scaled, dim=-1)
                pos_gestures = torch.argmax(pos_gesture_probs, dim=-1)

                gesture_ids.append(pos_gestures)

        # Stack outputs
        logits_tensor = torch.stack(
            logits, dim=1
        )  # (batch_size, message_length, vocabulary_size)
        token_ids_tensor = torch.stack(token_ids, dim=1)  # (batch_size, message_length)

        gesture_logits_tensor = None
        gesture_ids_tensor = None
        if self.config.multimodal:
            gesture_logits_tensor = torch.stack(
                gesture_logits, dim=1
            )  # (batch_size, message_length, gesture_size)
            gesture_ids_tensor = torch.stack(
                gesture_ids, dim=1
            )  # (batch_size, message_length)

        return (
            logits_tensor,
            token_ids_tensor,
            gesture_logits_tensor,
            gesture_ids_tensor,
        )


class Listener(nn.Module):
    """Listener agent that interprets messages to identify target objects.

    The Listener agent receives a message and a set of candidate objects,
    then computes scores for each candidate to determine which one the
    message refers to.

    Args:
        config: Communication configuration containing vocabulary and architecture parameters.
    """

    def __init__(self, config: CommunicationConfig):
        super().__init__()
        self.config = config

        # Input dimensions
        self.message_dim = config.vocabulary_size  # One-hot encoded message
        self.object_dim = TOTAL_ATTRIBUTES  # Encoded object

        # Calculate input dimension for message encoder
        message_input_dim = config.message_length * config.vocabulary_size
        if config.multimodal:
            message_input_dim += config.message_length * config.gesture_size

        # Transformer-based message encoder
        self.message_embedding = nn.Linear(message_input_dim, config.hidden_size)
        self.message_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            ),
            num_layers=3,
        )
        self.message_norm = nn.LayerNorm(config.hidden_size)

        # Transformer-based object encoder
        self.object_embedding = nn.Linear(self.object_dim, config.hidden_size)
        self.object_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            ),
            num_layers=3,
        )
        self.object_norm = nn.LayerNorm(config.hidden_size)

        # Residual projection for object encoder
        self.object_residual_proj = nn.Linear(self.object_dim, config.hidden_size)

        # Enhanced scoring network with attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )

        # Cross-attention for better message-object alignment
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )

        # Layer normalization for attention outputs
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.cross_attention_norm = nn.LayerNorm(config.hidden_size)

        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1),
        )

        # Initialize weights with improved initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using improved initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use He initialization for ReLU networks
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize attention weights
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)

    def forward(
        self,
        message_tokens: torch.Tensor,
        candidate_objects: torch.Tensor,
        gesture_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scores for each candidate object given the message.

        Args:
            message_tokens: Tensor of shape (batch_size, message_length) with token indices.
            candidate_objects: Tensor of shape (batch_size, num_candidates, object_dim) with encoded objects.
            gesture_tokens: Optional tensor of shape (batch_size, message_length) with gesture indices.

        Returns:
            Tensor of shape (batch_size, num_candidates) with probabilities over candidates.
        """
        batch_size, num_candidates = candidate_objects.size(0), candidate_objects.size(
            1
        )

        # One-hot encode message tokens
        message_onehot = F.one_hot(
            message_tokens, num_classes=self.config.vocabulary_size
        )
        message_onehot = message_onehot.view(
            batch_size, -1
        ).float()  # Flatten message positions and convert to float

        # One-hot encode gesture tokens if multimodal
        if self.config.multimodal and gesture_tokens is not None:
            gesture_onehot = F.one_hot(
                gesture_tokens, num_classes=self.config.gesture_size
            )
            gesture_onehot = gesture_onehot.view(
                batch_size, -1
            ).float()  # Flatten gesture positions and convert to float

            # Concatenate message and gesture encodings
            multimodal_input = torch.cat([message_onehot, gesture_onehot], dim=-1)
        else:
            multimodal_input = message_onehot

        # Encode message using transformer
        message_embedded = self.message_embedding(
            multimodal_input
        )  # (batch_size, hidden_size)
        message_features = self.message_transformer(
            message_embedded.unsqueeze(1)
        )  # (batch_size, 1, hidden_size)
        message_features = self.message_norm(
            message_features.squeeze(1)
        )  # (batch_size, hidden_size)

        # Encode all candidate objects using transformer
        candidate_flat = candidate_objects.view(
            -1, self.object_dim
        )  # (batch_size * num_candidates, object_dim)
        candidate_embedded = self.object_embedding(
            candidate_flat
        )  # (batch_size * num_candidates, hidden_size)
        candidate_features = self.object_transformer(
            candidate_embedded.unsqueeze(1)
        )  # (batch_size * num_candidates, 1, hidden_size)
        candidate_features = self.object_norm(
            candidate_features.squeeze(1)
        )  # (batch_size * num_candidates, hidden_size)
        candidate_residual = self.object_residual_proj(candidate_flat)
        candidate_features = candidate_features + candidate_residual
        candidate_features = candidate_features.view(
            batch_size, num_candidates, -1
        )  # (batch_size, num_candidates, hidden_size)

        # Apply cross-attention mechanism between message and candidates
        message_features_expanded = message_features.unsqueeze(1).expand(
            -1, num_candidates, -1
        )  # (batch_size, num_candidates, hidden_size)

        # First, apply self-attention to candidate features
        candidate_self_attended, _ = self.attention(
            query=candidate_features,
            key=candidate_features,
            value=candidate_features,
        )  # (batch_size, num_candidates, hidden_size)
        candidate_self_attended = self.attention_norm(
            candidate_self_attended + candidate_features
        )

        # Then apply cross-attention between message and candidates
        attended_features, _ = self.cross_attention(
            query=message_features_expanded,
            key=candidate_self_attended,
            value=candidate_self_attended,
        )  # (batch_size, num_candidates, hidden_size)
        attended_features = self.cross_attention_norm(
            attended_features + message_features_expanded
        )

        # Compute scores for each candidate using enhanced features
        scores = []
        for i in range(num_candidates):
            # Concatenate message and attended candidate features
            combined_features = torch.cat(
                [
                    message_features,  # (batch_size, hidden_size)
                    attended_features[:, i, :],  # (batch_size, hidden_size)
                ],
                dim=-1,
            )  # (batch_size, hidden_size * 2)

            # Compute score
            score = self.scorer(combined_features)  # (batch_size, 1)
            scores.append(score.squeeze(-1))  # (batch_size,)

        scores_tensor = torch.stack(scores, dim=1)  # (batch_size, num_candidates)

        # Convert scores to probabilities via softmax
        probabilities = F.softmax(scores_tensor, dim=-1)

        return probabilities


class PragmaticListener(nn.Module):
    """Pragmatic Listener agent that uses RSA-style reasoning for distractor scenes.

    The PragmaticListener implements Rational Speech Act (RSA) reasoning to handle
    ambiguous messages in distractor-heavy scenes. It considers speaker intent by
    reasoning about what the speaker would likely say given different candidate objects.

    Args:
        config: Communication configuration containing vocabulary and pragmatic parameters.
        literal_listener: Pre-trained literal listener for RSA computation.
        speaker: Pre-trained speaker for RSA computation.
    """

    def __init__(
        self, config: CommunicationConfig, literal_listener: Listener, speaker: Speaker
    ):
        super().__init__()
        self.config = config
        self.literal_listener = literal_listener
        self.speaker = speaker
        self.temperature = 1.0  # Temperature for RSA computation

    def forward(
        self,
        message_tokens: torch.Tensor,
        candidate_objects: torch.Tensor,
        gesture_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute pragmatic scores for each candidate object given the message.

        This implements RSA-style pragmatic reasoning:
        1. Compute literal listener probabilities P_L(obj | message)
        2. Compute speaker probabilities P_S(message | obj) for each candidate
        3. Compute pragmatic listener probabilities using RSA formula

        Args:
            message_tokens: Tensor of shape (batch_size, message_length) with token indices.
            candidate_objects: Tensor of shape (batch_size, num_candidates, object_dim) with encoded objects.
            gesture_tokens: Optional tensor of shape (batch_size, message_length) with gesture indices.

        Returns:
            Tensor of shape (batch_size, num_candidates) with pragmatic probabilities over candidates.
        """
        num_candidates = candidate_objects.size(1)

        # Step 1: Compute literal listener probabilities
        literal_probs = self.literal_listener(
            message_tokens, candidate_objects, gesture_tokens
        )

        # Step 2: Compute speaker probabilities for each candidate
        speaker_probs = []

        for i in range(num_candidates):
            # Get the i-th candidate object for each batch
            candidate_obj = candidate_objects[:, i, :]  # (batch_size, object_dim)

            # Generate speaker logits for this candidate
            if self.config.multimodal:
                logits, _, gesture_logits, _ = self.speaker(
                    candidate_obj, self.temperature
                )
            else:
                logits, _, _, _ = self.speaker(candidate_obj, self.temperature)
                gesture_logits = None

            # Compute probability of the observed message given this candidate
            message_probs = F.softmax(
                logits, dim=-1
            )  # (batch_size, message_length, vocab_size)

            # Get probability of observed tokens
            token_probs = torch.gather(
                message_probs, dim=-1, index=message_tokens.unsqueeze(-1)
            ).squeeze(
                -1
            )  # (batch_size, message_length)

            # Product over message positions
            candidate_message_prob = torch.prod(token_probs, dim=-1)  # (batch_size,)

            # Handle gestures if multimodal
            if (
                self.config.multimodal
                and gesture_tokens is not None
                and gesture_logits is not None
            ):
                gesture_probs = F.softmax(
                    gesture_logits, dim=-1
                )  # (batch_size, message_length, gesture_size)
                gesture_token_probs = torch.gather(
                    gesture_probs, dim=-1, index=gesture_tokens.unsqueeze(-1)
                ).squeeze(
                    -1
                )  # (batch_size, message_length)

                candidate_gesture_prob = torch.prod(
                    gesture_token_probs, dim=-1
                )  # (batch_size,)
                candidate_message_prob = candidate_message_prob * candidate_gesture_prob

            speaker_probs.append(candidate_message_prob)

        speaker_probs_tensor = torch.stack(
            speaker_probs, dim=1
        )  # (batch_size, num_candidates)

        # Step 3: RSA pragmatic listener computation
        # P_pragmatic(obj | message) ∝ P_literal(obj | message) * P_speaker(message | obj)
        pragmatic_scores = literal_probs * speaker_probs_tensor

        # Normalize to get probabilities
        pragmatic_probs: torch.Tensor = pragmatic_scores / pragmatic_scores.sum(
            dim=-1, keepdim=True
        )

        return pragmatic_probs


class SpeakerSeq(nn.Module):
    """Autoregressive Speaker agent that generates sequences of tokens.

    The SpeakerSeq agent uses a GRU to generate messages autoregressively,
    conditioning each token on the target object encoding and previously generated tokens.
    This enables more expressive communication for longer messages.

    Args:
        config: Communication configuration containing vocabulary and architecture parameters.
    """

    def __init__(self, config: CommunicationConfig):
        super().__init__()
        self.config = config

        # Input encoding dimension (object attributes)
        self.input_dim = TOTAL_ATTRIBUTES
        self.vocab_size = config.vocabulary_size
        self.message_length = config.message_length
        self.hidden_size = config.hidden_size

        # Object encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(self.input_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )

        # GRU for autoregressive generation
        self.gru = nn.GRU(
            input_size=config.hidden_size
            + config.vocabulary_size,  # object + token embedding
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Output projection to vocabulary
        self.output_proj = nn.Linear(config.hidden_size, config.vocabulary_size)

        # Token embedding
        self.token_embedding = nn.Embedding(
            config.vocabulary_size, config.vocabulary_size
        )

    def forward(
        self,
        object_encoding: torch.Tensor,
        temperature: float = 1.0,
        teacher_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate message logits and sampled tokens autoregressively.

        Args:
            object_encoding: Tensor of shape (batch_size, input_dim) containing encoded object.
            temperature: Temperature for sampling (default: 1.0).
            teacher_tokens: Optional tensor of shape (batch_size, message_length) for teacher forcing.

        Returns:
            A tuple containing:
            - logits: Tensor of shape (batch_size, message_length, vocabulary_size) with raw logits
            - token_ids: Tensor of shape (batch_size, message_length) with sampled token indices
        """
        batch_size = object_encoding.size(0)
        device = object_encoding.device

        # Encode object
        object_features = self.object_encoder(
            object_encoding
        )  # (batch_size, hidden_size)

        # Initialize hidden state
        hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)

        # Initialize outputs
        logits = []
        token_ids = []

        # Start with zero token (or special start token)
        current_token = torch.zeros(batch_size, dtype=torch.long, device=device)

        for step in range(self.message_length):
            # Embed current token
            token_emb = self.token_embedding(current_token)  # (batch_size, vocab_size)

            # Concatenate object features and token embedding
            gru_input = torch.cat(
                [object_features, token_emb], dim=-1
            )  # (batch_size, hidden_size + vocab_size)
            gru_input = gru_input.unsqueeze(
                1
            )  # (batch_size, 1, hidden_size + vocab_size)

            # GRU forward pass
            gru_output, hidden = self.gru(
                gru_input, hidden
            )  # gru_output: (batch_size, 1, hidden_size)
            gru_output = gru_output.squeeze(1)  # (batch_size, hidden_size)

            # Project to vocabulary logits
            step_logits = self.output_proj(gru_output)  # (batch_size, vocab_size)
            logits.append(step_logits)

            # Sample next token
            if teacher_tokens is not None and self.training:
                # Teacher forcing during training
                current_token = teacher_tokens[:, step]
            else:
                # Autoregressive sampling
                if self.training:
                    # Add Gumbel noise for exploration during training
                    gumbel_noise = -torch.log(
                        -torch.log(torch.rand_like(step_logits) + 1e-20) + 1e-20
                    )
                    step_logits_with_noise = step_logits + gumbel_noise
                else:
                    step_logits_with_noise = step_logits

                # Apply temperature scaling
                step_logits_scaled = step_logits_with_noise / temperature

                # Sample token
                step_probs = F.softmax(step_logits_scaled, dim=-1)
                current_token = torch.argmax(step_probs, dim=-1)

            token_ids.append(current_token)

        # Stack outputs
        logits_tensor = torch.stack(
            logits, dim=1
        )  # (batch_size, message_length, vocabulary_size)
        token_ids_tensor = torch.stack(token_ids, dim=1)  # (batch_size, message_length)

        return logits_tensor, token_ids_tensor


class ListenerSeq(nn.Module):
    """Sequence-aware Listener agent that processes token sequences.

    The ListenerSeq agent uses a GRU to encode message sequences and then
    scores candidate objects using bilinear or MLP scoring mechanisms.
    This enables better understanding of sequential message structure.

    Args:
        config: Communication configuration containing vocabulary and architecture parameters.
    """

    def __init__(self, config: CommunicationConfig):
        super().__init__()
        self.config = config

        # Input dimensions
        self.vocab_size = config.vocabulary_size
        self.object_dim = TOTAL_ATTRIBUTES
        self.hidden_size = config.hidden_size

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.hidden_size)

        # GRU encoder for message sequences
        self.message_encoder = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Object encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(self.object_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )

        # Bilinear scoring mechanism
        self.bilinear_scorer = nn.Bilinear(config.hidden_size, config.hidden_size, 1)

    def forward(
        self, message_tokens: torch.Tensor, candidate_objects: torch.Tensor
    ) -> torch.Tensor:
        """Compute scores for each candidate object given the message sequence.

        Args:
            message_tokens: Tensor of shape (batch_size, message_length) with token indices.
            candidate_objects: Tensor of shape (batch_size, num_candidates, object_dim) with encoded objects.

        Returns:
            Tensor of shape (batch_size, num_candidates) with probabilities over candidates.
        """
        batch_size, num_candidates = candidate_objects.size(0), candidate_objects.size(
            1
        )

        # Embed message tokens
        message_embeddings = self.token_embedding(
            message_tokens
        )  # (batch_size, message_length, hidden_size)

        # Encode message sequence with GRU
        message_output, message_hidden = self.message_encoder(message_embeddings)
        # Use the last hidden state as the message representation
        message_features = message_hidden.squeeze(0)  # (batch_size, hidden_size)

        # Encode all candidate objects
        candidate_flat = candidate_objects.view(
            -1, self.object_dim
        )  # (batch_size * num_candidates, object_dim)
        candidate_features = self.object_encoder(
            candidate_flat
        )  # (batch_size * num_candidates, hidden_size)
        candidate_features = candidate_features.view(
            batch_size, num_candidates, -1
        )  # (batch_size, num_candidates, hidden_size)

        # Compute bilinear scores for each candidate
        scores = []
        for i in range(num_candidates):
            # Bilinear scoring
            score = self.bilinear_scorer(
                message_features,  # (batch_size, hidden_size)
                candidate_features[:, i, :],  # (batch_size, hidden_size)
            )  # (batch_size, 1)
            scores.append(score.squeeze(-1))  # (batch_size,)

        scores_tensor = torch.stack(scores, dim=1)  # (batch_size, num_candidates)

        # Convert scores to probabilities via softmax
        probabilities = F.softmax(scores_tensor, dim=-1)

        return probabilities
