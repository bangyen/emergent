"""Neural network agents for referential games.

This module implements Speaker and Listener agents that participate in referential
games, learning to communicate about objects through discrete messages.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CommunicationConfig
from .world import TOTAL_ATTRIBUTES


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

        # Neural network layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )

        # Output layer for each message position
        self.output_layers = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.vocabulary_size)
                for _ in range(config.message_length)
            ]
        )

    def forward(
        self, object_encoding: torch.Tensor, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate message logits and sampled tokens for the given object.

        Args:
            object_encoding: Tensor of shape (batch_size, input_dim) containing encoded object.
            temperature: Temperature for Gumbel-Softmax sampling (default: 1.0).

        Returns:
            A tuple containing:
            - logits: Tensor of shape (batch_size, message_length, vocabulary_size) with raw logits
            - token_ids: Tensor of shape (batch_size, message_length) with sampled token indices
        """
        # Encode object
        hidden = self.encoder(object_encoding)  # (batch_size, hidden_size)

        # Generate logits for each message position
        logits = []
        token_ids = []

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

        # Stack outputs
        logits_tensor = torch.stack(
            logits, dim=1
        )  # (batch_size, message_length, vocabulary_size)
        token_ids_tensor = torch.stack(token_ids, dim=1)  # (batch_size, message_length)

        return logits_tensor, token_ids_tensor


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

        # Message encoder
        self.message_encoder = nn.Sequential(
            nn.Linear(
                config.message_length * config.vocabulary_size, config.hidden_size
            ),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )

        # Object encoder
        self.object_encoder = nn.Sequential(
            nn.Linear(self.object_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )

        # Scoring network
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(
        self, message_tokens: torch.Tensor, candidate_objects: torch.Tensor
    ) -> torch.Tensor:
        """Compute scores for each candidate object given the message.

        Args:
            message_tokens: Tensor of shape (batch_size, message_length) with token indices.
            candidate_objects: Tensor of shape (batch_size, num_candidates, object_dim) with encoded objects.

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

        # Encode message
        message_features = self.message_encoder(
            message_onehot
        )  # (batch_size, hidden_size)

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

        # Compute scores for each candidate
        scores = []
        for i in range(num_candidates):
            # Concatenate message and candidate features
            combined_features = torch.cat(
                [
                    message_features,  # (batch_size, hidden_size)
                    candidate_features[:, i, :],  # (batch_size, hidden_size)
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
