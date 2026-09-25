"""Neural network agents for referential games.

This module implements Speaker and Listener agents that participate in referential
games, learning to communicate about objects through discrete messages.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CommunicationConfig


def _sample_tokens(
    logits: torch.Tensor, temperature: float, stochastic: bool
) -> torch.Tensor:
    """Pick one token per row: a Gumbel-max sample if stochastic, else argmax.

    Gumbel-max draws an exact sample from softmax(logits / temperature), so the
    log-probability of the returned tokens is well defined for REINFORCE.
    """
    scaled = logits / temperature
    if stochastic:
        uniform = torch.rand_like(scaled).clamp_(1e-20, 1.0)
        scaled = scaled - torch.log(-torch.log(uniform))
    return torch.argmax(scaled, dim=-1)


@dataclass
class SpeakerOutput:
    """Structured output for Speaker agents.

    Attributes:
        logits: Tensor of shape (batch_size, message_length, vocabulary_size) with raw logits.
        tokens: Tensor of shape (batch_size, message_length) with sampled token indices.
    """

    logits: torch.Tensor
    tokens: torch.Tensor


@dataclass
class ListenerOutput:
    """Structured output for Listener agents.

    Attributes:
        probs: Tensor of shape (batch_size, num_candidates) with probabilities over candidates.
        preds: Tensor of shape (batch_size,) with the index of the predicted candidate.
    """

    probs: torch.Tensor
    preds: torch.Tensor


class Speaker(nn.Module):
    """Speaker agent that generates messages about target objects.

    The Speaker agent takes an encoded object representation and generates
    discrete messages about it. Tokens are sampled with the Gumbel-max trick
    during training (an exact sample from softmax(logits)) and chosen greedily
    in eval mode.

    Args:
        config: Communication configuration containing vocabulary and architecture parameters.
    """

    def __init__(self, config: CommunicationConfig):
        super().__init__()
        self.config = config

        # Input encoding dimension (object attributes)
        self.input_dim = config.object_dim

        # Enhanced neural network layers with residual connections and layer normalization
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Residual connection for the encoder
        self.residual_projection = (
            nn.Linear(self.input_dim, config.hidden_size)
            if self.input_dim != config.hidden_size
            else nn.Identity()
        )

        # Output layer for each message position with improved initialization
        self.output_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.LayerNorm(config.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_size, config.hidden_size // 2),
                    nn.LayerNorm(config.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_size // 2, config.vocabulary_size),
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
    ) -> SpeakerOutput:
        """Generate message logits and sampled tokens for the given object.

        Args:
            object_encoding: Tensor of shape (batch_size, input_dim) containing encoded object.
            temperature: Sampling temperature (default: 1.0).

        Returns:
            SpeakerOutput containing logits and tokens.
        """
        # Encode object with residual connection
        hidden = self.encoder(object_encoding)  # (batch_size, hidden_size)
        residual = self.residual_projection(object_encoding)
        hidden = hidden + residual

        # Generate logits for each message position
        logits = []
        token_ids = []

        for i in range(self.config.message_length):
            pos_logits = self.output_layers[i](hidden)  # (batch_size, vocabulary_size)
            logits.append(pos_logits)
            token_ids.append(_sample_tokens(pos_logits, temperature, self.training))

        return SpeakerOutput(
            logits=torch.stack(logits, dim=1),  # (batch_size, message_length, vocab)
            tokens=torch.stack(token_ids, dim=1),  # (batch_size, message_length)
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
        self.object_dim = config.object_dim  # Encoded object

        # Calculate input dimension for message encoder
        message_input_dim = config.message_length * config.vocabulary_size

        # Improved message encoder with better architecture
        self.message_encoder = nn.Sequential(
            nn.Linear(message_input_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Residual connections for encoders
        self.message_residual_proj = (
            nn.Linear(message_input_dim, config.hidden_size)
            if message_input_dim != config.hidden_size
            else nn.Identity()
        )
        self.object_residual_proj = (
            nn.Linear(self.object_dim, config.hidden_size)
            if self.object_dim != config.hidden_size
            else nn.Identity()
        )

        # Improved object encoder with better architecture
        self.object_encoder = nn.Sequential(
            nn.Linear(self.object_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Deeper scorer for better decision making
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
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

    def forward(
        self,
        tokens: torch.Tensor,
        candidate_objects: torch.Tensor,
    ) -> ListenerOutput:
        """Compute scores for each candidate object given the message.

        Args:
            tokens: Tensor of shape (batch_size, message_length) with token indices.
            candidate_objects: Tensor of shape (batch_size, num_candidates, object_dim) with encoded objects.

        Returns:
            ListenerOutput containing candidate probabilities and predictions.
        """
        batch_size, num_candidates = (
            candidate_objects.size(0),
            candidate_objects.size(1),
        )

        # One-hot encode message tokens and flatten message positions
        message_onehot = F.one_hot(tokens, num_classes=self.config.vocabulary_size)
        message_onehot = message_onehot.view(batch_size, -1).float()

        # Encode message using simplified encoder with residual connection
        message_features = self.message_encoder(
            message_onehot
        )  # (batch_size, hidden_size)
        message_residual = self.message_residual_proj(message_onehot)
        message_features = message_features + message_residual

        # Encode all candidate objects using simplified encoder with residual connection
        candidate_flat = candidate_objects.view(
            -1, self.object_dim
        )  # (batch_size * num_candidates, object_dim)
        candidate_features = self.object_encoder(
            candidate_flat
        )  # (batch_size * num_candidates, hidden_size)
        candidate_residual = self.object_residual_proj(candidate_flat)
        candidate_features = candidate_features + candidate_residual
        candidate_features = candidate_features.view(
            batch_size, num_candidates, -1
        )  # (batch_size, num_candidates, hidden_size)

        # Compute scores for each candidate using message and candidate features
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
        predictions = torch.argmax(probabilities, dim=1)

        return ListenerOutput(probs=probabilities, preds=predictions)


class DotListener(nn.Module):
    """Additive listener: score = <sum of token embeddings, linear object embedding>.

    Each (position, token) has its own embedding and objects are embedded
    linearly, so a candidate's score is a sum of (token, attribute value)
    terms. A novel combination of familiar tokens and attribute values is
    therefore scored exactly as the familiar parts suggest, which lets the
    listener generalize compositionally when the speaker's language allows it.

    Args:
        config: Communication configuration.
    """

    position_offsets: torch.Tensor

    def __init__(self, config: CommunicationConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.message_length * config.vocabulary_size, config.hidden_size
        )
        self.object_encoder = nn.Linear(
            config.object_dim, config.hidden_size, bias=False
        )
        self.register_buffer(
            "position_offsets",
            torch.arange(config.message_length) * config.vocabulary_size,
            persistent=False,
        )

    def forward(
        self, tokens: torch.Tensor, candidate_objects: torch.Tensor
    ) -> ListenerOutput:
        message = self.token_embedding(tokens + self.position_offsets).sum(dim=1)
        candidates = self.object_encoder(candidate_objects)  # (B, N, H)
        scores = torch.einsum("bh,bnh->bn", message, candidates)
        probabilities = F.softmax(scores, dim=-1)
        return ListenerOutput(probs=probabilities, preds=probabilities.argmax(dim=1))


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
        tokens: torch.Tensor,
        candidate_objects: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> ListenerOutput:
        """Compute pragmatic scores for each candidate object given the message.

        This implements RSA-style pragmatic reasoning:
        1. Compute literal listener probabilities P_L(obj | message)
        2. Compute speaker probabilities P_S(message | obj) for each candidate
        3. Compute pragmatic listener probabilities using RSA formula

        Args:
            tokens: Tensor of shape (batch_size, message_length) with token indices.
            candidate_objects: Tensor of shape (batch_size, num_candidates, object_dim) with encoded objects.
            temperature: Optional temperature for RSA computation (overrides internal state).

        Returns:
            ListenerOutput containing pragmatic probabilities and predictions.
        """
        temp = temperature if temperature is not None else self.temperature
        num_candidates = candidate_objects.size(1)

        # Step 1: Compute literal listener probabilities
        literal_output = self.literal_listener(tokens, candidate_objects)
        literal_probs = literal_output.probs

        # Step 2: Compute speaker probabilities for each candidate
        speaker_probs = []

        for i in range(num_candidates):
            # Get the i-th candidate object for each batch
            candidate_obj = candidate_objects[:, i, :]  # (batch_size, object_dim)

            # Generate speaker logits for this candidate
            speaker_output = self.speaker(candidate_obj, temp)
            logits = speaker_output.logits

            # Compute probability of the observed message given this candidate
            message_probs = F.softmax(
                logits, dim=-1
            )  # (batch_size, message_length, vocab_size)

            # Get probability of observed tokens
            token_probs = torch.gather(
                message_probs, dim=-1, index=tokens.unsqueeze(-1)
            ).squeeze(-1)  # (batch_size, message_length)

            # Product over message positions
            candidate_message_prob = torch.prod(token_probs, dim=-1)  # (batch_size,)

            speaker_probs.append(candidate_message_prob)

        speaker_probs_tensor = torch.stack(
            speaker_probs, dim=1
        )  # (batch_size, num_candidates)

        # Step 3: RSA pragmatic listener computation
        # P_pragmatic(obj | message) ∝ P_literal(obj | message) * P_speaker(message | obj)
        pragmatic_scores = literal_probs * speaker_probs_tensor

        # Normalize to get probabilities with numerical stability
        denominator = pragmatic_scores.sum(dim=-1, keepdim=True)

        # If the denominator is extremely small, the speaker model might be assigning
        # near-zero probability to the message for all candidates (untrained model).
        # Fall back to literal probabilities in this case to ensure valid distribution.
        mask = denominator > 1e-20
        probabilities = torch.where(
            mask, pragmatic_scores / (denominator + 1e-30), literal_probs
        )
        predictions = torch.argmax(probabilities, dim=1)

        return ListenerOutput(probs=probabilities, preds=predictions)


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
        self.input_dim = config.object_dim
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
    ) -> SpeakerOutput:
        """Generate message logits and sampled tokens autoregressively.

        Args:
            object_encoding: Tensor of shape (batch_size, input_dim) containing encoded object.
            temperature: Temperature for sampling (default: 1.0).
            teacher_tokens: Optional tensor of shape (batch_size, message_length) for teacher forcing.

        Returns:
            SpeakerOutput containing logits and tokens.
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
                current_token = _sample_tokens(step_logits, temperature, self.training)

            token_ids.append(current_token)

        # Stack outputs
        logits_tensor = torch.stack(
            logits, dim=1
        )  # (batch_size, message_length, vocabulary_size)
        token_ids_tensor = torch.stack(token_ids, dim=1)  # (batch_size, message_length)

        return SpeakerOutput(logits=logits_tensor, tokens=token_ids_tensor)


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
        self.object_dim = config.object_dim
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
        self,
        tokens: torch.Tensor,
        candidate_objects: torch.Tensor,
    ) -> ListenerOutput:
        """Compute scores for each candidate object given the message sequence.

        Args:
            tokens: Tensor of shape (batch_size, message_length) with token indices.
            candidate_objects: Tensor of shape (batch_size, num_candidates, object_dim) with encoded objects.

        Returns:
            ListenerOutput containing probabilities and predictions.
        """
        batch_size, num_candidates = (
            candidate_objects.size(0),
            candidate_objects.size(1),
        )

        # Embed message tokens
        message_embeddings = self.token_embedding(
            tokens
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
        predictions = torch.argmax(probabilities, dim=1)

        return ListenerOutput(probs=probabilities, preds=predictions)
