"""Comprehensive tests for meta-learning agents.

This module tests the Speaker and Listener agents with meta-learning
capabilities for quick adaptation to new tasks and environments.
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Tuple

from langlab.core.meta_agents import (
    MetaSpeaker,
    MetaListener,
    MetaLearner,
)
from langlab.core.config import CommunicationConfig


class TestMetaSpeaker:
    """Test the MetaSpeaker class."""

    @pytest.fixture
    def config(self) -> CommunicationConfig:
        """Create a test configuration."""
        return CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            hidden_size=32,
            multimodal=False,
        )

    def test_meta_speaker_init(self, config: CommunicationConfig) -> None:
        """Test meta speaker initialization."""
        speaker = MetaSpeaker(config)

        assert isinstance(speaker, nn.Module)
        assert speaker.config == config
        assert speaker.input_dim == config.object_dim
        assert speaker.inner_lr == 0.01
        assert speaker.outer_lr == 0.001
        assert hasattr(speaker, "object_encoder")
        assert hasattr(speaker, "meta_encoder")
        assert hasattr(speaker, "message_generator")
        assert hasattr(speaker, "output_layers")

    def test_meta_speaker_forward(self, config: CommunicationConfig) -> None:
        """Test meta speaker forward pass without adaptation."""
        speaker = MetaSpeaker(config)
        speaker.eval()

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)

        logits, tokens, gesture_logits, gesture_tokens = speaker(object_encoding)

        assert logits.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert tokens.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert gesture_logits is None  # Non-multimodal
        assert gesture_tokens is None  # Non-multimodal

    def test_meta_speaker_forward_with_adaptation(
        self, config: CommunicationConfig
    ) -> None:
        """Test meta speaker forward pass with adapted parameters."""
        speaker = MetaSpeaker(config)
        speaker.eval()

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)

        # Create dummy adapted parameters
        adapted_params = {
            "meta_encoder.weight": torch.randn(config.hidden_size, config.hidden_size),
            "meta_encoder.bias": torch.randn(config.hidden_size),
        }

        logits, tokens, gesture_logits, gesture_tokens = speaker(
            object_encoding, adapted_params=adapted_params
        )

        assert logits.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert tokens.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert gesture_logits is None
        assert gesture_tokens is None

    def test_meta_speaker_forward_with_temperature(
        self, config: CommunicationConfig
    ) -> None:
        """Test meta speaker forward pass with temperature scaling."""
        speaker = MetaSpeaker(config)
        speaker.eval()

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)
        temperature = 0.5

        logits, tokens, gesture_logits, gesture_tokens = speaker(
            object_encoding, temperature
        )

        assert logits.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert tokens.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )

    def test_meta_speaker_training_mode(self, config: CommunicationConfig) -> None:
        """Test meta speaker behavior in training mode."""
        speaker = MetaSpeaker(config)
        speaker.train()

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)

        logits, tokens, gesture_logits, gesture_tokens = speaker(object_encoding)

        assert logits.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )
        assert tokens.shape == (
            batch_size,
            config.message_length,
            config.vocabulary_size,
        )

    def test_meta_speaker_weight_initialization(
        self, config: CommunicationConfig
    ) -> None:
        """Test that weights are properly initialized."""
        speaker = MetaSpeaker(config)

        # Check that weights are initialized (not all zeros)
        for name, param in speaker.named_parameters():
            if "weight" in name and len(param.shape) > 1:  # Skip bias and 1D weights
                assert not torch.allclose(param, torch.zeros_like(param))

    def test_meta_speaker_different_temperatures(
        self, config: CommunicationConfig
    ) -> None:
        """Test meta speaker with different temperature values."""
        speaker = MetaSpeaker(config)
        speaker.eval()

        batch_size = 2
        object_encoding = torch.randn(batch_size, config.object_dim)

        # Test with different temperatures
        for temp in [0.1, 0.5, 1.0, 2.0]:
            logits, tokens, _, _ = speaker(object_encoding, temperature=temp)
            assert logits.shape == (
                batch_size,
                config.message_length,
                config.vocabulary_size,
            )
            assert tokens.shape == (
                batch_size,
                config.message_length,
                config.vocabulary_size,
            )


class TestMetaListener:
    """Test the MetaListener class."""

    @pytest.fixture
    def config(self) -> CommunicationConfig:
        """Create a test configuration."""
        return CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            hidden_size=32,
            multimodal=False,
        )

    @pytest.fixture
    def multimodal_config(self) -> CommunicationConfig:
        """Create a multimodal test configuration."""
        return CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            hidden_size=32,
            multimodal=True,
            gesture_size=5,
        )

    def test_meta_listener_init(self, config: CommunicationConfig) -> None:
        """Test meta listener initialization."""
        listener = MetaListener(config)

        assert isinstance(listener, nn.Module)
        assert listener.config == config
        assert listener.message_dim == config.vocabulary_size
        assert listener.object_dim == config.object_dim
        assert listener.inner_lr == 0.01
        assert listener.outer_lr == 0.001
        assert hasattr(listener, "message_encoder")
        assert hasattr(listener, "object_encoder")
        assert hasattr(listener, "meta_message_encoder")
        assert hasattr(listener, "meta_object_encoder")
        assert hasattr(listener, "cross_attention")
        assert hasattr(listener, "scorer")

    def test_meta_listener_forward(self, config: CommunicationConfig) -> None:
        """Test meta listener forward pass without adaptation."""
        listener = MetaListener(config)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randn(
            batch_size, config.message_length, config.vocabulary_size
        )
        candidate_objects = torch.randn(batch_size, num_candidates, config.object_dim)

        output = listener(message_tokens, candidate_objects)

        assert output.shape == (batch_size, num_candidates)
        # Probabilities should sum to 1 for each batch
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_meta_listener_forward_with_adaptation(
        self, config: CommunicationConfig
    ) -> None:
        """Test meta listener forward pass with adapted parameters."""
        listener = MetaListener(config)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randn(
            batch_size, config.message_length, config.vocabulary_size
        )
        candidate_objects = torch.randn(batch_size, num_candidates, config.object_dim)

        # Create dummy adapted parameters
        adapted_params = {
            "meta_message_encoder.weight": torch.randn(
                config.hidden_size, config.hidden_size
            ),
            "meta_message_encoder.bias": torch.randn(config.hidden_size),
            "meta_object_encoder.weight": torch.randn(
                config.hidden_size, config.hidden_size
            ),
            "meta_object_encoder.bias": torch.randn(config.hidden_size),
        }

        output = listener(
            message_tokens, candidate_objects, adapted_params=adapted_params
        )

        assert output.shape == (batch_size, num_candidates)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_meta_listener_forward_with_gestures(
        self, multimodal_config: CommunicationConfig
    ) -> None:
        """Test meta listener forward pass with gesture tokens."""
        listener = MetaListener(multimodal_config)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randn(
            batch_size,
            multimodal_config.message_length,
            multimodal_config.vocabulary_size,
        )
        candidate_objects = torch.randn(
            batch_size, num_candidates, multimodal_config.object_dim
        )
        gesture_tokens = torch.randn(
            batch_size, multimodal_config.message_length, multimodal_config.gesture_size
        )

        output = listener(message_tokens, candidate_objects, gesture_tokens)

        assert output.shape == (batch_size, num_candidates)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_meta_listener_weight_initialization(
        self, config: CommunicationConfig
    ) -> None:
        """Test that weights are properly initialized."""
        listener = MetaListener(config)

        # Check that weights are initialized (not all zeros)
        for name, param in listener.named_parameters():
            if "weight" in name and len(param.shape) > 1:  # Skip bias and 1D weights
                assert not torch.allclose(param, torch.zeros_like(param))

    def test_meta_listener_different_candidate_counts(
        self, config: CommunicationConfig
    ) -> None:
        """Test meta listener with different numbers of candidates."""
        listener = MetaListener(config)

        batch_size = 2
        message_tokens = torch.randn(
            batch_size, config.message_length, config.vocabulary_size
        )

        # Test with different numbers of candidates
        for num_candidates in [1, 2, 5, 10]:
            candidate_objects = torch.randn(
                batch_size, num_candidates, config.object_dim
            )
            output = listener(message_tokens, candidate_objects)

            assert output.shape == (batch_size, num_candidates)
            assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)


class TestMetaLearner:
    """Test the MetaLearner class."""

    @pytest.fixture
    def config(self) -> CommunicationConfig:
        """Create a test configuration."""
        return CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            hidden_size=32,
            multimodal=False,
        )

    @pytest.fixture
    def meta_learner(self, config: CommunicationConfig) -> MetaLearner:
        """Create a meta learner instance."""
        speaker = MetaSpeaker(config)
        listener = MetaListener(config)
        return MetaLearner(speaker, listener)

    @pytest.fixture
    def support_set(
        self, config: CommunicationConfig
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Create a dummy support set."""
        batch_size = 2
        num_candidates = 3

        support_set = []
        for _ in range(5):  # 5 support examples
            scene = torch.randn(batch_size, config.object_dim)
            target = torch.randint(0, num_candidates, (batch_size,))
            candidates = torch.randn(batch_size, num_candidates, config.object_dim)
            support_set.append((scene, target, candidates))

        return support_set

    def test_meta_learner_init(self, config: CommunicationConfig) -> None:
        """Test meta learner initialization."""
        speaker = MetaSpeaker(config)
        listener = MetaListener(config)
        meta_learner = MetaLearner(speaker, listener)

        assert meta_learner.speaker == speaker
        assert meta_learner.listener == listener
        assert meta_learner.inner_lr == 0.01

    def test_meta_learner_init_custom_lr(self, config: CommunicationConfig) -> None:
        """Test meta learner initialization with custom learning rate."""
        speaker = MetaSpeaker(config)
        listener = MetaListener(config)
        meta_learner = MetaLearner(speaker, listener, inner_lr=0.05)

        assert meta_learner.inner_lr == 0.05

    def test_meta_learner_adapt(
        self,
        meta_learner: MetaLearner,
        support_set: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        """Test meta learner adaptation."""
        speaker_params, listener_params = meta_learner.adapt(
            support_set, n_inner_steps=3
        )

        assert isinstance(speaker_params, dict)
        assert isinstance(listener_params, dict)

        # Check that parameters have the right structure
        for name, param in speaker_params.items():
            assert isinstance(param, torch.Tensor)
            assert "meta_encoder" in name or name in meta_learner.speaker.state_dict()

        for name, param in listener_params.items():
            assert isinstance(param, torch.Tensor)
            assert "meta_" in name or name in meta_learner.listener.state_dict()

    def test_meta_learner_adapt_empty_support_set(
        self, meta_learner: MetaLearner
    ) -> None:
        """Test meta learner adaptation with empty support set."""
        empty_support_set: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        speaker_params, listener_params = meta_learner.adapt(empty_support_set)

        assert isinstance(speaker_params, dict)
        assert isinstance(listener_params, dict)

    def test_meta_learner_adapt_zero_steps(
        self,
        meta_learner: MetaLearner,
        support_set: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        """Test meta learner adaptation with zero inner steps."""
        speaker_params, listener_params = meta_learner.adapt(
            support_set, n_inner_steps=0
        )

        assert isinstance(speaker_params, dict)
        assert isinstance(listener_params, dict)

    def test_meta_learner_adapt_different_steps(
        self,
        meta_learner: MetaLearner,
        support_set: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        """Test meta learner adaptation with different numbers of inner steps."""
        for n_steps in [1, 3, 5, 10]:
            speaker_params, listener_params = meta_learner.adapt(
                support_set, n_inner_steps=n_steps
            )

            assert isinstance(speaker_params, dict)
            assert isinstance(listener_params, dict)

    def test_meta_learner_compute_speaker_grads(
        self,
        meta_learner: MetaLearner,
        support_set: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        """Test speaker gradient computation."""
        speaker_params = {
            name: param.clone()
            for name, param in meta_learner.speaker.named_parameters()
        }

        grads = meta_learner._compute_speaker_grads(support_set, speaker_params)

        assert isinstance(grads, dict)
        # Should only have gradients for meta_encoder parameters
        for name in grads.keys():
            assert "meta_encoder" in name

    def test_meta_learner_compute_listener_grads(
        self,
        meta_learner: MetaLearner,
        support_set: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        """Test listener gradient computation."""
        listener_params = {
            name: param.clone()
            for name, param in meta_learner.listener.named_parameters()
        }

        grads = meta_learner._compute_listener_grads(support_set, listener_params)

        assert isinstance(grads, dict)
        # Should only have gradients for meta_ parameters
        for name in grads.keys():
            assert "meta_" in name

    def test_meta_learner_different_support_set_sizes(
        self, meta_learner: MetaLearner, config: CommunicationConfig
    ) -> None:
        """Test meta learner with different support set sizes."""
        batch_size = 2
        num_candidates = 3

        # Test with different support set sizes
        for support_size in [1, 3, 5, 10]:
            support_set = []
            for _ in range(support_size):
                scene = torch.randn(batch_size, config.object_dim)
                target = torch.randint(0, num_candidates, (batch_size,))
                candidates = torch.randn(batch_size, num_candidates, config.object_dim)
                support_set.append((scene, target, candidates))

            speaker_params, listener_params = meta_learner.adapt(
                support_set, n_inner_steps=2
            )

            assert isinstance(speaker_params, dict)
            assert isinstance(listener_params, dict)
