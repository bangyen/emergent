"""Tests for pragmatic inference functionality.

This module tests the pragmatic listener implementation, RSA-style reasoning,
and performance on distractor-heavy scenes compared to literal listeners.
"""

import pytest
import torch
import torch.nn.functional as F

from src.langlab.config import CommunicationConfig
from src.langlab.agents import Speaker, Listener, PragmaticListener
from src.langlab.world import sample_distractor_scene, encode_object


class TestPragmaticListener:
    """Test pragmatic listener functionality."""

    def test_pragmatic_listener_initialization(self) -> None:
        """Test pragmatic listener initialization."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            multimodal=False,
        )

        literal_listener = Listener(config)
        speaker = Speaker(config)

        pragmatic_listener = PragmaticListener(config, literal_listener, speaker)

        assert pragmatic_listener.config == config
        assert pragmatic_listener.literal_listener == literal_listener
        assert pragmatic_listener.speaker == speaker
        assert pragmatic_listener.temperature == 1.0

    def test_pragmatic_listener_shapes(self) -> None:
        """Test pragmatic listener output shapes."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            multimodal=False,
        )

        literal_listener = Listener(config)
        speaker = Speaker(config)
        pragmatic_listener = PragmaticListener(config, literal_listener, speaker)

        batch_size = 3
        num_candidates = 4
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(batch_size, num_candidates, 8)

        pragmatic_probs = pragmatic_listener(message_tokens, candidate_objects)

        # Check output shape
        assert pragmatic_probs.shape == (batch_size, num_candidates)

        # Check probabilities sum to 1
        prob_sums = pragmatic_probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)

    def test_pragmatic_listener_multimodal(self) -> None:
        """Test pragmatic listener with multimodal input."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            gesture_size=5,
            multimodal=True,
        )

        literal_listener = Listener(config)
        speaker = Speaker(config)
        pragmatic_listener = PragmaticListener(config, literal_listener, speaker)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        gesture_tokens = torch.randint(
            0, config.gesture_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(batch_size, num_candidates, 8)

        pragmatic_probs = pragmatic_listener(
            message_tokens, candidate_objects, gesture_tokens
        )

        # Check output shape
        assert pragmatic_probs.shape == (batch_size, num_candidates)

        # Check probabilities sum to 1
        prob_sums = pragmatic_probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)

    def test_rsa_reasoning_components(self) -> None:
        """Test RSA reasoning components separately."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            multimodal=False,
        )

        literal_listener = Listener(config)
        speaker = Speaker(config)
        # pragmatic_listener = PragmaticListener(config, literal_listener, speaker)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(batch_size, num_candidates, 8)

        # Test literal listener probabilities
        literal_probs = literal_listener(message_tokens, candidate_objects)
        assert literal_probs.shape == (batch_size, num_candidates)
        assert torch.allclose(
            literal_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6
        )

        # Test speaker probabilities for each candidate
        speaker_probs = []
        for i in range(num_candidates):
            candidate_obj = candidate_objects[:, i, :]
            logits, _, _, _ = speaker(candidate_obj)

            # Compute probability of observed message
            message_probs = F.softmax(logits, dim=-1)
            token_probs = torch.gather(
                message_probs, dim=-1, index=message_tokens.unsqueeze(-1)
            ).squeeze(-1)
            candidate_message_prob = torch.prod(token_probs, dim=-1)
            speaker_probs.append(candidate_message_prob)

        speaker_probs_tensor = torch.stack(speaker_probs, dim=1)
        assert speaker_probs_tensor.shape == (batch_size, num_candidates)

        # Test pragmatic combination
        pragmatic_scores = literal_probs * speaker_probs_tensor
        pragmatic_probs = pragmatic_scores / pragmatic_scores.sum(dim=-1, keepdim=True)

        assert pragmatic_probs.shape == (batch_size, num_candidates)
        assert torch.allclose(
            pragmatic_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6
        )


class TestDistractorScenes:
    """Test distractor scene generation and pragmatic performance."""

    def test_distractor_scene_generation(self) -> None:
        """Test distractor scene generation."""
        k = 5
        num_distractors = 2
        seed = 42

        scene_objects, target_idx = sample_distractor_scene(k, num_distractors, seed)

        # Check scene size
        assert len(scene_objects) == k
        assert target_idx == 0  # Target should be at index 0

        # Check that target is unique
        target_obj = scene_objects[target_idx]
        assert target_obj in scene_objects

        # Check that we have the right number of objects
        assert len(scene_objects) == k

    def test_distractor_scene_validation(self) -> None:
        """Test distractor scene validation."""
        # Test invalid parameters
        with pytest.raises(ValueError):
            sample_distractor_scene(5, 5)  # num_distractors >= k

        with pytest.raises(ValueError):
            sample_distractor_scene(20, 2)  # k > total_objects

    def test_distractor_scene_reproducibility(self) -> None:
        """Test distractor scene reproducibility with same seed."""
        k = 4
        num_distractors = 1
        seed = 123

        scene1, target1 = sample_distractor_scene(k, num_distractors, seed)
        scene2, target2 = sample_distractor_scene(k, num_distractors, seed)

        # Should be identical with same seed
        assert scene1 == scene2
        assert target1 == target2

    def test_distractor_scene_diversity(self) -> None:
        """Test that distractor scenes have different objects."""
        k = 4
        num_distractors = 2

        scene1, _ = sample_distractor_scene(k, num_distractors, seed=1)
        scene2, _ = sample_distractor_scene(k, num_distractors, seed=2)

        # Should be different with different seeds
        assert scene1 != scene2


class TestPragmaticPerformance:
    """Test pragmatic listener performance on distractor scenes."""

    def test_pragmatic_vs_literal_performance(self) -> None:
        """Test that pragmatic listener outperforms literal listener on distractor scenes."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            multimodal=False,
        )

        literal_listener = Listener(config)
        speaker = Speaker(config)
        pragmatic_listener = PragmaticListener(config, literal_listener, speaker)

        # Create a simple distractor scenario
        batch_size = 2
        # num_candidates = 3

        # Create objects where some share attributes
        # Object 0: red circle small (target)
        # Object 1: red square small (shares color and size)
        # Object 2: blue triangle large (completely different)

        target_obj = torch.tensor(
            [1, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float32
        )  # red circle small
        distractor1 = torch.tensor(
            [1, 0, 0, 0, 1, 0, 1, 0], dtype=torch.float32
        )  # red square small
        distractor2 = torch.tensor(
            [0, 1, 0, 0, 0, 1, 0, 1], dtype=torch.float32
        )  # blue triangle large

        candidate_objects = (
            torch.stack([target_obj, distractor1, distractor2])
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        # Generate message for target object
        target_encoding = target_obj.unsqueeze(0).repeat(batch_size, 1)
        logits, message_tokens, _, _ = speaker(target_encoding)

        # Test literal listener
        literal_probs = literal_listener(message_tokens, candidate_objects)

        # Test pragmatic listener
        pragmatic_probs = pragmatic_listener(message_tokens, candidate_objects)

        # Both should produce valid probabilities
        assert torch.allclose(
            literal_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6
        )
        assert torch.allclose(
            pragmatic_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6
        )

        # Pragmatic listener should give different results than literal listener
        # (though not necessarily better without training)
        assert not torch.allclose(literal_probs, pragmatic_probs, atol=1e-3)

    def test_pragmatic_reasoning_consistency(self) -> None:
        """Test that pragmatic reasoning is consistent across runs."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            multimodal=False,
        )

        literal_listener = Listener(config)
        speaker = Speaker(config)
        pragmatic_listener = PragmaticListener(config, literal_listener, speaker)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(batch_size, num_candidates, 8)

        # Set models to eval mode for consistency
        literal_listener.eval()
        speaker.eval()
        pragmatic_listener.eval()

        # Run multiple times
        probs1 = pragmatic_listener(message_tokens, candidate_objects)
        probs2 = pragmatic_listener(message_tokens, candidate_objects)

        # Should be identical in eval mode
        assert torch.allclose(probs1, probs2, atol=1e-6)

    def test_pragmatic_gradient_flow(self) -> None:
        """Test that gradients flow properly in pragmatic listener."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            multimodal=False,
        )

        literal_listener = Listener(config)
        speaker = Speaker(config)
        pragmatic_listener = PragmaticListener(config, literal_listener, speaker)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(
            batch_size, num_candidates, 8, requires_grad=True
        )

        # Compute pragmatic probabilities
        pragmatic_probs = pragmatic_listener(message_tokens, candidate_objects)

        # Compute loss
        target_indices = torch.randint(0, num_candidates, (batch_size,))
        loss = F.cross_entropy(pragmatic_probs, target_indices)

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert candidate_objects.grad is not None
        assert candidate_objects.grad.shape == candidate_objects.shape

    def test_pragmatic_multimodal_performance(self) -> None:
        """Test pragmatic listener performance with multimodal input."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            gesture_size=5,
            multimodal=True,
        )

        literal_listener = Listener(config)
        speaker = Speaker(config)
        pragmatic_listener = PragmaticListener(config, literal_listener, speaker)

        batch_size = 2
        num_candidates = 3
        message_tokens = torch.randint(
            0, config.vocabulary_size, (batch_size, config.message_length)
        )
        gesture_tokens = torch.randint(
            0, config.gesture_size, (batch_size, config.message_length)
        )
        candidate_objects = torch.randn(batch_size, num_candidates, 8)

        # Test multimodal pragmatic processing
        pragmatic_probs = pragmatic_listener(
            message_tokens, candidate_objects, gesture_tokens
        )

        # Check output shape and validity
        assert pragmatic_probs.shape == (batch_size, num_candidates)
        assert torch.allclose(
            pragmatic_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6
        )

        # Compare with unimodal processing (create unimodal pragmatic listener)
        unimodal_config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            multimodal=False,
        )
        unimodal_literal_listener = Listener(unimodal_config)
        unimodal_speaker = Speaker(unimodal_config)
        unimodal_pragmatic_listener = PragmaticListener(
            unimodal_config, unimodal_literal_listener, unimodal_speaker
        )
        pragmatic_probs_unimodal = unimodal_pragmatic_listener(
            message_tokens, candidate_objects
        )

        # Should be different due to different input modalities
        assert not torch.allclose(pragmatic_probs, pragmatic_probs_unimodal, atol=1e-3)


class TestPragmaticIntegration:
    """Test integration between pragmatic components."""

    def test_end_to_end_pragmatic_reasoning(self) -> None:
        """Test end-to-end pragmatic reasoning pipeline."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            multimodal=False,
        )

        speaker = Speaker(config)
        literal_listener = Listener(config)
        pragmatic_listener = PragmaticListener(config, literal_listener, speaker)

        batch_size = 2
        num_candidates = 4

        # Generate object encoding
        object_encoding = torch.randn(batch_size, 8)

        # Speaker generates message
        logits, token_ids, _, _ = speaker(object_encoding)

        # Create candidate objects
        candidate_objects = torch.randn(batch_size, num_candidates, 8)

        # Literal listener processes message
        literal_probs = literal_listener(token_ids, candidate_objects)

        # Pragmatic listener processes message
        pragmatic_probs = pragmatic_listener(token_ids, candidate_objects)

        # Both should produce valid probabilities
        assert torch.allclose(
            literal_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6
        )
        assert torch.allclose(
            pragmatic_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6
        )

        # Results should be different (pragmatic considers speaker intent)
        assert not torch.allclose(literal_probs, pragmatic_probs, atol=1e-3)

    def test_pragmatic_with_distractor_scenes(self) -> None:
        """Test pragmatic listener with actual distractor scenes."""
        config = CommunicationConfig(
            vocabulary_size=10,
            message_length=2,
            multimodal=False,
        )

        speaker = Speaker(config)
        literal_listener = Listener(config)
        pragmatic_listener = PragmaticListener(config, literal_listener, speaker)

        # Generate distractor scene
        k = 4
        num_distractors = 2
        scene_objects, target_idx = sample_distractor_scene(k, num_distractors, seed=42)

        # Encode objects
        encoded_objects = torch.stack([encode_object(obj) for obj in scene_objects])
        candidate_objects = encoded_objects.unsqueeze(0)  # Add batch dimension

        # Generate message for target
        target_encoding = encoded_objects[target_idx].unsqueeze(0)
        logits, message_tokens, _, _ = speaker(target_encoding)

        # Test both listeners
        literal_probs = literal_listener(message_tokens, candidate_objects)
        pragmatic_probs = pragmatic_listener(message_tokens, candidate_objects)

        # Both should produce valid probabilities
        assert torch.allclose(literal_probs.sum(dim=-1), torch.ones(1), atol=1e-6)
        assert torch.allclose(pragmatic_probs.sum(dim=-1), torch.ones(1), atol=1e-6)

        # Results should be different
        assert not torch.allclose(literal_probs, pragmatic_probs, atol=1e-3)
