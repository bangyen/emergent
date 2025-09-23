"""Tests for entropy bonus regularization.

This module contains tests to validate the entropy bonus computation
and ensure it behaves correctly for encouraging exploration.
"""

import torch
import torch.nn.functional as F

from src.langlab.training.train import compute_entropy_bonus


def test_entropy_nonnegative() -> None:
    """Test that entropy bonus is computed per step and is non-negative.

    Validates that entropy is computed correctly and is always >= 0.
    """
    batch_size = 3
    message_length = 4
    vocab_size = 10

    # Create logits with different distributions
    logits = torch.randn(batch_size, message_length, vocab_size)

    # Compute entropy bonus
    entropy_bonus = compute_entropy_bonus(logits)

    # Check that entropy bonus is non-negative
    assert entropy_bonus >= 0

    # Check that it's a scalar tensor
    assert entropy_bonus.dim() == 0

    # Check that it's finite
    assert torch.isfinite(entropy_bonus)


def test_entropy_maximum() -> None:
    """Test that entropy is maximized for uniform distributions."""
    batch_size = 2
    message_length = 3
    vocab_size = 5

    # Uniform distribution should have maximum entropy
    uniform_logits = torch.zeros(batch_size, message_length, vocab_size)
    uniform_entropy = compute_entropy_bonus(uniform_logits)

    # Deterministic distribution should have minimum entropy
    deterministic_logits = torch.zeros(batch_size, message_length, vocab_size)
    deterministic_logits[:, :, 0] = 10.0  # Make first token very likely
    deterministic_entropy = compute_entropy_bonus(deterministic_logits)

    # Uniform should have higher entropy
    assert uniform_entropy > deterministic_entropy


def test_entropy_calculation() -> None:
    """Test that entropy calculation matches manual computation."""
    # Create a known distribution
    logits = torch.tensor([[[1.0, 2.0, 0.0]]])  # Shape: (1, 1, 3)

    # Manual entropy calculation
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    manual_entropy = -(probs * log_probs).sum(dim=-1).mean()

    # Function entropy calculation
    function_entropy = compute_entropy_bonus(logits)

    # Should be equal
    assert torch.allclose(manual_entropy, function_entropy, atol=1e-6)


def test_entropy_different_shapes() -> None:
    """Test entropy bonus with different tensor shapes."""
    # Test different batch sizes
    for batch_size in [1, 2, 5]:
        logits = torch.randn(batch_size, 3, 8)
        entropy = compute_entropy_bonus(logits)
        assert entropy >= 0
        assert torch.isfinite(entropy)

    # Test different message lengths
    for message_length in [1, 2, 5]:
        logits = torch.randn(2, message_length, 6)
        entropy = compute_entropy_bonus(logits)
        assert entropy >= 0
        assert torch.isfinite(entropy)

    # Test different vocabulary sizes
    for vocab_size in [2, 5, 20]:
        logits = torch.randn(3, 2, vocab_size)
        entropy = compute_entropy_bonus(logits)
        assert entropy >= 0
        assert torch.isfinite(entropy)


def test_entropy_gradient() -> None:
    """Test that entropy bonus has proper gradients."""
    batch_size = 2
    message_length = 3
    vocab_size = 5

    logits = torch.randn(batch_size, message_length, vocab_size, requires_grad=True)
    entropy = compute_entropy_bonus(logits)

    # Backward pass
    entropy.backward()

    # Check that gradients exist
    assert logits.grad is not None
    assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))


def test_entropy_extreme_cases() -> None:
    """Test entropy bonus with extreme cases."""
    # Very large logits (should still be finite)
    large_logits = torch.tensor([[[100.0, 0.0, 0.0]]])
    entropy_large = compute_entropy_bonus(large_logits)
    assert torch.isfinite(entropy_large)
    assert entropy_large >= 0

    # Very small logits
    small_logits = torch.tensor([[[-100.0, -100.0, -100.0]]])
    entropy_small = compute_entropy_bonus(small_logits)
    assert torch.isfinite(entropy_small)
    assert entropy_small >= 0

    # Mixed large and small
    mixed_logits = torch.tensor([[[100.0, -100.0, 0.0]]])
    entropy_mixed = compute_entropy_bonus(mixed_logits)
    assert torch.isfinite(entropy_mixed)
    assert entropy_mixed >= 0
