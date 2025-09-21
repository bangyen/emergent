"""Tests for ablation study functionality.

This module tests the ablation study system including parameter grid generation,
experiment execution, and result aggregation.
"""

import pytest
from typing import Any
from unittest.mock import patch

from langlab.experiments.ablate import generate_parameter_grid, run_ablation_suite


def test_param_grid() -> None:
    """Test that parameter grid generates correct number of runs for given grid."""
    # Test with default parameters
    grid = generate_parameter_grid()
    expected_combinations = 3 * 3 * 3  # 3 vocab sizes × 3 noise levels × 3 length costs
    assert len(grid) == expected_combinations

    # Test with custom parameters
    vocab_sizes = [6, 12]
    noise_levels = [0.0, 0.1]
    length_costs = [0.0, 0.01]

    grid = generate_parameter_grid(vocab_sizes, noise_levels, length_costs)
    expected_combinations = 2 * 2 * 2  # 2 × 2 × 2
    assert len(grid) == expected_combinations

    # Verify all combinations are present
    seen_combinations = set()
    for params in grid:
        combination = (params["V"], params["channel_noise"], params["length_cost"])
        assert (
            combination not in seen_combinations
        ), f"Duplicate combination: {combination}"
        seen_combinations.add(combination)

        # Verify parameter values are correct
        assert params["V"] in vocab_sizes
        assert params["channel_noise"] in noise_levels
        assert params["length_cost"] in length_costs

    # Verify we have all expected combinations
    expected_combinations_set = set()
    for v in vocab_sizes:
        for noise in noise_levels:
            for length_cost in length_costs:
                expected_combinations_set.add((v, noise, length_cost))

    assert seen_combinations == expected_combinations_set


def test_param_grid_empty() -> None:
    """Test parameter grid with empty parameter lists."""
    grid = generate_parameter_grid([], [], [])
    assert len(grid) == 0

    grid = generate_parameter_grid([6], [], [0.0])
    assert len(grid) == 0


def test_param_grid_single_values() -> None:
    """Test parameter grid with single values."""
    grid = generate_parameter_grid([6], [0.0], [0.01])
    assert len(grid) == 1
    assert grid[0]["V"] == 6
    assert grid[0]["channel_noise"] == 0.0
    assert grid[0]["length_cost"] == 0.01


@patch("langlab.experiments.ablate.train")
@patch("langlab.experiments.ablate.evaluate_all_splits")
@patch("langlab.experiments.ablate.compute_zipf_slope_from_checkpoint")
def test_run_ablation_suite_mock(
    mock_zipf: Any, mock_eval: Any, mock_train: Any
) -> None:
    """Test ablation suite execution with mocked dependencies."""
    # Mock return values
    mock_eval.return_value = {
        "train": {"acc": 0.8},
        "iid": {"acc": 0.75},
        "compo": {"acc": 0.7},
    }
    mock_zipf.return_value = -0.8

    # Run ablation suite with minimal parameters
    results = run_ablation_suite(
        runs=1,
        vocab_sizes=[6],
        channel_noise_levels=[0.0],
        length_costs=[0.0],
        n_steps=10,  # Very small for testing
        k=3,
        message_length=1,
        batch_size=2,
    )

    # Verify results
    assert len(results) == 1
    result = results[0]

    assert "experiment_id" in result
    assert "params" in result
    assert "metrics" in result
    assert "zipf_slope" in result

    assert result["params"]["V"] == 6
    assert result["params"]["channel_noise"] == 0.0
    assert result["params"]["length_cost"] == 0.0
    assert result["metrics"]["train"]["acc"] == 0.8
    assert result["zipf_slope"] == -0.8

    # Verify train was called
    mock_train.assert_called_once()

    # Verify evaluate_all_splits was called
    mock_eval.assert_called_once()

    # Verify zipf computation was called
    mock_zipf.assert_called_once()


def test_run_ablation_suite_parameter_validation() -> None:
    """Test that ablation suite validates parameters correctly."""
    # Test with invalid parameter types - the function doesn't actually validate types
    # so we'll test a different validation scenario
    with pytest.raises((TypeError, ValueError, AttributeError)):
        run_ablation_suite(
            runs=1,
            vocab_sizes=None,  # This should cause an error
            channel_noise_levels=[0.0],
            length_costs=[0.0],
        )


@patch("langlab.experiments.ablate.os.makedirs")
@patch("langlab.experiments.ablate.train")
@patch("langlab.experiments.ablate.evaluate_all_splits")
@patch("langlab.experiments.ablate.compute_zipf_slope_from_checkpoint")
def test_run_ablation_suite_directory_creation(
    mock_zipf: Any, mock_eval: Any, mock_train: Any, mock_makedirs: Any
) -> None:
    """Test that ablation suite creates necessary directories."""
    # Mock return values
    mock_eval.return_value = {
        "train": {"acc": 0.8},
        "iid": {"acc": 0.75},
        "compo": {"acc": 0.7},
    }
    mock_zipf.return_value = -0.8

    # Run ablation suite
    run_ablation_suite(
        runs=1,
        vocab_sizes=[6],
        channel_noise_levels=[0.0],
        length_costs=[0.0],
        n_steps=10,
    )

    # Verify directories were created
    mock_makedirs.assert_called()


def test_run_ablation_suite_experiment_id_format() -> None:
    """Test that experiment IDs are formatted correctly."""
    with patch("langlab.experiments.ablate.train"), patch(
        "langlab.analysis.eval.evaluate_all_splits"
    ) as mock_eval, patch(
        "langlab.experiments.ablate.compute_zipf_slope_from_checkpoint"
    ) as mock_zipf, patch(
        "torch.load"
    ) as mock_load:

        mock_eval.return_value = {
            "train": {"acc": 0.8},
            "iid": {"acc": 0.75},
            "compo": {"acc": 0.7},
        }
        mock_zipf.return_value = -0.8

        # Mock checkpoint loading to avoid architecture mismatch
        import torch

        mock_checkpoint = {
            "config": type(
                "Config",
                (),
                {
                    "use_sequence_models": True,
                    "use_contrastive": False,  # Explicitly disable contrastive learning
                    "vocabulary_size": 12,
                    "message_length": 2,
                    "hidden_size": 128,
                    "multimodal": False,
                    "distractors": 0,
                    "pragmatic": False,
                    "seed": 42,
                },
            )(),
            "speaker_state_dict": {
                "object_encoder.0.weight": torch.randn(
                    128, 8
                ),  # hidden_size, TOTAL_ATTRIBUTES
                "object_encoder.0.bias": torch.randn(128),
                "object_encoder.2.weight": torch.randn(
                    128, 128
                ),  # hidden_size, hidden_size
                "object_encoder.2.bias": torch.randn(128),
                "gru.weight_ih_l0": torch.randn(
                    384, 140
                ),  # 3*hidden_size, hidden_size+vocab_size
                "gru.weight_hh_l0": torch.randn(384, 128),  # 3*hidden_size, hidden_size
                "gru.bias_ih_l0": torch.randn(384),
                "gru.bias_hh_l0": torch.randn(384),
                "output_proj.weight": torch.randn(12, 128),  # vocab_size, hidden_size
                "output_proj.bias": torch.randn(12),
                "token_embedding.weight": torch.randn(12, 12),  # vocab_size, vocab_size
            },
            "listener_state_dict": {
                "token_embedding.weight": torch.randn(
                    12, 128
                ),  # vocab_size, hidden_size
                "message_encoder.weight_ih_l0": torch.randn(
                    384, 128
                ),  # 3*hidden_size, hidden_size
                "message_encoder.weight_hh_l0": torch.randn(
                    384, 128
                ),  # 3*hidden_size, hidden_size
                "message_encoder.bias_ih_l0": torch.randn(384),
                "message_encoder.bias_hh_l0": torch.randn(384),
                "object_encoder.0.weight": torch.randn(
                    128, 8
                ),  # hidden_size, TOTAL_ATTRIBUTES
                "object_encoder.0.bias": torch.randn(128),
                "object_encoder.2.weight": torch.randn(
                    128, 128
                ),  # hidden_size, hidden_size
                "object_encoder.2.bias": torch.randn(128),
                "bilinear_scorer.weight": torch.randn(
                    1, 128, 128
                ),  # 1, hidden_size, hidden_size
                "bilinear_scorer.bias": torch.randn(1),
            },
        }
        mock_load.return_value = mock_checkpoint

        results = run_ablation_suite(
            runs=1,
            vocab_sizes=[12],
            channel_noise_levels=[0.05],
            length_costs=[0.01],
            n_steps=10,
            use_sequence_models=True,
        )

        assert len(results) == 1
        experiment_id = results[0]["experiment_id"]

        # Check that experiment ID contains parameter values
        assert "V12" in experiment_id
        assert "noise0.05" in experiment_id
        assert "len0.01" in experiment_id
