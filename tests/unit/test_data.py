"""Tests for the data module.

This module contains tests for the ReferentialGameDataset class and its
functionality in generating and managing referential game data.
"""

import pytest
import torch

from src.langlab.data.data import ReferentialGameDataset
from src.langlab.data.world import TOTAL_ATTRIBUTES


class TestReferentialGameDataset:
    """Test ReferentialGameDataset functionality."""

    def test_dataset_len(self) -> None:
        """Test that dataset length equals n_scenes."""
        n_scenes = 50
        k = 3
        dataset = ReferentialGameDataset(n_scenes, k, seed=42)
        assert len(dataset) == n_scenes

    def test_dataset_shapes(self) -> None:
        """Test that tensors have expected shapes."""
        n_scenes = 10
        k = 4
        dataset = ReferentialGameDataset(n_scenes, k, seed=42)

        scene_tensor, target_idx, candidate_encodings = dataset[0]

        # Scene tensor should be [K, TOTAL_ATTRIBUTES]
        assert scene_tensor.shape == (k, TOTAL_ATTRIBUTES)

        # Target index should be valid
        assert 0 <= target_idx < k

        # Candidate encodings should match scene tensor
        assert candidate_encodings.shape == scene_tensor.shape
        assert torch.equal(scene_tensor, candidate_encodings)

    def test_dataset_reproducible(self) -> None:
        """Test that dataset generation is reproducible with same seed."""
        n_scenes = 5
        k = 3
        seed = 123

        dataset1 = ReferentialGameDataset(n_scenes, k, seed)
        dataset2 = ReferentialGameDataset(n_scenes, k, seed)

        # Check that all scenes are identical
        for i in range(n_scenes):
            scene1, target1, candidates1 = dataset1[i]
            scene2, target2, candidates2 = dataset2[i]

            assert torch.equal(scene1, scene2)
            assert target1 == target2
            assert torch.equal(candidates1, candidates2)

    def test_dataset_different_seeds(self) -> None:
        """Test that different seeds produce different datasets."""
        n_scenes = 5
        k = 3

        dataset1 = ReferentialGameDataset(n_scenes, k, seed=123)
        dataset2 = ReferentialGameDataset(n_scenes, k, seed=456)

        # Check that at least some scenes are different
        scenes_different = False
        for i in range(n_scenes):
            scene1, _, _ = dataset1[i]
            scene2, _, _ = dataset2[i]
            if not torch.equal(scene1, scene2):
                scenes_different = True
                break

        assert scenes_different

    def test_dataset_iteration(self) -> None:
        """Test that dataset can be iterated over."""
        n_scenes = 3
        k = 2
        dataset = ReferentialGameDataset(n_scenes, k, seed=42)

        scenes = list(dataset)
        assert len(scenes) == n_scenes

        # Check that iteration produces same results as indexing
        for i, (scene_tensor, target_idx, candidates) in enumerate(scenes):
            expected_scene, expected_target, expected_candidates = dataset[i]
            assert torch.equal(scene_tensor, expected_scene)
            assert target_idx == expected_target
            assert torch.equal(candidates, expected_candidates)

    def test_dataset_index_bounds(self) -> None:
        """Test that dataset raises IndexError for out-of-bounds access."""
        n_scenes = 5
        k = 3
        dataset = ReferentialGameDataset(n_scenes, k, seed=42)

        # Valid indices should work
        dataset[0]
        dataset[n_scenes - 1]

        # Invalid indices should raise IndexError
        with pytest.raises(IndexError):
            dataset[n_scenes]

        # Negative indices should work (Python indexing)
        dataset[-1]  # This should work and return the last element

    def test_dataset_target_distribution(self) -> None:
        """Test that target indices are reasonably distributed."""
        n_scenes = 100
        k = 4
        dataset = ReferentialGameDataset(n_scenes, k, seed=42)

        target_indices = [dataset[i][1] for i in range(n_scenes)]

        # All targets should be valid
        assert all(0 <= idx < k for idx in target_indices)

        # Should have some variety in target selection
        unique_targets = set(target_indices)
        assert len(unique_targets) > 1  # Not all targets should be the same

    def test_dataset_tensor_types(self) -> None:
        """Test that dataset returns correct tensor types."""
        n_scenes = 2
        k = 3
        dataset = ReferentialGameDataset(n_scenes, k, seed=42)

        scene_tensor, target_idx, candidates = dataset[0]

        assert isinstance(scene_tensor, torch.Tensor)
        assert isinstance(candidates, torch.Tensor)
        assert isinstance(target_idx, int)

        # Tensors should be float32
        assert scene_tensor.dtype == torch.float32
        assert candidates.dtype == torch.float32
