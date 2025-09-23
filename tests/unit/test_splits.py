"""Tests for compositional data splits.

This module contains tests for the make_compositional_splits function and
CompositionalDataset class, ensuring proper separation of held-out combinations.
"""

import pytest
import torch

from src.langlab.data.data import make_compositional_splits, CompositionalDataset
from src.langlab.data.world import COLORS, SHAPES, SIZES


class TestCompositionalSplits:
    """Test compositional splits functionality."""

    def test_heldout_not_in_train(self) -> None:
        """Verify no training scene includes held-out pairs."""
        n_scenes = 100
        k = 3
        heldout_pairs = [("blue", "triangle")]

        splits = make_compositional_splits(n_scenes, k, heldout_pairs, seed=42)

        train_dataset = splits["train"]

        # Check that no training scene contains blue triangle objects
        for i in range(len(train_dataset)):
            scene_tensor, _, _ = train_dataset[i]

            # Decode objects from tensor to check attributes
            for obj_idx in range(k):
                obj_encoding = scene_tensor[obj_idx]

                # Check if this object is blue triangle
                is_blue = obj_encoding[COLORS.index("blue")].item() == 1.0
                is_triangle = (
                    obj_encoding[len(COLORS) + SHAPES.index("triangle")].item() == 1.0
                )

                # Training set should not contain blue triangle objects
                assert not (
                    is_blue and is_triangle
                ), f"Found blue triangle in training scene {i}, object {obj_idx}"

    def test_compositional_split_sizes(self) -> None:
        """Test that splits have reasonable sizes."""
        n_scenes = 100
        k = 3
        heldout_pairs = [("blue", "triangle")]

        splits = make_compositional_splits(n_scenes, k, heldout_pairs, seed=42)

        train_size = len(splits["train"])
        iid_size = len(splits["iid"])
        compo_size = len(splits["compo"])

        # All splits should have some data
        assert train_size > 0, "Training set should not be empty"
        assert iid_size > 0, "IID test set should not be empty"
        assert compo_size > 0, "Compositional test set should not be empty"

        # Total should be approximately n_scenes
        total_size = train_size + iid_size + compo_size
        assert (
            total_size <= n_scenes
        ), f"Total size {total_size} exceeds requested {n_scenes}"

    def test_compositional_split_reproducibility(self) -> None:
        """Test that splits are reproducible with same seed."""
        n_scenes = 50
        k = 3
        heldout_pairs = [("blue", "triangle")]
        seed = 123

        splits1 = make_compositional_splits(n_scenes, k, heldout_pairs, seed)
        splits2 = make_compositional_splits(n_scenes, k, heldout_pairs, seed)

        # Check that splits are identical
        for split_name in ["train", "iid", "compo"]:
            dataset1 = splits1[split_name]
            dataset2 = splits2[split_name]

            assert len(dataset1) == len(dataset2), f"Split {split_name} sizes differ"

            for i in range(len(dataset1)):
                scene1, target1, _ = dataset1[i]
                scene2, target2, _ = dataset2[i]

                assert torch.equal(
                    scene1, scene2
                ), f"Split {split_name} scene {i} differs"
                assert target1 == target2, f"Split {split_name} target {i} differs"

    def test_compositional_split_different_seeds(self) -> None:
        """Test that different seeds produce different splits."""
        n_scenes = 50
        k = 3
        heldout_pairs = [("blue", "triangle")]

        splits1 = make_compositional_splits(n_scenes, k, heldout_pairs, seed=123)
        splits2 = make_compositional_splits(n_scenes, k, heldout_pairs, seed=456)

        # At least some splits should be different
        splits_different = False
        for split_name in ["train", "iid", "compo"]:
            dataset1 = splits1[split_name]
            dataset2 = splits2[split_name]

            if len(dataset1) != len(dataset2):
                splits_different = True
                break

            for i in range(min(len(dataset1), len(dataset2))):
                scene1, _, _ = dataset1[i]
                scene2, _, _ = dataset2[i]
                if not torch.equal(scene1, scene2):
                    splits_different = True
                    break

            if splits_different:
                break

        assert splits_different, "Different seeds should produce different splits"

    def test_compositional_split_contains_heldout(self) -> None:
        """Test that compositional split contains held-out combinations."""
        n_scenes = 100
        k = 3
        heldout_pairs = [("blue", "triangle")]

        splits = make_compositional_splits(n_scenes, k, heldout_pairs, seed=42)
        compo_dataset = splits["compo"]

        # Check that compositional split contains blue triangle objects
        found_heldout = False
        for i in range(len(compo_dataset)):
            scene_tensor, _, _ = compo_dataset[i]

            for obj_idx in range(k):
                obj_encoding = scene_tensor[obj_idx]

                # Check if this object is blue triangle
                is_blue = obj_encoding[COLORS.index("blue")].item() == 1.0
                is_triangle = (
                    obj_encoding[len(COLORS) + SHAPES.index("triangle")].item() == 1.0
                )

                if is_blue and is_triangle:
                    found_heldout = True
                    break

            if found_heldout:
                break

        assert found_heldout, "Compositional split should contain held-out combinations"


class TestCompositionalDataset:
    """Test CompositionalDataset class."""

    def test_compositional_dataset_basic(self) -> None:
        """Test basic CompositionalDataset functionality."""
        # Create sample scenes
        scenes = [
            [{"color": "red", "shape": "circle", "size": "small"}],
            [{"color": "blue", "shape": "square", "size": "large"}],
        ]
        targets = [0, 0]

        dataset = CompositionalDataset(scenes, targets)

        assert len(dataset) == 2

        # Test indexing
        scene_tensor, target_idx, candidates = dataset[0]
        assert scene_tensor.shape == (1, len(COLORS) + len(SHAPES) + len(SIZES))
        assert target_idx == 0
        assert torch.equal(scene_tensor, candidates)

    def test_compositional_dataset_iteration(self) -> None:
        """Test CompositionalDataset iteration."""
        scenes = [
            [{"color": "red", "shape": "circle", "size": "small"}],
            [{"color": "blue", "shape": "square", "size": "large"}],
        ]
        targets = [0, 0]

        dataset = CompositionalDataset(scenes, targets)

        scenes_list = list(dataset)
        assert len(scenes_list) == 2

        # Check that iteration produces same results as indexing
        for i, (scene_tensor, target_idx, candidates) in enumerate(scenes_list):
            expected_scene, expected_target, expected_candidates = dataset[i]
            assert torch.equal(scene_tensor, expected_scene)
            assert target_idx == expected_target
            assert torch.equal(candidates, expected_candidates)

    def test_compositional_dataset_index_bounds(self) -> None:
        """Test CompositionalDataset index bounds checking."""
        scenes = [[{"color": "red", "shape": "circle", "size": "small"}]]
        targets = [0]

        dataset = CompositionalDataset(scenes, targets)

        # Valid index should work
        dataset[0]

        # Invalid index should raise IndexError
        with pytest.raises(IndexError):
            dataset[1]
