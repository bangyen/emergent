"""Tests for the world module.

This module contains tests for object creation, scene generation, and encoding
functionality in the Language Emergence Lab world module.
"""

import pytest
import torch

from langlab.data.world import (
    make_object,
    sample_scene,
    encode_object,
    COLORS,
    SHAPES,
    SIZES,
    TOTAL_ATTRIBUTES,
)


class TestMakeObject:
    """Test object creation functionality."""

    def test_make_object_valid(self) -> None:
        """Test creating objects with valid attributes."""
        obj = make_object("red", "circle", "small")
        assert obj["color"] == "red"
        assert obj["shape"] == "circle"
        assert obj["size"] == "small"

    def test_make_object_invalid_color(self) -> None:
        """Test that invalid colors raise ValueError."""
        with pytest.raises(ValueError, match="Invalid color"):
            make_object("purple", "circle", "small")

    def test_make_object_invalid_shape(self) -> None:
        """Test that invalid shapes raise ValueError."""
        with pytest.raises(ValueError, match="Invalid shape"):
            make_object("red", "hexagon", "small")

    def test_make_object_invalid_size(self) -> None:
        """Test that invalid sizes raise ValueError."""
        with pytest.raises(ValueError, match="Invalid size"):
            make_object("red", "circle", "medium")


class TestSampleScene:
    """Test scene generation functionality."""

    def test_sample_scene_reproducible(self) -> None:
        """Test that same seed produces identical scene and target."""
        seed = 42
        scene1, target1 = sample_scene(3, seed)
        scene2, target2 = sample_scene(3, seed)

        assert scene1 == scene2
        assert target1 == target2

    def test_scene_has_unique_objects(self) -> None:
        """Test that all objects in a scene are unique."""
        scene, _ = sample_scene(3, seed=123)

        # Convert to tuples for comparison
        object_tuples = [tuple(obj.values()) for obj in scene]
        assert len(object_tuples) == len(set(object_tuples))

    def test_sample_scene_k_too_large(self) -> None:
        """Test that requesting too many objects raises ValueError."""
        max_objects = len(COLORS) * len(SHAPES) * len(SIZES)
        with pytest.raises(ValueError):
            sample_scene(max_objects + 1, seed=42)

    def test_sample_scene_k_valid(self) -> None:
        """Test that valid k values work."""
        max_objects = len(COLORS) * len(SHAPES) * len(SIZES)
        scene, target_idx = sample_scene(max_objects, seed=42)
        assert len(scene) == max_objects
        assert 0 <= target_idx < max_objects

    def test_sample_scene_target_in_range(self) -> None:
        """Test that target index is always in valid range."""
        k = 5
        scene, target_idx = sample_scene(k, seed=42)
        assert 0 <= target_idx < k


class TestEncodeObject:
    """Test object encoding functionality."""

    def test_encode_object_shape(self) -> None:
        """Test that encoded objects have correct shape."""
        obj = make_object("red", "circle", "small")
        encoding = encode_object(obj)
        assert encoding.shape == (TOTAL_ATTRIBUTES,)

    def test_encode_object_one_hot(self) -> None:
        """Test that encoding is properly one-hot."""
        obj = make_object("red", "circle", "small")
        encoding = encode_object(obj)

        # Should have exactly 3 ones (one for each attribute)
        assert torch.sum(encoding) == 3.0

        # All values should be 0 or 1
        assert torch.all((encoding == 0.0) | (encoding == 1.0))

    def test_encode_object_correct_indices(self) -> None:
        """Test that encoding uses correct indices for each attribute."""
        obj = make_object("red", "circle", "small")
        encoding = encode_object(obj)

        # Check color encoding
        red_idx = COLORS.index("red")
        assert encoding[red_idx] == 1.0

        # Check shape encoding
        circle_idx = SHAPES.index("circle")
        shape_start = len(COLORS)
        assert encoding[shape_start + circle_idx] == 1.0

        # Check size encoding
        small_idx = SIZES.index("small")
        size_start = len(COLORS) + len(SHAPES)
        assert encoding[size_start + small_idx] == 1.0

    def test_encode_object_different_objects(self) -> None:
        """Test that different objects produce different encodings."""
        obj1 = make_object("red", "circle", "small")
        obj2 = make_object("blue", "square", "large")

        encoding1 = encode_object(obj1)
        encoding2 = encode_object(obj2)

        assert not torch.equal(encoding1, encoding2)

    def test_encode_object_same_objects(self) -> None:
        """Test that identical objects produce identical encodings."""
        obj1 = make_object("red", "circle", "small")
        obj2 = make_object("red", "circle", "small")

        encoding1 = encode_object(obj1)
        encoding2 = encode_object(obj2)

        assert torch.equal(encoding1, encoding2)
