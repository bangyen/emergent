"""World and object definitions for referential games.

This module defines the objects and scenes used in referential games, providing
the foundation for studying proto-language emergence in multi-agent systems.
"""

import random
from typing import Dict, List, Tuple, Optional

import torch

from ..utils.utils import set_seed


# Object attribute definitions
COLORS = ["red", "green", "blue"]
SHAPES = ["circle", "square", "triangle"]
SIZES = ["small", "large"]

# One-hot encoding dimensions
N_COLORS = len(COLORS)
N_SHAPES = len(SHAPES)
N_SIZES = len(SIZES)
TOTAL_ATTRIBUTES = N_COLORS + N_SHAPES + N_SIZES


def make_object(color: str, shape: str, size: str) -> Dict[str, str]:
    """Create an object with specified attributes.

    This function creates a dictionary representing an object with the given
    color, shape, and size attributes. Used for constructing scenes in
    referential games.

    Args:
        color: The color of the object (must be in COLORS).
        shape: The shape of the object (must be in SHAPES).
        size: The size of the object (must be in SIZES).

    Returns:
        A dictionary with 'color', 'shape', and 'size' keys.

    Raises:
        ValueError: If any attribute is not in the allowed values.
    """
    if color not in COLORS:
        raise ValueError(f"Invalid color '{color}'. Must be one of {COLORS}")
    if shape not in SHAPES:
        raise ValueError(f"Invalid shape '{shape}'. Must be one of {SHAPES}")
    if size not in SIZES:
        raise ValueError(f"Invalid size '{size}'. Must be one of {SIZES}")

    return {"color": color, "shape": shape, "size": size}


def sample_scene(
    k: int, seed: Optional[int] = None
) -> Tuple[List[Dict[str, str]], int]:
    """Generate a scene with K distinct objects and select a target.

    This function creates a scene containing K unique objects by sampling
    from all possible combinations of attributes. A target object is randomly
    selected from the scene for referential game tasks.

    Args:
        k: Number of objects in the scene (must be <= total possible objects).
        seed: Random seed for reproducible scene generation.

    Returns:
        A tuple containing:
        - List of K unique object dictionaries
        - Index of the target object in the scene

    Raises:
        ValueError: If k exceeds the total number of possible unique objects.
    """
    if seed is not None:
        set_seed(seed)

    total_objects = N_COLORS * N_SHAPES * N_SIZES
    if k > total_objects:
        raise ValueError(
            f"Cannot create {k} unique objects. Maximum is {total_objects}"
        )

    # Generate all possible objects
    all_objects = []
    for color in COLORS:
        for shape in SHAPES:
            for size in SIZES:
                all_objects.append(make_object(color, shape, size))

    # Sample K unique objects
    scene_objects = random.sample(all_objects, k)

    # Select target index
    target_idx = random.randint(0, k - 1)

    return scene_objects, target_idx


def sample_distractor_scene(
    k: int, num_distractors: int, seed: Optional[int] = None
) -> Tuple[List[Dict[str, str]], int]:
    """Generate a distractor-heavy scene for pragmatic inference.

    This function creates a scene with K objects where some objects share
    attributes with the target, making literal interpretation ambiguous.
    The distractors are chosen to create pragmatic challenges.

    Args:
        k: Number of objects in the scene (must be <= total possible objects).
        num_distractors: Number of distractor objects that share attributes with target.
        seed: Random seed for reproducible scene generation.

    Returns:
        A tuple containing:
        - List of K unique object dictionaries
        - Index of the target object in the scene

    Raises:
        ValueError: If k exceeds the total number of possible unique objects or num_distractors >= k.
    """
    if seed is not None:
        set_seed(seed)

    total_objects = N_COLORS * N_SHAPES * N_SIZES
    if k > total_objects:
        raise ValueError(
            f"Cannot create {k} unique objects. Maximum is {total_objects}"
        )
    if num_distractors >= k:
        raise ValueError(
            f"Number of distractors ({num_distractors}) must be less than scene size ({k})"
        )

    # Generate all possible objects
    all_objects = []
    for color in COLORS:
        for shape in SHAPES:
            for size in SIZES:
                all_objects.append(make_object(color, shape, size))

    # First, sample the target object
    target_obj = random.choice(all_objects)
    scene_objects = [target_obj]
    target_idx = 0

    # Add distractors that share attributes with the target
    distractor_candidates = []
    for obj in all_objects:
        if obj != target_obj:
            # Check if object shares at least one attribute with target
            shared_attrs = sum(
                1
                for attr in ["color", "shape", "size"]
                if obj[attr] == target_obj[attr]
            )
            if shared_attrs >= 1:
                distractor_candidates.append(obj)

    # Sample distractors
    if len(distractor_candidates) >= num_distractors:
        distractors = random.sample(distractor_candidates, num_distractors)
    else:
        # If not enough distractors, fill with random objects
        remaining_objects = [obj for obj in all_objects if obj not in scene_objects]
        distractors = distractor_candidates + random.sample(
            remaining_objects,
            min(num_distractors - len(distractor_candidates), len(remaining_objects)),
        )

    scene_objects.extend(distractors)

    # Fill remaining slots with random objects
    remaining_slots = k - len(scene_objects)
    if remaining_slots > 0:
        remaining_objects = [obj for obj in all_objects if obj not in scene_objects]
        if len(remaining_objects) >= remaining_slots:
            additional_objects = random.sample(remaining_objects, remaining_slots)
            scene_objects.extend(additional_objects)
        else:
            scene_objects.extend(remaining_objects)

    # Shuffle the scene (except target stays at index 0)
    non_target_objects = scene_objects[1:]
    random.shuffle(non_target_objects)
    scene_objects = [target_obj] + non_target_objects

    return scene_objects, target_idx


def encode_object(obj: Dict[str, str]) -> torch.Tensor:
    """Encode an object as a one-hot tensor.

    This function converts an object dictionary into a one-hot encoded tensor
    suitable for neural network processing. The encoding concatenates one-hot
    vectors for color, shape, and size attributes.

    Args:
        obj: Object dictionary with 'color', 'shape', and 'size' keys.

    Returns:
        A one-hot encoded tensor of shape (TOTAL_ATTRIBUTES,).
    """
    encoding = torch.zeros(TOTAL_ATTRIBUTES, dtype=torch.float32)

    # Encode color
    color_idx = COLORS.index(obj["color"])
    encoding[color_idx] = 1.0

    # Encode shape
    shape_idx = SHAPES.index(obj["shape"])
    encoding[N_COLORS + shape_idx] = 1.0

    # Encode size
    size_idx = SIZES.index(obj["size"])
    encoding[N_COLORS + N_SHAPES + size_idx] = 1.0

    return encoding
