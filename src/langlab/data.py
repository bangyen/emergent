"""Dataset classes for referential games.

This module provides dataset classes for generating and managing data for
referential game experiments, enabling systematic study of proto-language emergence.
"""

from typing import Tuple, Iterator, Optional, List, Dict
import torch
from torch.utils.data import Dataset

from .world import (
    sample_scene,
    sample_distractor_scene,
    encode_object,
    COLORS,
    SHAPES,
    SIZES,
)
from .utils import set_seed


class ReferentialGameDataset(Dataset):
    """Dataset for referential game experiments.

    This dataset generates scenes with K objects and provides the necessary
    data for training agents in referential games. Each sample contains a
    scene tensor, target index, and candidate encodings for the referential task.
    """

    def __init__(self, n_scenes: int, k: int, seed: Optional[int] = None):
        """Initialize the referential game dataset.

        Args:
            n_scenes: Number of scenes to generate in the dataset.
            k: Number of objects per scene.
            seed: Random seed for reproducible dataset generation.
        """
        self.n_scenes = n_scenes
        self.k = k
        self.seed = seed

        # Generate all scenes upfront for efficiency
        self._generate_scenes()

    def _generate_scenes(self) -> None:
        """Generate all scenes for the dataset."""
        if self.seed is not None:
            set_seed(self.seed)

        self.scenes = []
        self.target_indices = []

        for i in range(self.n_scenes):
            # Use scene index as additional seed component for variety
            scene_seed = self.seed + i if self.seed is not None else None
            scene_objects, target_idx = sample_scene(self.k, scene_seed)

            # Encode scene as tensor
            scene_tensor = torch.stack([encode_object(obj) for obj in scene_objects])

            self.scenes.append(scene_tensor)
            self.target_indices.append(target_idx)

    def __len__(self) -> int:
        """Return the number of scenes in the dataset."""
        return self.n_scenes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Get a single scene from the dataset.

        Args:
            idx: Index of the scene to retrieve.

        Returns:
            A tuple containing:
            - scene_tensor: Tensor of shape (K, TOTAL_ATTRIBUTES) with encoded objects
            - target_idx: Index of the target object in the scene
            - candidate_encodings: Same as scene_tensor (for compatibility)
        """
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        scene_tensor = self.scenes[idx]
        target_idx = self.target_indices[idx]

        # For referential games, candidates are all objects in the scene
        candidate_encodings = scene_tensor

        return scene_tensor, target_idx, candidate_encodings

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int, torch.Tensor]]:
        """Iterate over all scenes in the dataset."""
        for i in range(len(self)):
            yield self[i]


class DistractorDataset(Dataset):
    """Dataset for distractor-heavy referential game experiments.

    This dataset generates scenes with distractor objects that share attributes
    with the target, creating pragmatic challenges for literal interpretation.
    """

    def __init__(
        self, n_scenes: int, k: int, num_distractors: int, seed: Optional[int] = None
    ):
        """Initialize the distractor dataset.

        Args:
            n_scenes: Number of scenes to generate in the dataset.
            k: Number of objects per scene.
            num_distractors: Number of distractor objects that share attributes with target.
            seed: Random seed for reproducible dataset generation.
        """
        self.n_scenes = n_scenes
        self.k = k
        self.num_distractors = num_distractors
        self.seed = seed

        # Generate all scenes upfront for efficiency
        self._generate_scenes()

    def _generate_scenes(self) -> None:
        """Generate all distractor scenes for the dataset."""
        if self.seed is not None:
            set_seed(self.seed)

        self.scenes = []
        self.target_indices = []

        for i in range(self.n_scenes):
            # Use scene index as additional seed component for variety
            scene_seed = self.seed + i if self.seed is not None else None
            scene_objects, target_idx = sample_distractor_scene(
                self.k, self.num_distractors, scene_seed
            )

            self.scenes.append(scene_objects)
            self.target_indices.append(target_idx)

    def __len__(self) -> int:
        """Return the number of scenes in the dataset."""
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing:
            - scene_tensor: Tensor of shape (k, object_dim) with encoded objects
            - target_idx: Index of the target object in the scene
            - candidate_encodings: Same as scene_tensor (for compatibility)
        """
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        scene_objects = self.scenes[idx]
        target_idx = self.target_indices[idx]

        # Encode all objects in the scene
        encoded_objects = [encode_object(obj) for obj in scene_objects]
        scene_tensor = torch.stack(encoded_objects)

        # For referential games, candidates are all objects in the scene
        candidate_encodings = scene_tensor

        return scene_tensor, target_idx, candidate_encodings

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int, torch.Tensor]]:
        """Iterate over all scenes in the dataset."""
        for i in range(len(self)):
            yield self[i]


def make_compositional_splits(
    n_scenes: int,
    k: int,
    heldout_pairs: List[Tuple[str, str]],
    seed: Optional[int] = None,
) -> Dict[str, "CompositionalDataset"]:
    """Create compositional splits for testing generalization.

    This function creates train/test splits where the training set excludes
    scenes containing specific held-out attribute combinations, enabling
    evaluation of compositional generalization capabilities.

    Args:
        n_scenes: Total number of scenes to generate.
        k: Number of objects per scene.
        heldout_pairs: List of (attribute1, attribute2) pairs to hold out from training.
        seed: Random seed for reproducible generation.

    Returns:
        Dictionary with keys 'train', 'iid', 'compo' containing datasets:
        - 'train': Training set excluding held-out combinations
        - 'iid': In-distribution test set (same distribution as train)
        - 'compo': Compositional test set containing held-out combinations
    """
    if seed is not None:
        set_seed(seed)

    # Generate all possible objects
    all_objects = []
    for color in COLORS:
        for shape in SHAPES:
            for size in SIZES:
                all_objects.append({"color": color, "shape": shape, "size": size})

    # Create held-out object set
    heldout_objects = set()
    for attr1, attr2 in heldout_pairs:
        # Find objects that contain both attributes
        for obj in all_objects:
            if attr1 in obj.values() and attr2 in obj.values():
                heldout_objects.add(tuple(sorted(obj.values())))

    # Generate scenes for each split
    train_scenes: List[List[Dict[str, str]]] = []
    iid_scenes: List[List[Dict[str, str]]] = []
    compo_scenes: List[List[Dict[str, str]]] = []

    train_targets: List[int] = []
    iid_targets: List[int] = []
    compo_targets: List[int] = []

    # Generate more scenes than needed to ensure we get enough for each split
    max_attempts = n_scenes * 3
    scene_count = 0

    while (
        scene_count < max_attempts
        and (len(train_scenes) + len(iid_scenes) + len(compo_scenes)) < n_scenes
    ):
        # Generate a scene
        scene_objects, target_idx = sample_scene(
            k, seed + scene_count if seed is not None else None
        )

        # Check if scene contains held-out combinations
        scene_combinations = set()
        for obj in scene_objects:
            obj_tuple = tuple(sorted(obj.values()))
            scene_combinations.add(obj_tuple)

        has_heldout = bool(scene_combinations.intersection(heldout_objects))

        # Assign to appropriate split
        if has_heldout:
            # Scene contains held-out combinations -> compositional test
            compo_scenes.append(scene_objects)
            compo_targets.append(target_idx)
        else:
            # Scene doesn't contain held-out combinations
            if len(train_scenes) < n_scenes * 0.6:  # 60% for training
                train_scenes.append(scene_objects)
                train_targets.append(target_idx)
            elif len(iid_scenes) < n_scenes * 0.2:  # 20% for iid test
                iid_scenes.append(scene_objects)
                iid_targets.append(target_idx)

        scene_count += 1

    # Create datasets
    train_dataset = CompositionalDataset(train_scenes, train_targets)
    iid_dataset = CompositionalDataset(iid_scenes, iid_targets)
    compo_dataset = CompositionalDataset(compo_scenes, compo_targets)

    return {"train": train_dataset, "iid": iid_dataset, "compo": compo_dataset}


class CompositionalDataset(Dataset):
    """Dataset for compositional splits with pre-generated scenes.

    This dataset class stores pre-generated scenes and targets, enabling
    precise control over train/test splits for compositional generalization.
    """

    def __init__(self, scenes: List[List[Dict[str, str]]], targets: List[int]):
        """Initialize compositional dataset.

        Args:
            scenes: List of scenes, where each scene is a list of object dictionaries.
            targets: List of target indices corresponding to each scene.
        """
        self.scenes = scenes
        self.targets = targets

        # Pre-encode all scenes for efficiency
        self.encoded_scenes = []
        for scene in scenes:
            scene_tensor = torch.stack([encode_object(obj) for obj in scene])
            self.encoded_scenes.append(scene_tensor)

    def __len__(self) -> int:
        """Return the number of scenes in the dataset."""
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Get a single scene from the dataset.

        Args:
            idx: Index of the scene to retrieve.

        Returns:
            A tuple containing:
            - scene_tensor: Tensor of shape (K, TOTAL_ATTRIBUTES) with encoded objects
            - target_idx: Index of the target object in the scene
            - candidate_encodings: Same as scene_tensor (for compatibility)
        """
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        scene_tensor = self.encoded_scenes[idx]
        target_idx = self.targets[idx]
        candidate_encodings = scene_tensor

        return scene_tensor, target_idx, candidate_encodings

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int, torch.Tensor]]:
        """Iterate over all scenes in the dataset."""
        for i in range(len(self)):
            yield self[i]
