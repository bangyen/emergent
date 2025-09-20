"""Dataset classes for referential games.

This module provides dataset classes for generating and managing data for
referential game experiments, enabling systematic study of proto-language emergence.
"""

from typing import Tuple, Iterator, Optional
import torch
from torch.utils.data import Dataset

from .world import sample_scene, encode_object
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
