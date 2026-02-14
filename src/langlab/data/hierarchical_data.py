"""Hierarchical data generation for studying deep compositionality.

This module implements datasets with nested attributes and spatial relations
to push emergent language beyond simple flat attribute mapping.
"""

import torch
import random
from typing import Dict, Tuple, Optional
from torch.utils.data import Dataset

from .world import COLORS, SHAPES, SIZES, encode_object
from ..utils.utils import set_seed

# Additional relational attributes
RELATIONS = ["left-of", "right-of", "above", "below", "centered"]


def encode_hierarchical_object(obj: Dict, relations: Dict[str, str]) -> torch.Tensor:
    """Encode an object including its spatial relations.

    Args:
        obj: Base object dictionary (color, shape, size).
        relations: Dictionary of relations (e.g., {'spatial': 'above'}).

    Returns:
        Tensor encoding all attributes including relations.
    """
    base_encoding = encode_object(obj)

    # Encode relations
    rel_encoding = torch.zeros(len(RELATIONS))
    if "spatial" in relations and relations["spatial"] in RELATIONS:
        rel_idx = RELATIONS.index(relations["spatial"])
        rel_encoding[rel_idx] = 1.0

    return torch.cat([base_encoding, rel_encoding])


class HierarchicalReferentialDataset(Dataset):
    """Dataset with nested attributes and spatial relations.

    Each scene consists of objects that may have relative properties
    (e.g., being the 'largest' or 'to the left of' another object).
    """

    def __init__(self, n_scenes: int, k: int, seed: Optional[int] = None):
        self.n_scenes = n_scenes
        self.k = k
        self.seed = seed
        self._generate_scenes()

    def _generate_scenes(self) -> None:
        if self.seed is not None:
            set_seed(self.seed)

        self.scenes = []
        self.target_indices = []

        for i in range(self.n_scenes):
            scene_objects = []
            for _ in range(self.k):
                obj = {
                    "color": random.choice(COLORS),
                    "shape": random.choice(SHAPES),
                    "size": random.choice(SIZES),
                }
                scene_objects.append(obj)

            target_idx = random.randint(0, self.k - 1)

            # Assign spatial relations relative to a "focus" point or other objects
            # For simplicity, we assign a spatial tag to each object based on its
            # hypothetical position in a grid.
            encoded_scene = []
            for j in range(self.k):
                # Pseudo-spatial relation: random assignment for this experiment
                spatial_rel = random.choice(RELATIONS)
                encoded_obj = encode_hierarchical_object(
                    scene_objects[j], {"spatial": spatial_rel}
                )
                encoded_scene.append(encoded_obj)

            self.scenes.append(torch.stack(encoded_scene))
            self.target_indices.append(target_idx)

    def __len__(self) -> int:
        return self.n_scenes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        scene_tensor = self.scenes[idx]
        target_idx = self.target_indices[idx]
        return scene_tensor, target_idx, scene_tensor
