"""Dataset classes for referential games.

This module provides dataset classes for generating and managing data for
referential game experiments, enabling systematic study of proto-language emergence.
"""

import random
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset

from .world import (
    DEFAULT_WORLD,
    Object,
    ObjectKey,
    World,
    sample_distractor_scene,
    sample_scene,
)
from ..utils.utils import set_seed

Sample = Tuple[torch.Tensor, int]


def object_key(obj: Dict[str, str]) -> ObjectKey:
    """Order-independent hashable key for an object."""
    return World.key(obj)


def heldout_objects(
    heldout_pairs: Sequence[Tuple[str, str]], world: World = DEFAULT_WORLD
) -> Set[ObjectKey]:
    """Return keys of every object carrying both attributes of any held-out pair."""
    return world.heldout_keys(heldout_pairs)


def encode_scene(scene: Sequence[Object], world: World = DEFAULT_WORLD) -> torch.Tensor:
    return torch.stack([world.encode(obj) for obj in scene])


class SceneListDataset(Dataset):
    """Dataset over a fixed list of pre-generated scenes."""

    def __init__(
        self,
        scenes: List[List[Object]],
        targets: List[int],
        world: World = DEFAULT_WORLD,
    ):
        self.scenes = scenes
        self.targets = targets
        self.encoded_scenes = [encode_scene(s, world) for s in scenes]

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Sample:
        if idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )
        return self.encoded_scenes[idx], self.targets[idx]

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]


# Kept under its historical name for callers of make_compositional_splits.
CompositionalDataset = SceneListDataset


class ReferentialGameDataset(SceneListDataset):
    """Fixed set of ``n_scenes`` random scenes with K objects each.

    Scene ``i`` is drawn with seed ``seed + i`` (reseeding global RNGs), so
    datasets are reproducible and prefixes of one another.
    """

    def __init__(
        self,
        n_scenes: int,
        k: int,
        seed: Optional[int] = None,
        world: World = DEFAULT_WORLD,
    ):
        self.n_scenes = n_scenes
        self.k = k
        self.seed = seed
        if seed is not None:
            set_seed(seed)
        scenes, targets = [], []
        for i in range(n_scenes):
            scene_seed = seed + i if seed is not None else None
            scene, target = sample_scene(k, scene_seed, world=world)
            scenes.append(scene)
            targets.append(target)
        super().__init__(scenes, targets, world)


class DistractorDataset(SceneListDataset):
    """Scenes where ``num_distractors`` objects share attributes with the target.

    The target is placed at a random index so position carries no signal.
    """

    def __init__(
        self,
        n_scenes: int,
        k: int,
        num_distractors: int,
        seed: Optional[int] = None,
        world: World = DEFAULT_WORLD,
    ):
        self.n_scenes = n_scenes
        self.k = k
        self.num_distractors = num_distractors
        self.seed = seed
        rng = random.Random(seed)
        scenes, targets = [], []
        for _ in range(n_scenes):
            scene, target = sample_distractor_scene(
                k, num_distractors, rng=rng, world=world, shuffle_target=True
            )
            scenes.append(scene)
            targets.append(target)
        super().__init__(scenes, targets, world)


class SceneStream(IterableDataset):
    """Endless stream of freshly sampled scenes, generated on the fly.

    Uses a private RNG so iterating does not disturb global random state, and
    skips any scene containing a held-out object so those combinations are
    never seen in training.

    Args:
        k: Number of objects per scene.
        seed: Seed for the stream's private RNG.
        heldout_pairs: Attribute pairs whose objects must never appear.
        world: World to draw objects from.
        hard_distractors: How many of the k - 1 distractors must share at least
            one attribute with the target (0 = uniformly random scenes). Hard
            distractors force the speaker to describe more than one attribute.
    """

    def __init__(
        self,
        k: int,
        seed: Optional[int] = None,
        heldout_pairs: Optional[Sequence[Tuple[str, str]]] = None,
        world: World = DEFAULT_WORLD,
        hard_distractors: int = 0,
    ):
        if not 0 <= hard_distractors < k:
            raise ValueError("hard_distractors must be in [0, k)")
        self.k = k
        self.seed = seed
        self.world = world
        self.hard_distractors = hard_distractors
        self.excluded = world.heldout_keys(heldout_pairs or [])
        if world.n_objects - len(self.excluded) < k:
            raise ValueError("Too many held-out objects to fill a scene")
        self._allowed_objects = [
            o for o in world.objects if World.key(o) not in self.excluded
        ]

    def __iter__(self) -> Iterator[Sample]:
        rng = random.Random(self.seed)
        objects = self._allowed_objects
        encodings = [self.world.encode(o) for o in objects]
        if not self.hard_distractors:
            while True:
                idx = rng.sample(range(len(objects)), self.k)
                target = rng.randint(0, self.k - 1)
                yield torch.stack([encodings[i] for i in idx]), target

        names = self.world.names
        similar = [
            [
                j
                for j, other in enumerate(objects)
                if j != i and any(other[n] == obj[n] for n in names)
            ]
            for i, obj in enumerate(objects)
        ]
        while True:
            t = rng.randrange(len(objects))
            hard = rng.sample(similar[t], min(self.hard_distractors, len(similar[t])))
            rest = [j for j in range(len(objects)) if j != t and j not in hard]
            others = hard + rng.sample(rest, self.k - 1 - len(hard))
            rng.shuffle(others)
            target = rng.randint(0, self.k - 1)
            others.insert(target, t)
            yield torch.stack([encodings[i] for i in others]), target


def make_compositional_splits(
    n_scenes: int,
    k: int,
    heldout_pairs: List[Tuple[str, str]],
    seed: Optional[int] = None,
    world: World = DEFAULT_WORLD,
) -> Dict[str, SceneListDataset]:
    """Create compositional splits for testing generalization.

    Returns:
        Dictionary with datasets (60/20/20 of ``n_scenes``):
        - 'train': scenes without any held-out object
        - 'iid': more scenes without any held-out object
        - 'compo': scenes containing at least one held-out object
    """
    if seed is not None:
        set_seed(seed)
    excluded = world.heldout_keys(heldout_pairs)

    sizes = {"train": int(n_scenes * 0.6), "iid": int(n_scenes * 0.2)}
    sizes["compo"] = n_scenes - sizes["train"] - sizes["iid"]
    scenes: Dict[str, List[List[Object]]] = {name: [] for name in sizes}
    targets: Dict[str, List[int]] = {name: [] for name in sizes}

    def full(name: str) -> bool:
        return len(scenes[name]) >= sizes[name]

    attempt = 0
    while attempt < n_scenes * 10 and not all(full(n) for n in sizes):
        scene, target = sample_scene(
            k, seed + attempt if seed is not None else None, world=world
        )
        attempt += 1
        if any(World.key(obj) in excluded for obj in scene):
            name = "compo"
        else:
            name = "train" if not full("train") else "iid"
        if not full(name):
            scenes[name].append(scene)
            targets[name].append(target)

    return {n: SceneListDataset(scenes[n], targets[n], world) for n in sizes}


def make_heldout_target_dataset(
    n_scenes: int,
    k: int,
    heldout_pairs: Sequence[Tuple[str, str]],
    seed: Optional[int] = None,
    world: World = DEFAULT_WORLD,
) -> SceneListDataset:
    """Scenes whose *target* is a held-out object (the strict compositional test).

    Distractors are drawn from the remaining objects and the target is placed at a
    random index.
    """
    excluded = world.heldout_keys(heldout_pairs)
    if not excluded:
        raise ValueError("heldout_pairs selects no objects")
    heldout = [o for o in world.objects if World.key(o) in excluded]
    rng = random.Random(seed)
    scenes, targets = [], []
    for _ in range(n_scenes):
        target_obj = rng.choice(heldout)
        others = rng.sample([o for o in world.objects if o != target_obj], k - 1)
        target = rng.randint(0, k - 1)
        scene = [dict(o) for o in others]
        scene.insert(target, dict(target_obj))
        scenes.append(scene)
        targets.append(target)
    return SceneListDataset(scenes, targets, world)
