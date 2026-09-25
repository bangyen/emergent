"""World and object definitions for referential games.

A :class:`World` is an ordered set of attributes, each with a list of values.
Objects are dicts mapping attribute name to value and are one-hot encoded by
concatenating one block per attribute. The module-level helpers operate on the
default 3-colour x 3-shape x 2-size world unless a ``world`` is passed.
"""

import itertools
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from ..utils.utils import set_seed

Object = Dict[str, str]
ObjectKey = Tuple[str, ...]


class World:
    """An attribute space from which objects and scenes are drawn.

    Args:
        attributes: Ordered mapping of attribute name to its possible values.
            Values must be unique across all attributes, so that held-out pairs
            like ``("red", "circle")`` are unambiguous.
    """

    def __init__(self, attributes: Mapping[str, Sequence[str]]):
        self.attributes: Dict[str, List[str]] = {
            name: list(values) for name, values in attributes.items()
        }
        all_values = [v for vs in self.attributes.values() for v in vs]
        if len(all_values) != len(set(all_values)):
            raise ValueError("Attribute values must be unique across attributes")
        if any(not vs for vs in self.attributes.values()):
            raise ValueError("Every attribute needs at least one value")

        self.names = list(self.attributes)
        self.dim = len(all_values)
        self._offsets: Dict[str, int] = {}
        offset = 0
        for name, values in self.attributes.items():
            self._offsets[name] = offset
            offset += len(values)
        self.objects: List[Object] = [
            dict(zip(self.names, combo))
            for combo in itertools.product(*self.attributes.values())
        ]
        self._encodings = {self.key(obj): self._encode(obj) for obj in self.objects}

    def __repr__(self) -> str:
        sizes = "x".join(str(len(v)) for v in self.attributes.values())
        return f"World({sizes}: {', '.join(self.names)})"

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @staticmethod
    def key(obj: Mapping[str, str]) -> ObjectKey:
        """Order-independent hashable key for an object."""
        return tuple(sorted(obj.values()))

    def validate(self, obj: Mapping[str, str]) -> None:
        for name, values in self.attributes.items():
            if obj.get(name) not in values:
                raise ValueError(
                    f"Invalid {name} '{obj.get(name)}'. Must be one of {values}"
                )

    def _encode(self, obj: Mapping[str, str]) -> torch.Tensor:
        encoding = torch.zeros(self.dim, dtype=torch.float32)
        for name, values in self.attributes.items():
            encoding[self._offsets[name] + values.index(obj[name])] = 1.0
        return encoding

    def encode(self, obj: Mapping[str, str]) -> torch.Tensor:
        """One-hot encode an object as a tensor of shape ``(dim,)``."""
        cached = self._encodings.get(self.key(obj))
        if cached is None:
            self.validate(obj)
            cached = self._encode(obj)
        return cached.clone()

    def attribute_indices(self, obj: Mapping[str, str]) -> Tuple[int, ...]:
        """Value index of each attribute, in attribute order."""
        return tuple(vs.index(obj[n]) for n, vs in self.attributes.items())

    def heldout_keys(self, heldout_pairs: Sequence[Tuple[str, str]]) -> Set[ObjectKey]:
        """Keys of every object carrying both values of any held-out pair."""
        known = {v for vs in self.attributes.values() for v in vs}
        for pair in heldout_pairs:
            unknown = [v for v in pair if v not in known]
            if unknown:
                raise ValueError(
                    f"Unknown attribute(s) {unknown}; expected one of {sorted(known)}"
                )
        return {
            self.key(obj)
            for obj in self.objects
            if any(a in obj.values() and b in obj.values() for a, b in heldout_pairs)
        }


DEFAULT_WORLD = World(
    {
        "color": ["red", "green", "blue"],
        "shape": ["circle", "square", "triangle"],
        "size": ["small", "large"],
    }
)

WORLDS: Dict[str, World] = {
    "default": DEFAULT_WORLD,
    # 5 x 5 x 3 x 3 = 225 objects
    "large": World(
        {
            "color": ["red", "green", "blue", "yellow", "purple"],
            "shape": ["circle", "square", "triangle", "star", "hexagon"],
            "size": ["small", "medium", "large"],
            "texture": ["plain", "striped", "dotted"],
        }
    ),
}


def get_world(name: str) -> World:
    try:
        return WORLDS[name]
    except KeyError:
        raise ValueError(f"Unknown world '{name}'; choose from {sorted(WORLDS)}")


# Default-world constants kept for backwards compatibility
COLORS = DEFAULT_WORLD.attributes["color"]
SHAPES = DEFAULT_WORLD.attributes["shape"]
SIZES = DEFAULT_WORLD.attributes["size"]
N_COLORS = len(COLORS)
N_SHAPES = len(SHAPES)
N_SIZES = len(SIZES)
TOTAL_ATTRIBUTES = DEFAULT_WORLD.dim


def make_object(color: str, shape: str, size: str) -> Object:
    """Create a default-world object, validating each attribute.

    Raises:
        ValueError: If any attribute is not in the allowed values.
    """
    obj = {"color": color, "shape": shape, "size": size}
    DEFAULT_WORLD.validate(obj)
    return obj


def _sampler(seed: Optional[int], rng: Optional[random.Random]) -> Any:
    if rng is not None:
        return rng
    if seed is not None:
        set_seed(seed)
    return random


def sample_scene(
    k: int,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
    world: World = DEFAULT_WORLD,
) -> Tuple[List[Object], int]:
    """Generate a scene with K distinct objects and select a target.

    Args:
        k: Number of objects in the scene (must be <= objects in the world).
        seed: Random seed for reproducible scene generation (reseeds global RNGs).
        rng: Optional local random generator; when given, ``seed`` is ignored and
            global RNG state is left untouched.
        world: World to draw objects from.

    Returns:
        The list of K unique objects and the index of the target.

    Raises:
        ValueError: If k exceeds the number of unique objects.
    """
    sampler = _sampler(seed, rng)
    if k > world.n_objects:
        raise ValueError(
            f"Cannot create {k} unique objects. Maximum is {world.n_objects}"
        )
    scene_objects = [dict(o) for o in sampler.sample(world.objects, k)]
    target_idx = sampler.randint(0, k - 1)
    return scene_objects, target_idx


def sample_distractor_scene(
    k: int,
    num_distractors: int,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
    world: World = DEFAULT_WORLD,
    shuffle_target: bool = False,
) -> Tuple[List[Object], int]:
    """Generate a scene where ``num_distractors`` objects share attributes with the target.

    Args:
        k: Number of objects in the scene.
        num_distractors: Objects sharing at least one attribute with the target.
        seed: Random seed (reseeds global RNGs).
        rng: Optional local random generator, overriding ``seed``.
        world: World to draw objects from.
        shuffle_target: Place the target at a random index instead of index 0.

    Raises:
        ValueError: If k is too large or num_distractors >= k.
    """
    sampler = _sampler(seed, rng)
    if k > world.n_objects:
        raise ValueError(
            f"Cannot create {k} unique objects. Maximum is {world.n_objects}"
        )
    if num_distractors >= k:
        raise ValueError(
            f"Number of distractors ({num_distractors}) must be less than scene size ({k})"
        )

    target_obj = dict(sampler.choice(world.objects))
    candidates = [
        o
        for o in world.objects
        if o != target_obj and any(o[n] == target_obj[n] for n in world.names)
    ]
    distractors = sampler.sample(candidates, min(num_distractors, len(candidates)))
    chosen = [target_obj] + [dict(o) for o in distractors]
    remaining = [o for o in world.objects if o not in chosen]
    chosen += [dict(o) for o in sampler.sample(remaining, k - len(chosen))]

    others = chosen[1:]
    sampler.shuffle(others)
    if shuffle_target:
        target_idx = sampler.randint(0, k - 1)
        others.insert(target_idx, target_obj)
        return others, target_idx
    return [target_obj] + others, 0


def encode_object(obj: Mapping[str, str], world: World = DEFAULT_WORLD) -> torch.Tensor:
    """Encode an object as a one-hot tensor of shape ``(world.dim,)``."""
    return world.encode(obj)
