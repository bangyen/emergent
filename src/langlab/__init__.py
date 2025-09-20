"""Language Emergence Lab - studying proto-language in referential games.

This package provides tools for studying how proto-language emerges in referential
games between agents, focusing on object reference and communication patterns.
"""

__version__ = "0.1.0"
__author__ = "Language Emergence Lab"

from .data.world import make_object, sample_scene, encode_object
from .data.data import ReferentialGameDataset
from .utils.utils import set_seed, get_device, get_logger
from .core.config import CommunicationConfig
from .core.agents import Speaker, Listener
from .core.channel import DiscreteChannel

__all__ = [
    "make_object",
    "sample_scene",
    "encode_object",
    "ReferentialGameDataset",
    "set_seed",
    "get_device",
    "get_logger",
    "CommunicationConfig",
    "Speaker",
    "Listener",
    "DiscreteChannel",
]
