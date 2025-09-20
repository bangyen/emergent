"""Language Emergence Lab - studying proto-language in referential games.

This package provides tools for studying how proto-language emerges in referential
games between agents, focusing on object reference and communication patterns.
"""

__version__ = "0.1.0"
__author__ = "Language Emergence Lab"

from .world import make_object, sample_scene, encode_object
from .data import ReferentialGameDataset
from .utils import set_seed, get_device, get_logger
from .config import CommunicationConfig
from .agents import Speaker, Listener
from .channel import DiscreteChannel

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
