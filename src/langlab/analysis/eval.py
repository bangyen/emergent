"""Simplified evaluation module for emergent language experiments.

This module provides essential evaluation functionality for referential games.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from ..core.agents import Listener, ListenerSeq, Speaker, SpeakerSeq
from ..core.config import CommunicationConfig
from ..data.data import ReferentialGameDataset, make_compositional_splits
from ..utils.utils import get_device, get_logger

logger = get_logger(__name__)


def build_agents(config: CommunicationConfig, device: torch.device) -> Tuple[Any, Any]:
    """Instantiate the Speaker/Listener pair that matches ``config``."""
    if getattr(config, "use_sequence_models", False):
        return SpeakerSeq(config).to(device), ListenerSeq(config).to(device)
    return Speaker(config).to(device), Listener(config).to(device)


def accuracy(
    speaker: Any,
    listener: Any,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """Greedy referential accuracy of a Speaker/Listener pair on ``dataset``.

    Agents are put in eval mode for the pass and restored to their previous mode.
    """
    was_training = speaker.training, listener.training
    speaker.eval()
    listener.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for scene, targets in DataLoader(dataset, batch_size=batch_size):
            scene, targets = scene.to(device), targets.to(device)
            target_objs = scene[torch.arange(scene.size(0)), targets]
            tokens = speaker(target_objs).tokens
            preds = listener(tokens, scene).preds
            correct += int((preds == targets).sum().item())
            total += targets.numel()

    speaker.train(was_training[0])
    listener.train(was_training[1])
    return correct / total if total else 0.0


def evaluate(
    model_path: str,
    split: str = "train",
    heldout_pairs: Optional[List[Tuple[str, str]]] = None,
    n_scenes: int = 1000,
    k: Optional[int] = None,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evaluate a trained model on a specific data split.

    ``k`` and ``heldout_pairs`` default to the values stored in the checkpoint.
    """
    device = get_device()

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if k is None:
        k = checkpoint.get("k", 5)
    if heldout_pairs is None:
        heldout_pairs = checkpoint.get("heldout_pairs")

    speaker, listener = build_agents(config, device)
    speaker.load_state_dict(checkpoint["speaker_state_dict"])
    listener.load_state_dict(checkpoint["listener_state_dict"])

    dataset: Any
    if split == "train":
        dataset = ReferentialGameDataset(n_scenes, k, seed=7)
    elif split in ["iid", "compo"]:
        if not heldout_pairs:
            raise ValueError("heldout_pairs must be provided for compositional splits")
        dataset = make_compositional_splits(n_scenes, k, heldout_pairs, seed=7)[split]
    else:
        raise ValueError(f"Unsupported split: {split}")

    return {"acc": accuracy(speaker, listener, dataset, device, batch_size)}
