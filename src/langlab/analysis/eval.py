"""Simplified evaluation module for emergent language experiments.

This module provides essential evaluation functionality for referential games.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..core.agents import Listener, ListenerSeq, Speaker, SpeakerSeq
from ..data.data import ReferentialGameDataset
from ..utils.utils import get_device, get_logger

logger = get_logger(__name__)


def evaluate(
    model_path: str,
    split: str = "train",
    heldout_pairs: Optional[List[Tuple[str, str]]] = None,
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evaluate a trained model on a specific data split."""
    device = get_device()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Create agents based on checkpoint type
    speaker: Any
    listener: Any

    if hasattr(config, "use_sequence_models") and config.use_sequence_models:
        speaker = SpeakerSeq(config).to(device)
        listener = ListenerSeq(config).to(device)
    else:
        speaker = Speaker(config).to(device)
        listener = Listener(config).to(device)

    # Load model states
    speaker.load_state_dict(checkpoint["speaker_state_dict"])
    listener.load_state_dict(checkpoint["listener_state_dict"])

    speaker.eval()
    listener.eval()

    # Create dataset based on split
    dataset: Any
    if split == "train":
        dataset = ReferentialGameDataset(n_scenes, k, seed=7)
    elif split in ["iid", "compo"]:
        if heldout_pairs is None:
            raise ValueError("heldout_pairs must be provided for compositional splits")

        from ..data.data import make_compositional_splits

        splits = make_compositional_splits(n_scenes, k, heldout_pairs, seed=7)
        dataset = splits[split]
    else:
        raise ValueError(f"Unsupported split: {split}")

    dataloader = DataLoader(dataset, batch_size=batch_size)

    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            scene_tensor, target_indices = batch
            scene_tensor, target_indices = (
                scene_tensor.to(device),
                target_indices.to(device),
            )

            batch_size = scene_tensor.size(0)
            target_objects = scene_tensor[torch.arange(batch_size), target_indices]

            # Speaker
            speaker_output = speaker(target_objects)
            tokens = speaker_output.tokens

            # Listener
            listener_output = listener(tokens, scene_tensor)
            preds = listener_output.preds

            acc = (preds == target_indices).float().mean()
            total_acc += acc.item()
            n_batches += 1

    return {"acc": total_acc / n_batches if n_batches > 0 else 0.0}
