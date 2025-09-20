"""Evaluation module for emergent language experiments.

This module provides evaluation functionality for testing compositional
generalization in referential games, enabling systematic assessment of
agent performance across different data splits.
"""

import json
import os
from typing import Dict, Optional, Union
import torch
from torch.utils.data import DataLoader

from .agents import Speaker, Listener, SpeakerSeq, ListenerSeq
from .data import ReferentialGameDataset, CompositionalDataset
from .utils import get_logger, get_device

logger = get_logger(__name__)


def evaluate(
    model_path: str,
    split: str = "train",
    heldout_pairs: Optional[list] = None,
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate model performance on specified data split.

    This function loads a trained model and evaluates its performance on
    the specified data split, returning accuracy metrics for compositional
    generalization assessment.

    Args:
        model_path: Path to the saved model checkpoint.
        split: Data split to evaluate on ("train", "iid", or "compo").
        heldout_pairs: List of held-out attribute pairs for compositional splits.
        n_scenes: Number of scenes to generate for evaluation.
        k: Number of objects per scene.
        batch_size: Batch size for evaluation.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing accuracy metrics.

    Raises:
        ValueError: If split is not supported or model loading fails.
    """
    if device is None:
        device = get_device()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Create agents
    if hasattr(config, "use_sequence_models") and config.use_sequence_models:
        speaker: Union[Speaker, SpeakerSeq] = SpeakerSeq(config).to(device)
        listener: Union[Listener, ListenerSeq] = ListenerSeq(config).to(device)
    else:
        speaker = Speaker(config).to(device)
        listener = Listener(config).to(device)

    # Load model states
    speaker.load_state_dict(checkpoint["speaker_state_dict"])
    listener.load_state_dict(checkpoint["listener_state_dict"])

    # Set to evaluation mode
    speaker.eval()
    listener.eval()

    # Create dataset based on split
    if split == "train":
        # Use regular dataset for training evaluation
        dataset: Union[ReferentialGameDataset, CompositionalDataset] = (
            ReferentialGameDataset(n_scenes, k, seed=42)
        )
    elif split in ["iid", "compo"]:
        # Use compositional splits
        if heldout_pairs is None:
            raise ValueError("heldout_pairs must be provided for compositional splits")

        from .data import make_compositional_splits

        splits = make_compositional_splits(n_scenes, k, heldout_pairs, seed=42)
        dataset = splits[split]
    else:
        raise ValueError(f"Unsupported split: {split}")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Evaluate
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            scene_tensor, target_indices, candidate_objects = batch

            # Move to device
            scene_tensor = scene_tensor.to(device)
            target_indices = target_indices.to(device)
            candidate_objects = candidate_objects.to(device)

            # Extract target objects
            batch_size = scene_tensor.size(0)
            target_objects = scene_tensor[torch.arange(batch_size), target_indices]

            # Speaker generates messages
            _, message_tokens = speaker(target_objects)

            # Listener makes predictions
            listener_probs = listener(message_tokens, candidate_objects)
            listener_predictions = torch.argmax(listener_probs, dim=1)

            # Count correct predictions
            correct = (listener_predictions == target_indices).sum().item()
            total_correct += correct
            total_samples += batch_size

    # Compute accuracy
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    logger.info(
        f"Evaluation on {split} split: {total_correct}/{total_samples} = {accuracy:.4f}"
    )

    return {"acc": accuracy}


def evaluate_all_splits(
    model_path: str,
    heldout_pairs: list,
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model on all splits and save results.

    This function evaluates a model on train, iid, and compositional splits,
    then saves the results to a JSON file for analysis.

    Args:
        model_path: Path to the saved model checkpoint.
        heldout_pairs: List of held-out attribute pairs for compositional splits.
        n_scenes: Number of scenes to generate for evaluation.
        k: Number of objects per scene.
        batch_size: Batch size for evaluation.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing results for all splits.
    """
    results = {}

    # Evaluate on each split
    for split in ["train", "iid", "compo"]:
        logger.info(f"Evaluating on {split} split...")
        results[split] = evaluate(
            model_path=model_path,
            split=split,
            heldout_pairs=heldout_pairs,
            n_scenes=n_scenes,
            k=k,
            batch_size=batch_size,
            device=device,
        )

    # Save results to JSON
    os.makedirs("outputs", exist_ok=True)
    metrics_path = "outputs/metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved to {metrics_path}")

    return results
