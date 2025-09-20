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

from ..core.agents import Speaker, Listener, SpeakerSeq, ListenerSeq, PragmaticListener
from ..data.data import ReferentialGameDataset, CompositionalDataset
from ..utils.utils import get_logger, get_device
from ..data.world import sample_scene, sample_distractor_scene, encode_object

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

        from ..data.data import make_compositional_splits

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
            _, message_tokens, _, _ = speaker(target_objects)

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


def evaluate_multimodal_intelligibility(
    speaker: Union[Speaker, SpeakerSeq],
    listener: Union[Listener, ListenerSeq],
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate mutual intelligibility in multimodal communication.

    This function measures how well agents can communicate when using both
    tokens and gestures, comparing multimodal vs unimodal performance.

    Args:
        speaker: Trained speaker model.
        listener: Trained listener model.
        n_scenes: Number of scenes to evaluate on.
        k: Number of objects per scene.
        batch_size: Batch size for evaluation.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing multimodal intelligibility metrics.
    """
    if device is None:
        device = get_device()

    speaker.eval()
    listener.eval()

    total_correct = 0
    total_scenes = 0
    multimodal_correct = 0
    unimodal_correct = 0

    with torch.no_grad():
        for i in range(0, n_scenes, batch_size):
            current_batch_size = min(batch_size, n_scenes - i)

            # Generate scenes
            scenes = []
            target_indices = []

            for j in range(current_batch_size):
                scene_objects, target_idx = sample_scene(k, seed=i + j)
                scenes.append(scene_objects)
                target_indices.append(target_idx)

            # Encode scenes
            scene_tensors = []
            for scene in scenes:
                encoded_scene = torch.stack([encode_object(obj) for obj in scene])
                scene_tensors.append(encoded_scene)

            scene_tensor = torch.stack(scene_tensors).to(device)
            target_tensor = torch.tensor(target_indices, device=device)

            # Generate messages
            if speaker.config.multimodal:
                logits, token_ids, gesture_logits, gesture_ids = speaker(
                    scene_tensor[:, 0, :]
                )
            else:
                logits, token_ids, _, _ = speaker(scene_tensor[:, 0, :])
                gesture_ids = None

            # Evaluate listener
            if listener.config.multimodal and gesture_ids is not None:
                # Multimodal evaluation
                probs = listener(token_ids, scene_tensor, gesture_ids)
                multimodal_correct += (
                    (probs.argmax(dim=-1) == target_tensor).sum().item()
                )
            else:
                # Unimodal evaluation
                probs = listener(token_ids, scene_tensor)
                unimodal_correct += (probs.argmax(dim=-1) == target_tensor).sum().item()

            total_correct += (probs.argmax(dim=-1) == target_tensor).sum().item()
            total_scenes += current_batch_size

    # Calculate metrics
    overall_accuracy = total_correct / total_scenes
    multimodal_accuracy = (
        multimodal_correct / total_scenes if speaker.config.multimodal else 0.0
    )
    unimodal_accuracy = unimodal_correct / total_scenes

    intelligibility_gain = (
        multimodal_accuracy - unimodal_accuracy if speaker.config.multimodal else 0.0
    )

    return {
        "overall_accuracy": overall_accuracy,
        "multimodal_accuracy": multimodal_accuracy,
        "unimodal_accuracy": unimodal_accuracy,
        "intelligibility_gain": intelligibility_gain,
    }


def evaluate_pragmatic_performance(
    speaker: Union[Speaker, SpeakerSeq],
    literal_listener: Union[Listener, ListenerSeq],
    pragmatic_listener: PragmaticListener,
    n_scenes: int = 1000,
    k: int = 5,
    num_distractors: int = 2,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate pragmatic listener performance on distractor scenes.

    This function compares literal vs pragmatic listener performance on
    distractor-heavy scenes where literal interpretation is ambiguous.

    Args:
        speaker: Trained speaker model.
        literal_listener: Trained literal listener model.
        pragmatic_listener: Pragmatic listener model.
        n_scenes: Number of distractor scenes to evaluate on.
        k: Number of objects per scene.
        num_distractors: Number of distractor objects.
        batch_size: Batch size for evaluation.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing pragmatic performance metrics.
    """
    if device is None:
        device = get_device()

    speaker.eval()
    literal_listener.eval()
    pragmatic_listener.eval()

    literal_correct = 0
    pragmatic_correct = 0
    total_scenes = 0

    with torch.no_grad():
        for i in range(0, n_scenes, batch_size):
            current_batch_size = min(batch_size, n_scenes - i)

            # Generate distractor scenes
            scenes = []
            target_indices = []

            for j in range(current_batch_size):
                scene_objects, target_idx = sample_distractor_scene(
                    k, num_distractors, seed=i + j
                )
                scenes.append(scene_objects)
                target_indices.append(target_idx)

            # Encode scenes
            scene_tensors = []
            for scene in scenes:
                encoded_scene = torch.stack([encode_object(obj) for obj in scene])
                scene_tensors.append(encoded_scene)

            scene_tensor = torch.stack(scene_tensors).to(device)
            target_tensor = torch.tensor(target_indices, device=device)

            # Generate messages for target objects
            target_encodings = scene_tensor[
                torch.arange(current_batch_size), target_tensor
            ]

            if speaker.config.multimodal:
                logits, token_ids, gesture_logits, gesture_ids = speaker(
                    target_encodings
                )
            else:
                logits, token_ids, _, _ = speaker(target_encodings)
                gesture_ids = None

            # Evaluate literal listener
            if literal_listener.config.multimodal and gesture_ids is not None:
                literal_probs = literal_listener(token_ids, scene_tensor, gesture_ids)
            else:
                literal_probs = literal_listener(token_ids, scene_tensor)

            literal_correct += (
                (literal_probs.argmax(dim=-1) == target_tensor).sum().item()
            )

            # Evaluate pragmatic listener
            if pragmatic_listener.config.multimodal and gesture_ids is not None:
                pragmatic_probs = pragmatic_listener(
                    token_ids, scene_tensor, gesture_ids
                )
            else:
                pragmatic_probs = pragmatic_listener(token_ids, scene_tensor)

            pragmatic_correct += (
                (pragmatic_probs.argmax(dim=-1) == target_tensor).sum().item()
            )
            total_scenes += current_batch_size

    # Calculate metrics
    literal_accuracy = literal_correct / total_scenes
    pragmatic_accuracy = pragmatic_correct / total_scenes
    pragmatic_gain = pragmatic_accuracy - literal_accuracy

    return {
        "literal_accuracy": literal_accuracy,
        "pragmatic_accuracy": pragmatic_accuracy,
        "pragmatic_gain": pragmatic_gain,
        "distractor_scenes": total_scenes,
    }


def evaluate_compositional_generalization_multimodal(
    speaker: Union[Speaker, SpeakerSeq],
    listener: Union[Listener, ListenerSeq],
    heldout_pairs: list,
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate compositional generalization in multimodal communication.

    This function tests how well multimodal agents generalize to unseen
    combinations of attributes, measuring compositional understanding.

    Args:
        speaker: Trained speaker model.
        listener: Trained listener model.
        heldout_pairs: List of held-out attribute pairs.
        n_scenes: Number of scenes to evaluate on.
        k: Number of objects per scene.
        batch_size: Batch size for evaluation.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing compositional generalization metrics.
    """
    if device is None:
        device = get_device()

    speaker.eval()
    listener.eval()

    total_correct = 0
    total_scenes = 0
    compositional_correct = 0

    with torch.no_grad():
        for i in range(0, n_scenes, batch_size):
            current_batch_size = min(batch_size, n_scenes - i)

            # Generate compositional scenes
            dataset = CompositionalDataset(scenes=[], targets=[])

            dataloader = DataLoader(
                dataset, batch_size=current_batch_size, shuffle=False
            )

            for batch_scenes, batch_targets, batch_candidates in dataloader:
                batch_scenes = batch_scenes.to(device)
                batch_targets = batch_targets.to(device)
                batch_candidates = batch_candidates.to(device)

                # Generate messages
                if speaker.config.multimodal:
                    logits, token_ids, gesture_logits, gesture_ids = speaker(
                        batch_scenes
                    )
                else:
                    logits, token_ids, _, _ = speaker(batch_scenes)
                    gesture_ids = None

                # Evaluate listener
                if listener.config.multimodal and gesture_ids is not None:
                    probs = listener(token_ids, batch_candidates, gesture_ids)
                else:
                    probs = listener(token_ids, batch_candidates)

                predictions = probs.argmax(dim=-1)
                correct = (predictions == batch_targets).sum().item()

                total_correct += correct
                total_scenes += current_batch_size

                # Check compositional accuracy
                compositional_correct += correct

    # Calculate metrics
    overall_accuracy = total_correct / total_scenes
    compositional_accuracy = compositional_correct / total_scenes

    return {
        "overall_accuracy": overall_accuracy,
        "compositional_accuracy": compositional_accuracy,
        "generalization_gap": overall_accuracy - compositional_accuracy,
    }
