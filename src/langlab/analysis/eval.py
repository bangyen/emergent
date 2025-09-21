"""Evaluation module for emergent language experiments.

This module provides evaluation functionality for testing compositional
generalization in referential games, enabling systematic assessment of
agent performance across different data splits.
"""

import json
import os
from typing import Dict, Optional, Union, List
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy import stats

from ..core.agents import Speaker, Listener, SpeakerSeq, ListenerSeq, PragmaticListener
from ..core.contrastive_agents import ContrastiveSpeaker, ContrastiveListener
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

    # Create agents based on checkpoint type
    speaker: Union[Speaker, SpeakerSeq, ContrastiveSpeaker]
    listener: Union[Listener, ListenerSeq, ContrastiveListener]

    # Check if checkpoint was saved with contrastive agents by looking at state dict keys
    speaker_state_dict = checkpoint["speaker_state_dict"]
    has_contrastive_keys = any(
        "contrastive_head" in key or "message_generator" in key
        for key in speaker_state_dict.keys()
    )

    # Prioritize config flags over key detection
    if (
        hasattr(config, "use_contrastive")
        and config.use_contrastive
        and has_contrastive_keys
    ):
        speaker = ContrastiveSpeaker(config).to(device)
        listener = ContrastiveListener(config).to(device)
    elif hasattr(config, "use_sequence_models") and config.use_sequence_models:
        speaker = SpeakerSeq(config).to(device)
        listener = ListenerSeq(config).to(device)
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
        # Use regular dataset for training evaluation - use same seed as training for consistency
        dataset: Union[ReferentialGameDataset, CompositionalDataset] = (
            ReferentialGameDataset(n_scenes, k, seed=7)  # Match training seed
        )
    elif split in ["iid", "compo"]:
        # Use compositional splits
        if heldout_pairs is None:
            raise ValueError("heldout_pairs must be provided for compositional splits")

        from ..data.data import make_compositional_splits

        splits = make_compositional_splits(
            n_scenes, k, heldout_pairs, seed=7
        )  # Match training seed
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
            speaker_output = speaker(target_objects)
            if len(speaker_output) == 4:
                _, message_tokens, _, _ = speaker_output
            else:
                _, message_tokens = speaker_output

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


def evaluate_with_confidence_intervals(
    model_path: str,
    split: str = "train",
    heldout_pairs: Optional[list] = None,
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
    n_runs: int = 10,
    confidence_level: float = 0.95,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate model performance with confidence intervals across multiple runs.

    This function runs evaluation multiple times with different random seeds to
    provide robust accuracy estimates with confidence intervals, addressing
    the deterministic evaluation limitation.

    Args:
        model_path: Path to the saved model checkpoint.
        split: Data split to evaluate on ("train", "iid", or "compo").
        heldout_pairs: List of held-out attribute pairs for compositional splits.
        n_scenes: Number of scenes to generate for evaluation.
        k: Number of objects per scene.
        batch_size: Batch size for evaluation.
        n_runs: Number of evaluation runs for confidence intervals.
        confidence_level: Confidence level for intervals (e.g., 0.95 for 95%).
        device: Device to run evaluation on.

    Returns:
        Dictionary containing accuracy metrics with confidence intervals.
    """
    if device is None:
        device = get_device()

    # Run evaluation multiple times with different seeds
    accuracies = []
    for run in range(n_runs):
        # Use different seed for each run to get variance
        eval_seed = 7 + run * 1000  # Offset from training seed

        # Temporarily modify the evaluate function to accept seed
        accuracy = _evaluate_with_seed(
            model_path=model_path,
            split=split,
            heldout_pairs=heldout_pairs,
            n_scenes=n_scenes,
            k=k,
            batch_size=batch_size,
            seed=eval_seed,
            device=device,
        )
        accuracies.append(accuracy)

    # Calculate statistics
    accuracies = np.array(accuracies)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies, ddof=1)  # Sample standard deviation

    # Calculate confidence interval
    n = len(accuracies)
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin_of_error = t_critical * (std_accuracy / np.sqrt(n))

    ci_lower = mean_accuracy - margin_of_error
    ci_upper = mean_accuracy + margin_of_error

    logger.info(
        f"Multi-run evaluation on {split} split: "
        f"mean={mean_accuracy:.4f}±{std_accuracy:.4f}, "
        f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
    )

    return {
        "acc_mean": mean_accuracy,
        "acc_std": std_accuracy,
        "acc_ci_lower": ci_lower,
        "acc_ci_upper": ci_upper,
        "n_runs": n_runs,
        "confidence_level": confidence_level,
    }


def _evaluate_with_seed(
    model_path: str,
    split: str = "train",
    heldout_pairs: Optional[list] = None,
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
    seed: int = 7,
    device: Optional[torch.device] = None,
) -> float:
    """Internal function to evaluate with a specific seed."""
    if device is None:
        device = get_device()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Create agents based on checkpoint type
    speaker: Union[Speaker, SpeakerSeq, ContrastiveSpeaker]
    listener: Union[Listener, ListenerSeq, ContrastiveListener]

    # Check if checkpoint was saved with contrastive agents by looking at state dict keys
    speaker_state_dict = checkpoint["speaker_state_dict"]
    has_contrastive_keys = any(
        "contrastive_head" in key or "message_generator" in key
        for key in speaker_state_dict.keys()
    )

    # Prioritize config flags over key detection
    if (
        hasattr(config, "use_contrastive")
        and config.use_contrastive
        and has_contrastive_keys
    ):
        speaker = ContrastiveSpeaker(config).to(device)
        listener = ContrastiveListener(config).to(device)
    elif hasattr(config, "use_sequence_models") and config.use_sequence_models:
        speaker = SpeakerSeq(config).to(device)
        listener = ListenerSeq(config).to(device)
    else:
        speaker = Speaker(config).to(device)
        listener = Listener(config).to(device)

    # Load model states
    speaker.load_state_dict(checkpoint["speaker_state_dict"])
    listener.load_state_dict(checkpoint["listener_state_dict"])

    # Set to evaluation mode
    speaker.eval()
    listener.eval()

    # Create dataset based on split with specific seed
    if split == "train":
        dataset: Union[ReferentialGameDataset, CompositionalDataset] = (
            ReferentialGameDataset(n_scenes, k, seed=seed)
        )
    elif split in ["iid", "compo"]:
        if heldout_pairs is None:
            raise ValueError("heldout_pairs must be provided for compositional splits")

        from ..data.data import make_compositional_splits

        splits = make_compositional_splits(n_scenes, k, heldout_pairs, seed=seed)
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
            speaker_output = speaker(target_objects)
            if len(speaker_output) == 4:
                _, message_tokens, _, _ = speaker_output
            else:
                _, message_tokens = speaker_output

            # Listener makes predictions
            listener_probs = listener(message_tokens, candidate_objects)
            listener_predictions = torch.argmax(listener_probs, dim=1)

            # Count correct predictions
            correct = (listener_predictions == target_indices).sum().item()
            total_correct += correct
            total_samples += batch_size

    # Compute accuracy
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return accuracy


def evaluate_with_temperature_scaling(
    model_path: str,
    split: str = "train",
    heldout_pairs: Optional[list] = None,
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate model performance with temperature scaling for better calibration.

    Temperature scaling can improve model calibration by scaling the logits
    before applying softmax, potentially improving accuracy on difficult cases.

    Args:
        model_path: Path to the saved model checkpoint.
        split: Data split to evaluate on ("train", "iid", or "compo").
        heldout_pairs: List of held-out attribute pairs for compositional splits.
        n_scenes: Number of scenes to generate for evaluation.
        k: Number of objects per scene.
        batch_size: Batch size for evaluation.
        temperature: Temperature scaling factor (1.0 = no scaling).
        device: Device to run evaluation on.

    Returns:
        Dictionary containing accuracy metrics with temperature scaling.
    """
    if device is None:
        device = get_device()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Create agents based on checkpoint type
    speaker: Union[Speaker, SpeakerSeq, ContrastiveSpeaker]
    listener: Union[Listener, ListenerSeq, ContrastiveListener]

    # Check if checkpoint was saved with contrastive agents by looking at state dict keys
    speaker_state_dict = checkpoint["speaker_state_dict"]
    has_contrastive_keys = any(
        "contrastive_head" in key or "message_generator" in key
        for key in speaker_state_dict.keys()
    )

    # Prioritize config flags over key detection
    if (
        hasattr(config, "use_contrastive")
        and config.use_contrastive
        and has_contrastive_keys
    ):
        speaker = ContrastiveSpeaker(config).to(device)
        listener = ContrastiveListener(config).to(device)
    elif hasattr(config, "use_sequence_models") and config.use_sequence_models:
        speaker = SpeakerSeq(config).to(device)
        listener = ListenerSeq(config).to(device)
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
        dataset: Union[ReferentialGameDataset, CompositionalDataset] = (
            ReferentialGameDataset(n_scenes, k, seed=7)
        )
    elif split in ["iid", "compo"]:
        if heldout_pairs is None:
            raise ValueError("heldout_pairs must be provided for compositional splits")

        from ..data.data import make_compositional_splits

        splits = make_compositional_splits(n_scenes, k, heldout_pairs, seed=7)
        dataset = splits[split]
    else:
        raise ValueError(f"Unsupported split: {split}")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Evaluate with temperature scaling
    total_correct = 0
    total_samples = 0
    confidence_scores: List[float] = []

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

            # Speaker generates messages with temperature
            speaker_output = speaker(target_objects, temperature=temperature)
            if len(speaker_output) == 4:
                _, message_tokens, _, _ = speaker_output
            else:
                _, message_tokens = speaker_output

            # Listener makes predictions
            listener_probs = listener(message_tokens, candidate_objects)

            # Apply temperature scaling to listener probabilities
            if temperature != 1.0:
                # Convert probabilities back to logits, scale, then back to probabilities
                logits = torch.log(
                    listener_probs + 1e-8
                )  # Add small epsilon for numerical stability
                scaled_logits = logits / temperature
                listener_probs = torch.softmax(scaled_logits, dim=1)

            listener_predictions = torch.argmax(listener_probs, dim=1)

            # Store confidence scores (max probability)
            max_probs = torch.max(listener_probs, dim=1)[0]
            confidence_scores.extend(max_probs.cpu().numpy())

            # Count correct predictions
            correct = (listener_predictions == target_indices).sum().item()
            total_correct += correct
            total_samples += batch_size

    # Compute accuracy
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_confidence = np.mean(confidence_scores)

    logger.info(
        f"Temperature-scaled evaluation on {split} split: "
        f"acc={accuracy:.4f}, avg_confidence={avg_confidence:.4f}, temp={temperature}"
    )

    return {
        "acc": float(accuracy),
        "avg_confidence": float(avg_confidence),
        "temperature": float(temperature),
        "confidence_std": float(np.std(confidence_scores)),
    }


def evaluate_with_uncertainty_quantification(
    model_path: str,
    split: str = "train",
    heldout_pairs: Optional[list] = None,
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
    n_samples: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate model performance with uncertainty quantification.

    This function uses Monte Carlo sampling to estimate model uncertainty
    by running multiple forward passes with dropout enabled.

    Args:
        model_path: Path to the saved model checkpoint.
        split: Data split to evaluate on ("train", "iid", or "compo").
        heldout_pairs: List of held-out attribute pairs for compositional splits.
        n_scenes: Number of scenes to generate for evaluation.
        k: Number of objects per scene.
        batch_size: Batch size for evaluation.
        n_samples: Number of Monte Carlo samples for uncertainty estimation.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing accuracy metrics with uncertainty quantification.
    """
    if device is None:
        device = get_device()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Create agents based on checkpoint type
    speaker: Union[Speaker, SpeakerSeq, ContrastiveSpeaker]
    listener: Union[Listener, ListenerSeq, ContrastiveListener]

    # Check if checkpoint was saved with contrastive agents by looking at state dict keys
    speaker_state_dict = checkpoint["speaker_state_dict"]
    has_contrastive_keys = any(
        "contrastive_head" in key or "message_generator" in key
        for key in speaker_state_dict.keys()
    )

    # Prioritize config flags over key detection
    if (
        hasattr(config, "use_contrastive")
        and config.use_contrastive
        and has_contrastive_keys
    ):
        speaker = ContrastiveSpeaker(config).to(device)
        listener = ContrastiveListener(config).to(device)
    elif hasattr(config, "use_sequence_models") and config.use_sequence_models:
        speaker = SpeakerSeq(config).to(device)
        listener = ListenerSeq(config).to(device)
    else:
        speaker = Speaker(config).to(device)
        listener = Listener(config).to(device)

    # Load model states
    speaker.load_state_dict(checkpoint["speaker_state_dict"])
    listener.load_state_dict(checkpoint["listener_state_dict"])

    # Set to evaluation mode but enable dropout for uncertainty
    speaker.eval()
    listener.eval()

    # Enable dropout for uncertainty estimation
    def enable_dropout(model: torch.nn.Module) -> None:
        for module in model.modules():
            if module.__class__.__name__.startswith("Dropout"):
                module.train()

    enable_dropout(speaker)
    enable_dropout(listener)

    # Create dataset based on split
    if split == "train":
        dataset: Union[ReferentialGameDataset, CompositionalDataset] = (
            ReferentialGameDataset(n_scenes, k, seed=7)
        )
    elif split in ["iid", "compo"]:
        if heldout_pairs is None:
            raise ValueError("heldout_pairs must be provided for compositional splits")

        from ..data.data import make_compositional_splits

        splits = make_compositional_splits(n_scenes, k, heldout_pairs, seed=7)
        dataset = splits[split]
    else:
        raise ValueError(f"Unsupported split: {split}")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Collect predictions and uncertainties
    all_predictions = []
    all_uncertainties = []
    all_targets = []

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

            # Monte Carlo sampling for uncertainty
            batch_predictions = []
            batch_probs = []

            for _ in range(n_samples):
                # Speaker generates messages
                speaker_output = speaker(target_objects)
                if len(speaker_output) == 4:
                    _, message_tokens, _, _ = speaker_output
                else:
                    _, message_tokens = speaker_output

                # Listener makes predictions
                listener_probs = listener(message_tokens, candidate_objects)
                predictions = torch.argmax(listener_probs, dim=1)

                batch_predictions.append(predictions.cpu().numpy())
                batch_probs.append(listener_probs.cpu().numpy())

            # Calculate uncertainty (variance across samples)
            batch_predictions = np.array(batch_predictions)  # [n_samples, batch_size]
            batch_probs = np.array(batch_probs)  # [n_samples, batch_size, k]

            # Prediction uncertainty (variance in predictions)
            pred_uncertainty = np.var(batch_predictions, axis=0)

            # Probability uncertainty (variance in max probabilities)
            # max_probs = np.max(batch_probs, axis=2)  # [n_samples, batch_size]
            # prob_uncertainty = np.var(max_probs, axis=0)  # Unused for now

            # Use majority vote for final predictions
            final_predictions = stats.mode(batch_predictions, axis=0)[0].flatten()

            all_predictions.extend(final_predictions)
            all_uncertainties.extend(pred_uncertainty)
            all_targets.extend(target_indices.cpu().numpy())

    # Calculate accuracy
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_uncertainties = np.array(all_uncertainties)

    accuracy = np.mean(all_predictions == all_targets)
    avg_uncertainty = np.mean(all_uncertainties)

    # Calculate accuracy vs uncertainty correlation
    correct_mask = all_predictions == all_targets
    uncertainty_correlation = np.corrcoef(
        correct_mask.astype(float), all_uncertainties
    )[0, 1]

    logger.info(
        f"Uncertainty-quantified evaluation on {split} split: "
        f"acc={accuracy:.4f}, avg_uncertainty={avg_uncertainty:.4f}, "
        f"uncertainty_correlation={uncertainty_correlation:.4f}"
    )

    return {
        "acc": accuracy,
        "avg_uncertainty": avg_uncertainty,
        "uncertainty_correlation": uncertainty_correlation,
        "n_samples": n_samples,
    }


def evaluate_with_confidence_metrics(
    model_path: str,
    split: str = "train",
    heldout_pairs: Optional[list] = None,
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate model performance with confidence-based metrics.

    This function provides additional metrics beyond simple accuracy, including
    confidence calibration, top-k accuracy, and confidence-weighted accuracy.

    Args:
        model_path: Path to the saved model checkpoint.
        split: Data split to evaluate on ("train", "iid", or "compo").
        heldout_pairs: List of held-out attribute pairs for compositional splits.
        n_scenes: Number of scenes to generate for evaluation.
        k: Number of objects per scene.
        batch_size: Batch size for evaluation.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing confidence-based evaluation metrics.
    """
    if device is None:
        device = get_device()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Create agents based on checkpoint type
    speaker: Union[Speaker, SpeakerSeq, ContrastiveSpeaker]
    listener: Union[Listener, ListenerSeq, ContrastiveListener]

    # Check if checkpoint was saved with contrastive agents by looking at state dict keys
    speaker_state_dict = checkpoint["speaker_state_dict"]
    has_contrastive_keys = any(
        "contrastive_head" in key or "message_generator" in key
        for key in speaker_state_dict.keys()
    )

    # Prioritize config flags over key detection
    if (
        hasattr(config, "use_contrastive")
        and config.use_contrastive
        and has_contrastive_keys
    ):
        speaker = ContrastiveSpeaker(config).to(device)
        listener = ContrastiveListener(config).to(device)
    elif hasattr(config, "use_sequence_models") and config.use_sequence_models:
        speaker = SpeakerSeq(config).to(device)
        listener = ListenerSeq(config).to(device)
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
        dataset: Union[ReferentialGameDataset, CompositionalDataset] = (
            ReferentialGameDataset(n_scenes, k, seed=7)
        )
    elif split in ["iid", "compo"]:
        if heldout_pairs is None:
            raise ValueError("heldout_pairs must be provided for compositional splits")

        from ..data.data import make_compositional_splits

        splits = make_compositional_splits(n_scenes, k, heldout_pairs, seed=7)
        dataset = splits[split]
    else:
        raise ValueError(f"Unsupported split: {split}")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Collect predictions and confidences
    all_predictions: List[int] = []
    all_confidences: List[float] = []
    all_targets = []
    all_probs = []

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
            speaker_output = speaker(target_objects)
            if len(speaker_output) == 4:
                _, message_tokens, _, _ = speaker_output
            else:
                _, message_tokens = speaker_output

            # Listener makes predictions
            listener_probs = listener(message_tokens, candidate_objects)
            listener_predictions = torch.argmax(listener_probs, dim=1)

            # Get confidence scores (max probability)
            max_probs = torch.max(listener_probs, dim=1)[0]

            all_predictions.extend(listener_predictions.cpu().numpy())
            all_confidences.extend(max_probs.cpu().numpy())
            all_targets.extend(target_indices.cpu().numpy())
            all_probs.extend(listener_probs.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_confidences = np.array(all_confidences)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate basic accuracy
    accuracy = np.mean(all_predictions == all_targets)

    # Calculate top-k accuracy (k=2, 3)
    top2_correct = 0
    top3_correct = 0

    for i, target in enumerate(all_targets):
        # Get top-k predictions
        top2_preds = np.argsort(all_probs[i])[-2:]
        top3_preds = np.argsort(all_probs[i])[-3:]

        if target in top2_preds:
            top2_correct += 1
        if target in top3_preds:
            top3_correct += 1

    top2_accuracy = top2_correct / len(all_targets)
    top3_accuracy = top3_correct / len(all_targets)

    # Calculate confidence-weighted accuracy
    correct_mask = all_predictions == all_targets
    confidence_weighted_accuracy: float = np.sum(
        correct_mask * all_confidences
    ) / np.sum(all_confidences)

    # Calculate confidence calibration (ECE - Expected Calibration Error)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (all_confidences > bin_lower) & (all_confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = correct_mask[in_bin].mean()
            avg_confidence_in_bin = all_confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    # Calculate confidence distribution statistics
    confidence_mean = np.mean(all_confidences)
    confidence_std = np.std(all_confidences)
    confidence_entropy: float = float(
        -np.sum(np.array(all_confidences) * np.log(np.array(all_confidences) + 1e-8))
    )

    logger.info(
        f"Confidence-based evaluation on {split} split: "
        f"acc={accuracy:.4f}, top2_acc={top2_accuracy:.4f}, top3_acc={top3_accuracy:.4f}, "
        f"conf_weighted_acc={confidence_weighted_accuracy:.4f}, ece={ece:.4f}"
    )

    return {
        "acc": float(accuracy),
        "top2_acc": float(top2_accuracy),
        "top3_acc": float(top3_accuracy),
        "confidence_weighted_acc": float(confidence_weighted_accuracy),
        "ece": float(ece),
        "confidence_mean": float(confidence_mean),
        "confidence_std": float(confidence_std),
        "confidence_entropy": float(confidence_entropy),
    }


def evaluate_ensemble_robustness(
    model_paths: List[str],
    split: str = "train",
    heldout_pairs: Optional[list] = None,
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
    ensemble_method: str = "voting",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate ensemble of models for improved robustness and accuracy.

    This function loads multiple models and combines their predictions using
    voting or averaging to improve evaluation accuracy and robustness.

    Args:
        model_paths: List of paths to saved model checkpoints.
        split: Data split to evaluate on ("train", "iid", or "compo").
        heldout_pairs: List of held-out attribute pairs for compositional splits.
        n_scenes: Number of scenes to generate for evaluation.
        k: Number of objects per scene.
        batch_size: Batch size for evaluation.
        ensemble_method: Method for combining predictions ("voting" or "averaging").
        device: Device to run evaluation on.

    Returns:
        Dictionary containing ensemble evaluation metrics.
    """
    if device is None:
        device = get_device()

    if not model_paths:
        raise ValueError("At least one model path must be provided")

    # Load all models
    models = []
    for model_path in model_paths:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint["config"]

        # Create agents
        speaker: Union[Speaker, SpeakerSeq]
        listener: Union[Listener, ListenerSeq]

        if hasattr(config, "use_sequence_models") and config.use_sequence_models:
            speaker = SpeakerSeq(config).to(device)
            listener = ListenerSeq(config).to(device)
        else:
            speaker = Speaker(config).to(device)
            listener = Listener(config).to(device)

        # Load model states
        speaker.load_state_dict(checkpoint["speaker_state_dict"])
        listener.load_state_dict(checkpoint["listener_state_dict"])

        # Set to evaluation mode
        speaker.eval()
        listener.eval()

        models.append((speaker, listener, config))

    # Create dataset based on split
    if split == "train":
        dataset: Union[ReferentialGameDataset, CompositionalDataset] = (
            ReferentialGameDataset(n_scenes, k, seed=7)
        )
    elif split in ["iid", "compo"]:
        if heldout_pairs is None:
            raise ValueError("heldout_pairs must be provided for compositional splits")

        from ..data.data import make_compositional_splits

        splits = make_compositional_splits(n_scenes, k, heldout_pairs, seed=7)
        dataset = splits[split]
    else:
        raise ValueError(f"Unsupported split: {split}")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Evaluate ensemble
    total_correct = 0
    total_samples = 0
    individual_accuracies = []

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

            # Collect predictions from all models
            all_predictions = []
            all_probabilities = []

            for speaker, listener, config in models:
                # Speaker generates messages
                speaker_output = speaker(target_objects)
                if len(speaker_output) == 4:
                    _, message_tokens, _, _ = speaker_output
                else:
                    _, message_tokens = speaker_output

                # Listener makes predictions
                listener_probs = listener(message_tokens, candidate_objects)
                predictions = torch.argmax(listener_probs, dim=1)

                all_predictions.append(predictions.cpu().numpy())
                all_probabilities.append(listener_probs.cpu().numpy())

            # Combine predictions
            all_predictions = np.array(all_predictions)  # [n_models, batch_size]
            all_probabilities = np.array(all_probabilities)  # [n_models, batch_size, k]

            if ensemble_method == "voting":
                # Majority voting
                ensemble_predictions = stats.mode(all_predictions, axis=0)[0].flatten()
            elif ensemble_method == "averaging":
                # Average probabilities then take argmax
                avg_probs = np.mean(all_probabilities, axis=0)
                ensemble_predictions = np.argmax(avg_probs, axis=1)
            else:
                raise ValueError(f"Unknown ensemble method: {ensemble_method}")

            # Count correct predictions
            correct: int = np.sum(ensemble_predictions == target_indices.cpu().numpy())
            total_correct += correct
            total_samples += batch_size

            # Calculate individual model accuracies for this batch
            for i, preds in enumerate(all_predictions):
                individual_correct: int = np.sum(preds == target_indices.cpu().numpy())
                individual_accuracies.append(individual_correct / batch_size)

    # Calculate ensemble accuracy
    ensemble_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # Calculate individual model accuracies
    individual_accuracy = np.mean(individual_accuracies)
    individual_std = np.std(individual_accuracies)

    # Calculate ensemble improvement
    ensemble_improvement = ensemble_accuracy - individual_accuracy

    logger.info(
        f"Ensemble evaluation on {split} split: "
        f"ensemble_acc={ensemble_accuracy:.4f}, individual_acc={individual_accuracy:.4f}±{individual_std:.4f}, "
        f"improvement={ensemble_improvement:.4f}, method={ensemble_method}"
    )

    return {
        "ensemble_acc": float(ensemble_accuracy),
        "individual_acc_mean": float(individual_accuracy),
        "individual_acc_std": float(individual_std),
        "ensemble_improvement": float(ensemble_improvement),
        "n_models": len(model_paths),
        "ensemble_method": str(ensemble_method),  # type: ignore[dict-item]
    }


def evaluate_with_bootstrap_confidence_intervals(
    model_path: str,
    split: str = "train",
    heldout_pairs: Optional[list] = None,
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate model performance with bootstrap confidence intervals.

    This function uses bootstrap resampling to estimate confidence intervals
    for accuracy metrics, providing robust uncertainty estimates.

    Args:
        model_path: Path to the saved model checkpoint.
        split: Data split to evaluate on ("train", "iid", or "compo").
        heldout_pairs: List of held-out attribute pairs for compositional splits.
        n_scenes: Number of scenes to generate for evaluation.
        k: Number of objects per scene.
        batch_size: Batch size for evaluation.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level for intervals (e.g., 0.95 for 95%).
        device: Device to run evaluation on.

    Returns:
        Dictionary containing accuracy metrics with bootstrap confidence intervals.
    """
    if device is None:
        device = get_device()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Create agents based on checkpoint type
    speaker: Union[Speaker, SpeakerSeq, ContrastiveSpeaker]
    listener: Union[Listener, ListenerSeq, ContrastiveListener]

    # Check if checkpoint was saved with contrastive agents by looking at state dict keys
    speaker_state_dict = checkpoint["speaker_state_dict"]
    has_contrastive_keys = any(
        "contrastive_head" in key or "message_generator" in key
        for key in speaker_state_dict.keys()
    )

    # Prioritize config flags over key detection
    if (
        hasattr(config, "use_contrastive")
        and config.use_contrastive
        and has_contrastive_keys
    ):
        speaker = ContrastiveSpeaker(config).to(device)
        listener = ContrastiveListener(config).to(device)
    elif hasattr(config, "use_sequence_models") and config.use_sequence_models:
        speaker = SpeakerSeq(config).to(device)
        listener = ListenerSeq(config).to(device)
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
        dataset: Union[ReferentialGameDataset, CompositionalDataset] = (
            ReferentialGameDataset(n_scenes, k, seed=7)
        )
    elif split in ["iid", "compo"]:
        if heldout_pairs is None:
            raise ValueError("heldout_pairs must be provided for compositional splits")

        from ..data.data import make_compositional_splits

        splits = make_compositional_splits(n_scenes, k, heldout_pairs, seed=7)
        dataset = splits[split]
    else:
        raise ValueError(f"Unsupported split: {split}")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Collect all predictions and targets for bootstrap
    all_predictions: List[int] = []
    all_targets = []

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
            speaker_output = speaker(target_objects)
            if len(speaker_output) == 4:
                _, message_tokens, _, _ = speaker_output
            else:
                _, message_tokens = speaker_output

            # Listener makes predictions
            listener_probs = listener(message_tokens, candidate_objects)
            listener_predictions = torch.argmax(listener_probs, dim=1)

            all_predictions.extend(listener_predictions.cpu().numpy())
            all_targets.extend(target_indices.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate original accuracy
    original_accuracy = np.mean(all_predictions == all_targets)

    # Bootstrap sampling
    n_samples = len(all_predictions)
    bootstrap_accuracies = []

    np.random.seed(42)  # For reproducibility
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_predictions = all_predictions[bootstrap_indices]
        bootstrap_targets = all_targets[bootstrap_indices]

        # Calculate accuracy for this bootstrap sample
        bootstrap_accuracy = np.mean(bootstrap_predictions == bootstrap_targets)
        bootstrap_accuracies.append(bootstrap_accuracy)

    # Calculate bootstrap statistics
    bootstrap_accuracies = np.array(bootstrap_accuracies)
    bootstrap_mean = np.mean(bootstrap_accuracies)
    bootstrap_std = np.std(bootstrap_accuracies)

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_accuracies, lower_percentile)
    ci_upper = np.percentile(bootstrap_accuracies, upper_percentile)

    logger.info(
        f"Bootstrap evaluation on {split} split: "
        f"original_acc={original_accuracy:.4f}, bootstrap_mean={bootstrap_mean:.4f}±{bootstrap_std:.4f}, "
        f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
    )

    return {
        "acc_original": float(original_accuracy),
        "acc_bootstrap_mean": float(bootstrap_mean),
        "acc_bootstrap_std": float(bootstrap_std),
        "acc_ci_lower": float(ci_lower),
        "acc_ci_upper": float(ci_upper),
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
    }


def comprehensive_evaluation(
    model_path: str,
    split: str = "train",
    heldout_pairs: Optional[list] = None,
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """Run comprehensive evaluation with all improvement methods.

    This function runs all evaluation improvements and compares their results
    to identify which methods provide significant accuracy improvements.

    Args:
        model_path: Path to the saved model checkpoint.
        split: Data split to evaluate on ("train", "iid", or "compo").
        heldout_pairs: List of held-out attribute pairs for compositional splits.
        n_scenes: Number of scenes to generate for evaluation.
        k: Number of objects per scene.
        batch_size: Batch size for evaluation.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing results from all evaluation methods.
    """
    logger.info(f"Running comprehensive evaluation on {split} split...")

    results = {}

    # 1. Baseline evaluation
    logger.info("Running baseline evaluation...")
    results["baseline"] = evaluate(
        model_path=model_path,
        split=split,
        heldout_pairs=heldout_pairs,
        n_scenes=n_scenes,
        k=k,
        batch_size=batch_size,
        device=device,
    )

    # 2. Multi-run evaluation with confidence intervals
    logger.info("Running multi-run evaluation with confidence intervals...")
    results["multi_run"] = evaluate_with_confidence_intervals(
        model_path=model_path,
        split=split,
        heldout_pairs=heldout_pairs,
        n_scenes=n_scenes,
        k=k,
        batch_size=batch_size,
        n_runs=10,
        device=device,
    )

    # 3. Temperature scaling evaluation
    logger.info("Running temperature scaling evaluation...")
    temperatures = [0.5, 0.8, 1.0, 1.2, 1.5]
    temp_results = {}
    for temp in temperatures:
        temp_results[f"temp_{temp}"] = evaluate_with_temperature_scaling(
            model_path=model_path,
            split=split,
            heldout_pairs=heldout_pairs,
            n_scenes=n_scenes,
            k=k,
            batch_size=batch_size,
            temperature=temp,
            device=device,
        )
    results["temperature_scaling"] = temp_results  # type: ignore[assignment]

    # 4. Confidence-based metrics
    logger.info("Running confidence-based evaluation...")
    results["confidence_metrics"] = evaluate_with_confidence_metrics(
        model_path=model_path,
        split=split,
        heldout_pairs=heldout_pairs,
        n_scenes=n_scenes,
        k=k,
        batch_size=batch_size,
        device=device,
    )

    # 5. Uncertainty quantification
    logger.info("Running uncertainty quantification...")
    results["uncertainty"] = evaluate_with_uncertainty_quantification(
        model_path=model_path,
        split=split,
        heldout_pairs=heldout_pairs,
        n_scenes=n_scenes,
        k=k,
        batch_size=batch_size,
        n_samples=5,
        device=device,
    )

    # 6. Bootstrap confidence intervals
    logger.info("Running bootstrap confidence intervals...")
    results["bootstrap"] = evaluate_with_bootstrap_confidence_intervals(
        model_path=model_path,
        split=split,
        heldout_pairs=heldout_pairs,
        n_scenes=n_scenes,
        k=k,
        batch_size=batch_size,
        n_bootstrap=1000,
        device=device,
    )

    # 7. Ensemble evaluation (if multiple models available)
    ensemble_models = [
        "outputs/checkpoints/ensemble_model_0.pt",
        "outputs/checkpoints/ensemble_model_1.pt",
        "outputs/checkpoints/ensemble_model_2.pt",
    ]

    available_models = [path for path in ensemble_models if os.path.exists(path)]
    if len(available_models) > 1:
        logger.info("Running ensemble evaluation...")
        results["ensemble"] = evaluate_ensemble_robustness(
            model_paths=available_models,
            split=split,
            heldout_pairs=heldout_pairs,
            n_scenes=n_scenes,
            k=k,
            batch_size=batch_size,
            ensemble_method="voting",
            device=device,
        )
    else:
        logger.info("Skipping ensemble evaluation - insufficient models available")
        results["ensemble"] = {"error": "Insufficient ensemble models available"}  # type: ignore[dict-item]

    # Summary analysis
    logger.info("Analyzing results...")
    baseline_acc = results["baseline"]["acc"]

    # Find best temperature
    best_temp_acc = baseline_acc
    best_temp = 1.0
    for temp_key, temp_result in results["temperature_scaling"].items():
        if temp_result["acc"] > best_temp_acc:  # type: ignore[index]
            best_temp_acc = temp_result["acc"]  # type: ignore[index]
            best_temp = temp_result["temperature"]  # type: ignore[index]

    # Calculate improvements
    improvements = {
        "multi_run_improvement": results["multi_run"]["acc_mean"] - baseline_acc,
        "best_temp_improvement": best_temp_acc - baseline_acc,
        "confidence_weighted_improvement": results["confidence_metrics"][
            "confidence_weighted_acc"
        ]
        - baseline_acc,
        "uncertainty_improvement": results["uncertainty"]["acc"] - baseline_acc,
        "bootstrap_improvement": results["bootstrap"]["acc_bootstrap_mean"]
        - baseline_acc,
    }

    if "ensemble" in results and "error" not in results["ensemble"]:
        improvements["ensemble_improvement"] = results["ensemble"][
            "ensemble_improvement"
        ]

    results["summary"] = {
        "baseline_accuracy": baseline_acc,
        "best_temperature": best_temp,
        "improvements": improvements,  # type: ignore[dict-item]
    }

    logger.info(
        f"Comprehensive evaluation completed. Baseline accuracy: {baseline_acc:.4f}"
    )
    for method, improvement in improvements.items():
        logger.info(f"{method}: {improvement:+.4f}")

    return results
