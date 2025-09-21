"""Ablation study module for systematic parameter exploration.

This module provides functionality for running ablation studies across different
parameter configurations to understand the impact of vocabulary size, channel noise,
and length cost on emergent language performance.
"""

import itertools
import json
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

from ..training.train import train
from ..analysis.eval import evaluate_all_splits
from ..utils.utils import get_logger, set_seed

logger = get_logger(__name__)


def generate_parameter_grid(
    vocab_sizes: List[int] = [6, 12, 24],
    channel_noise_levels: List[float] = [0.0, 0.05, 0.1],
    length_costs: List[float] = [0.0, 0.01, 0.05],
) -> List[Dict[str, Any]]:
    """Generate parameter grid for ablation studies.

    This function creates all combinations of the specified parameter values
    to enable systematic exploration of their effects on emergent language.

    Args:
        vocab_sizes: List of vocabulary sizes to test.
        channel_noise_levels: List of channel noise levels to test.
        length_costs: List of length cost weights to test.

    Returns:
        List of parameter dictionaries, each containing one combination.
    """
    param_grid = []

    for v, noise, length_cost in itertools.product(
        vocab_sizes, channel_noise_levels, length_costs
    ):
        param_grid.append(
            {
                "V": v,
                "channel_noise": noise,
                "length_cost": length_cost,
            }
        )

    logger.info(f"Generated parameter grid with {len(param_grid)} combinations")
    return param_grid


def add_channel_noise(
    message_tokens: torch.Tensor, noise_level: float, vocab_size: int
) -> torch.Tensor:
    """Add channel noise to message tokens.

    This function simulates noisy communication by randomly replacing tokens
    with other tokens from the vocabulary according to the noise level.

    Args:
        message_tokens: Tensor of shape (batch_size, message_length) with token indices.
        noise_level: Probability of replacing each token with a random one.
        vocab_size: Size of vocabulary for random token generation.

    Returns:
        Tensor with same shape as input, potentially with noisy tokens.
    """
    if noise_level == 0.0:
        return message_tokens

    batch_size, message_length = message_tokens.shape
    device = message_tokens.device

    # Create noise mask
    noise_mask = torch.rand(batch_size, message_length, device=device) < noise_level

    # Generate random tokens for noisy positions
    random_tokens = torch.randint(
        0, vocab_size, (batch_size, message_length), device=device
    )

    # Apply noise
    noisy_tokens = torch.where(noise_mask, random_tokens, message_tokens)

    return noisy_tokens


def compute_zipf_slope(message_tokens: torch.Tensor) -> float:
    """Compute Zipf slope from message token frequencies.

    This function analyzes the frequency distribution of tokens in messages
    and computes the slope of the Zipfian distribution to measure language
    structure emergence.

    Args:
        message_tokens: Tensor of shape (batch_size, message_length) with token indices.

    Returns:
        Zipf slope value (negative number, closer to -1 indicates Zipfian distribution).
    """
    # Flatten all tokens
    all_tokens = message_tokens.flatten().cpu().numpy()

    # Count frequencies
    unique_tokens, counts = np.unique(all_tokens, return_counts=True)

    if len(unique_tokens) < 2:
        return 0.0

    # Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    frequencies = counts[sorted_indices]

    # Compute ranks and log values
    ranks = np.arange(1, len(frequencies) + 1)
    log_ranks = np.log(ranks)
    log_frequencies = np.log(frequencies)

    # Fit linear regression to get slope
    if len(log_ranks) > 1:
        slope = np.polyfit(log_ranks, log_frequencies, 1)[0]
        return float(slope)
    else:
        return 0.0


def run_single_experiment(
    params: Dict[str, Any],
    experiment_id: str,
    base_seed: int = 42,
    n_steps: int = 5000,
    k: int = 5,
    message_length: int = 2,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    hidden_size: int = 128,
    entropy_weight: float = 0.05,
    heldout_pairs: Optional[List[Tuple[str, str]]] = None,
    temperature_start: float = 2.0,
    temperature_end: float = 0.5,
    use_sequence_models: bool = True,
) -> Dict[str, Any]:
    """Run a single ablation experiment with given parameters.

    This function trains a model with the specified parameters and evaluates
    its performance, returning comprehensive metrics including accuracy and
    Zipf slope measurements.

    Args:
        params: Parameter dictionary containing V, channel_noise, length_cost.
        experiment_id: Unique identifier for this experiment.
        base_seed: Base random seed (will be modified for each experiment).
        n_steps: Number of training steps.
        k: Number of objects per scene.
        message_length: Message length.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizers.
        hidden_size: Hidden dimension for neural networks.
        entropy_weight: Weight for entropy bonus regularization.
        heldout_pairs: List of held-out attribute pairs for compositional evaluation.

    Returns:
        Dictionary containing experiment results and metrics.
    """
    if heldout_pairs is None:
        heldout_pairs = [("blue", "triangle")]

    # Set unique seed for this experiment
    experiment_seed = base_seed + hash(experiment_id) % 10000
    set_seed(experiment_seed)

    logger.info(f"Running experiment {experiment_id} with params: {params}")

    # Create experiment directory
    exp_dir = f"outputs/experiments/{experiment_id}"
    os.makedirs(exp_dir, exist_ok=True)

    # Train model
    train(
        n_steps=n_steps,
        k=k,
        v=params["V"],
        message_length=message_length,
        seed=experiment_seed,
        log_every=200,
        eval_every=500,
        lambda_speaker=1.0,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        use_sequence_models=use_sequence_models,
        entropy_weight=entropy_weight,
        length_weight=params["length_cost"],
        heldout_pairs=heldout_pairs,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
    )

    # Evaluate on all splits
    checkpoint_path = "outputs/checkpoints/checkpoint.pt"
    results = evaluate_all_splits(
        model_path=checkpoint_path,
        heldout_pairs=heldout_pairs,
        n_scenes=1000,
        k=k,
        batch_size=batch_size,
    )

    # Compute Zipf slope from final messages
    zipf_slope = compute_zipf_slope_from_checkpoint(checkpoint_path, params["V"], k)

    # Prepare results
    experiment_results = {
        "experiment_id": experiment_id,
        "params": params,
        "seed": experiment_seed,
        "metrics": results,
        "zipf_slope": zipf_slope,
    }

    # Save results
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(experiment_results, f, indent=2)

    # Save Zipf slope separately
    zipf_path = os.path.join(exp_dir, "zipf_slope.json")
    with open(zipf_path, "w") as f:
        json.dump({"zipf_slope": zipf_slope}, f, indent=2)

    logger.info(
        f"Experiment {experiment_id} completed. Acc: {results['train']['acc']:.4f}, Zipf: {zipf_slope:.4f}"
    )

    return experiment_results


def compute_zipf_slope_from_checkpoint(
    checkpoint_path: str,
    vocab_size: int,
    k: int,
    n_samples: int = 1000,
) -> float:
    """Compute Zipf slope from a trained model checkpoint.

    This function loads a trained model and generates messages to analyze
    the Zipfian distribution of token usage patterns.

    Args:
        checkpoint_path: Path to the model checkpoint.
        vocab_size: Vocabulary size used in training.
        k: Number of objects per scene.
        n_samples: Number of samples to generate for analysis.

    Returns:
        Zipf slope value.
    """
    from ..core.agents import Speaker, SpeakerSeq
    from ..data.data import ReferentialGameDataset
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Create speaker based on the model type used in training
    if hasattr(config, "use_contrastive") and config.use_contrastive:
        from ..core.contrastive_agents import ContrastiveSpeaker

        speaker = ContrastiveSpeaker(config).to(device)
    elif hasattr(config, "use_sequence_models") and config.use_sequence_models:
        speaker = SpeakerSeq(config).to(device)
    else:
        speaker = Speaker(config).to(device)

    speaker.load_state_dict(checkpoint["speaker_state_dict"])
    speaker.eval()

    # Generate messages
    dataset = ReferentialGameDataset(n_samples, k, seed=42)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_messages = []

    with torch.no_grad():
        for batch in dataloader:
            scene_tensor, target_indices, _ = batch
            scene_tensor = scene_tensor.to(device)
            target_indices = target_indices.to(device)

            # Extract target objects
            batch_size = scene_tensor.size(0)
            target_objects = scene_tensor[torch.arange(batch_size), target_indices]

            # Generate messages
            speaker_output = speaker(target_objects)
            if len(speaker_output) == 4:
                _, message_tokens, _, _ = speaker_output
            else:
                _, message_tokens = speaker_output
            all_messages.append(message_tokens)

    # Concatenate all messages
    all_messages_tensor = torch.cat(all_messages, dim=0)

    # Compute Zipf slope
    return compute_zipf_slope(all_messages_tensor)


def run_ablation_suite(
    runs: int = 6,
    vocab_sizes: List[int] = [12, 24, 48],
    channel_noise_levels: List[float] = [0.0, 0.05, 0.1],
    length_costs: List[float] = [0.0, 0.01, 0.05],
    base_seed: int = 42,
    n_steps: int = 5000,
    k: int = 5,
    message_length: int = 2,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    hidden_size: int = 128,
    entropy_weight: float = 0.05,
    heldout_pairs: Optional[List[Tuple[str, str]]] = None,
    use_sequence_models: bool = True,
) -> List[Dict[str, Any]]:
    """Run complete ablation study suite.

    This function runs a systematic ablation study across all parameter
    combinations, training and evaluating models for each configuration.

    Args:
        runs: Number of runs to perform (currently unused, for future extension).
        vocab_sizes: List of vocabulary sizes to test.
        channel_noise_levels: List of channel noise levels to test.
        length_costs: List of length cost weights to test.
        base_seed: Base random seed for reproducibility.
        n_steps: Number of training steps per experiment.
        k: Number of objects per scene.
        message_length: Message length.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizers.
        hidden_size: Hidden dimension for neural networks.
        entropy_weight: Weight for entropy bonus regularization.
        heldout_pairs: List of held-out attribute pairs for compositional evaluation.

    Returns:
        List of experiment result dictionaries.
    """
    # Generate parameter grid
    param_grid = generate_parameter_grid(
        vocab_sizes, channel_noise_levels, length_costs
    )

    logger.info(f"Starting ablation suite with {len(param_grid)} experiments")

    # Create experiments directory
    os.makedirs("outputs/experiments", exist_ok=True)

    all_results = []

    # Run each experiment
    for i, params in enumerate(param_grid):
        experiment_id = f"exp_{i:03d}_V{params['V']}_noise{params['channel_noise']:.2f}_len{params['length_cost']:.2f}"

        try:
            result = run_single_experiment(
                params=params,
                experiment_id=experiment_id,
                base_seed=base_seed,
                n_steps=n_steps,
                k=k,
                message_length=message_length,
                batch_size=batch_size,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                entropy_weight=entropy_weight,
                heldout_pairs=heldout_pairs,
                use_sequence_models=use_sequence_models,
            )
            all_results.append(result)

        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            continue

    logger.info(
        f"Ablation suite completed. {len(all_results)}/{len(param_grid)} experiments successful"
    )

    return all_results
