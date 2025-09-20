"""Training module for emergent language experiments.

This module implements the training loop for referential games where language
emerges through interaction between Speaker and Listener agents. It includes
supervised learning for the Listener and REINFORCE for the Speaker.
"""

import os
import csv
from typing import Dict, Tuple, Optional, Union, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..core.agents import Speaker, Listener, SpeakerSeq, ListenerSeq
from ..core.config import CommunicationConfig
from ..data.data import ReferentialGameDataset, CompositionalDataset, DistractorDataset
from ..utils.utils import get_logger, get_device, set_seed

logger = get_logger(__name__)


class MovingAverage:
    """Exponential moving average baseline for REINFORCE training.

    This class maintains an exponential moving average of rewards to use as a baseline
    in REINFORCE training, providing better stability than simple moving averages.
    """

    def __init__(self, window_size: int = 100, alpha: float = 0.1):
        """Initialize the exponential moving average baseline.

        Args:
            window_size: Number of recent rewards to include in the average (for compatibility).
            alpha: Exponential decay factor (0 < alpha <= 1). Higher values give more weight to recent rewards.
        """
        self.window_size = window_size
        self.alpha = alpha
        self._average = 0.0
        self.count = 0

    def update(self, reward: float) -> None:
        """Update the exponential moving average with a new reward.

        Args:
            reward: The reward value to add to the moving average.
        """
        self.count += 1
        if self.count == 1:
            self._average = reward
        else:
            self._average = (1 - self.alpha) * self._average + self.alpha * reward

    @property
    def average(self) -> float:
        """Get the current exponential moving average."""
        return self._average


def compute_entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy bonus to encourage exploration in token distributions.

    This function computes the entropy of token distributions at each position
    to encourage the Speaker to maintain diversity in generated messages.

    Args:
        logits: Tensor of shape (batch_size, message_length, vocabulary_size) with logits.

    Returns:
        Scalar tensor containing the entropy bonus (negative entropy).
    """
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Compute entropy: -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)  # (batch_size, message_length)

    # Average across batch and sequence length
    entropy_bonus = entropy.mean()

    return entropy_bonus


def compute_length_cost(message_tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Compute length cost to penalize longer messages.

    This function computes a cost based on message length to encourage
    more concise communication when using variable-length messages.

    Args:
        message_tokens: Tensor of shape (batch_size, message_length) with token indices.
        vocab_size: Size of vocabulary (for EOS token detection).

    Returns:
        Scalar tensor containing the length cost.
    """
    batch_size, message_length = message_tokens.shape

    # For fixed-length messages, we don't apply length cost
    # This function is prepared for future variable-length extensions
    length_cost = torch.tensor(0.0, device=message_tokens.device)

    return length_cost


def compute_listener_loss(
    listener: Union[Listener, ListenerSeq],
    message_tokens: torch.Tensor,
    candidate_objects: torch.Tensor,
    target_indices: torch.Tensor,
    gesture_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute supervised cross-entropy loss for the Listener.

    This function computes the cross-entropy loss between the Listener's
    predicted probabilities over candidates and the true target indices.

    Args:
        listener: The Listener agent.
        message_tokens: Tensor of shape (batch_size, message_length) with message tokens.
        candidate_objects: Tensor of shape (batch_size, num_candidates, object_dim) with candidates.
        target_indices: Tensor of shape (batch_size,) with true target indices.

    Returns:
        Scalar tensor containing the cross-entropy loss.
    """
    # Get listener predictions
    if gesture_tokens is not None:
        probabilities = listener(message_tokens, candidate_objects, gesture_tokens)
    else:
        probabilities = listener(message_tokens, candidate_objects)

    # Compute cross-entropy loss
    loss = F.cross_entropy(probabilities, target_indices)

    return loss


def compute_speaker_loss(
    speaker: Union[Speaker, SpeakerSeq],
    speaker_logits: torch.Tensor,
    rewards: torch.Tensor,
    baseline: float,
    entropy_weight: float = 0.01,
    length_weight: float = 0.0,
) -> torch.Tensor:
    """Compute REINFORCE loss for the Speaker with regularization.

    This function computes the policy gradient loss using REINFORCE with
    a baseline to reduce variance in the gradient estimates, plus entropy
    bonus and length cost regularizers.

    Args:
        speaker: The Speaker agent.
        speaker_logits: Tensor of shape (batch_size, message_length, vocab_size) with logits.
        rewards: Tensor of shape (batch_size,) with rewards (1 if correct, 0 if incorrect).
        baseline: Baseline value for variance reduction.
        entropy_weight: Weight for entropy bonus regularization.
        length_weight: Weight for length cost regularization.

    Returns:
        Scalar tensor containing the REINFORCE loss with regularization.
    """
    batch_size, message_length, vocab_size = speaker_logits.shape

    # Compute log probabilities for each message position
    log_probs = F.log_softmax(speaker_logits, dim=-1)

    # For REINFORCE, we need the log probability of the sampled actions
    # Since we used argmax during forward pass, we need to recompute with Gumbel-Softmax
    # For simplicity, we'll use the log probabilities of the argmax actions
    # This is an approximation but works for the basic implementation

    # Get the most likely tokens (argmax)
    sampled_tokens = torch.argmax(
        speaker_logits, dim=-1
    )  # (batch_size, message_length)

    # Compute log probabilities of sampled actions
    log_probs_sampled = []
    for i in range(message_length):
        pos_log_probs = log_probs[:, i, :]  # (batch_size, vocab_size)
        pos_sampled = sampled_tokens[:, i]  # (batch_size,)
        pos_log_probs_sampled = pos_log_probs.gather(
            1, pos_sampled.unsqueeze(1)
        ).squeeze(1)
        log_probs_sampled.append(pos_log_probs_sampled)

    # Sum log probabilities across message positions
    total_log_probs = torch.stack(log_probs_sampled, dim=1).sum(dim=1)  # (batch_size,)

    # Compute REINFORCE loss with baseline
    advantages = rewards - baseline
    reinforce_loss = -(total_log_probs * advantages).mean()

    # Add regularization terms
    entropy_bonus = compute_entropy_bonus(speaker_logits)

    # Get sampled tokens for length cost (approximate with argmax)
    sampled_tokens = torch.argmax(speaker_logits, dim=-1)
    length_cost = compute_length_cost(sampled_tokens, vocab_size)

    # Combine losses
    total_loss = (
        reinforce_loss - entropy_weight * entropy_bonus + length_weight * length_cost
    )

    return total_loss


def train_step(
    speaker: Union[Speaker, SpeakerSeq],
    listener: Union[Listener, ListenerSeq],
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    speaker_optimizer: torch.optim.Optimizer,
    listener_optimizer: torch.optim.Optimizer,
    speaker_baseline: MovingAverage,
    config: CommunicationConfig,
    lambda_speaker: float = 1.0,
    entropy_weight: float = 0.01,
    length_weight: float = 0.0,
    device: Optional[torch.device] = None,
    temperature: float = 1.0,
    use_sequence_models: bool = False,
) -> Dict[str, float]:
    """Perform one training step for both Speaker and Listener.

    This function executes a single training step, computing losses for both
    agents and updating their parameters using the respective optimizers.
    Supports both regular and sequence-aware models.

    Args:
        speaker: The Speaker agent (Speaker or SpeakerSeq).
        listener: The Listener agent (Listener or ListenerSeq).
        batch: Tuple containing (scene_tensor, target_indices, candidate_objects).
        speaker_optimizer: Optimizer for the Speaker.
        listener_optimizer: Optimizer for the Listener.
        speaker_baseline: Moving average baseline for Speaker rewards.
        lambda_speaker: Weight for combining Speaker and Listener losses.
        entropy_weight: Weight for entropy bonus regularization.
        length_weight: Weight for length cost regularization.
        device: Device to run computations on.

    Returns:
        Dictionary containing loss values and accuracy metrics.
    """
    if device is None:
        device = get_device()

    scene_tensor, target_indices, candidate_objects = batch

    # Move tensors to device
    scene_tensor = scene_tensor.to(device)
    target_indices = target_indices.to(device)
    candidate_objects = candidate_objects.to(device)

    # Extract target objects from scene tensor
    batch_size = scene_tensor.size(0)
    target_objects = scene_tensor[torch.arange(batch_size), target_indices]

    # Speaker generates messages
    if config.multimodal:
        speaker_logits, message_tokens, gesture_logits, gesture_tokens = speaker(
            target_objects, temperature=temperature
        )
    else:
        if use_sequence_models:
            speaker_logits, message_tokens = speaker(
                target_objects, temperature=temperature
            )
            gesture_tokens = None
        else:
            speaker_logits, message_tokens, _, _ = speaker(
                target_objects, temperature=temperature
            )
            gesture_tokens = None

    # Listener makes predictions
    if config.multimodal and gesture_tokens is not None:
        listener_probs = listener(message_tokens, candidate_objects, gesture_tokens)
    else:
        listener_probs = listener(message_tokens, candidate_objects)
    listener_predictions = torch.argmax(listener_probs, dim=1)

    # Compute rewards (1 if correct, 0 if incorrect)
    rewards = (listener_predictions == target_indices).float()

    # Update speaker baseline
    avg_reward = rewards.mean().item()
    speaker_baseline.update(avg_reward)

    # Compute losses
    listener_loss = compute_listener_loss(
        listener, message_tokens, candidate_objects, target_indices, gesture_tokens
    )
    speaker_loss = compute_speaker_loss(
        speaker,
        speaker_logits,
        rewards,
        speaker_baseline.average,
        entropy_weight,
        length_weight,
    )

    # Combined loss
    total_loss = listener_loss + lambda_speaker * speaker_loss

    # Backward pass
    speaker_optimizer.zero_grad()
    listener_optimizer.zero_grad()
    total_loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(speaker.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(listener.parameters(), max_norm=1.0)

    speaker_optimizer.step()
    listener_optimizer.step()

    # Compute accuracy
    accuracy = rewards.mean().item()

    # Compute message length statistics
    # For fixed-length messages, the length is constant, but we can still track it
    avg_message_length = float(message_tokens.shape[1])  # message_length dimension
    message_length_std = 0.0  # For fixed-length messages, std is 0

    return {
        "total_loss": total_loss.item(),
        "listener_loss": listener_loss.item(),
        "speaker_loss": speaker_loss.item(),
        "accuracy": accuracy,
        "baseline": speaker_baseline.average,
        "avg_message_length": avg_message_length,
        "message_length_std": message_length_std,
    }


def train(
    n_steps: int,
    k: int,
    v: int,
    message_length: int,
    seed: int,
    log_every: int = 100,
    eval_every: int = 500,
    lambda_speaker: float = 1.0,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    use_sequence_models: bool = False,
    entropy_weight: float = 0.01,
    length_weight: float = 0.0,
    heldout_pairs: Optional[List[Tuple[str, str]]] = None,
    multimodal: bool = False,
    distractors: int = 0,
    temperature_start: float = 2.0,
    temperature_end: float = 0.5,
) -> None:
    """Train Speaker and Listener agents for emergent language.

    This function runs the main training loop for emergent language experiments,
    training both agents through interaction in referential games. Supports both
    regular and sequence-aware models, and compositional splits for generalization testing.

    Args:
        n_steps: Number of training steps to perform.
        k: Number of objects per scene.
        v: Vocabulary size.
        message_length: Message length.
        seed: Random seed for reproducibility.
        log_every: Frequency of logging training metrics.
        eval_every: Frequency of saving checkpoints.
        lambda_speaker: Weight for Speaker loss in combined objective.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizers.
        hidden_size: Hidden dimension for neural networks.
        use_sequence_models: Whether to use sequence-aware models (SpeakerSeq/ListenerSeq).
        entropy_weight: Weight for entropy bonus regularization.
        length_weight: Weight for length cost regularization.
        heldout_pairs: List of held-out attribute pairs for compositional splits.
        multimodal: Whether to enable multimodal communication with gestures.
        distractors: Number of distractor objects for pragmatic inference.
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Create output directories
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    # Get device
    device = get_device()
    logger.info(f"Training on device: {device}")

    # Create configuration
    config = CommunicationConfig(
        vocabulary_size=v,
        message_length=message_length,
        hidden_size=hidden_size,
        multimodal=multimodal,
        distractors=distractors,
        seed=seed,
    )

    # Create agents
    if use_sequence_models:
        speaker: Union[Speaker, SpeakerSeq] = SpeakerSeq(config).to(device)
        listener: Union[Listener, ListenerSeq] = ListenerSeq(config).to(device)
        logger.info("Using sequence-aware models (SpeakerSeq/ListenerSeq)")
    else:
        speaker = Speaker(config).to(device)
        listener = Listener(config).to(device)
        logger.info("Using regular models (Speaker/Listener)")

    # Create optimizers
    speaker_optimizer = torch.optim.Adam(speaker.parameters(), lr=learning_rate)
    listener_optimizer = torch.optim.Adam(listener.parameters(), lr=learning_rate)

    # Create baseline
    speaker_baseline = MovingAverage(window_size=100)

    # Create dataset
    if heldout_pairs is not None:
        from ..data.data import make_compositional_splits

        splits = make_compositional_splits(n_steps * batch_size, k, heldout_pairs, seed)
        dataset: Union[
            ReferentialGameDataset, CompositionalDataset, DistractorDataset
        ] = splits[
            "train"
        ]  # Use training split
        logger.info(f"Using compositional splits with heldout pairs: {heldout_pairs}")
        logger.info(f"Training set size: {len(dataset)}")
    else:
        if distractors > 0:
            dataset = DistractorDataset(
                n_scenes=n_steps * batch_size,
                k=k,
                num_distractors=distractors,
                seed=seed,
            )
            logger.info(
                f"Using distractor dataset with {distractors} distractors per scene"
            )
        else:
            dataset = ReferentialGameDataset(
                n_scenes=n_steps * batch_size, k=k, seed=seed
            )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    logger.info(f"Starting training for {n_steps} steps")
    logger.info(f"Configuration: K={k}, V={v}, L={message_length}, seed={seed}")

    # Initialize metrics logging
    metrics_file = "outputs/logs/metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "total_loss",
                "listener_loss",
                "speaker_loss",
                "accuracy",
                "baseline",
                "avg_message_length",
                "message_length_std",
            ]
        )

    step = 0
    dataloader_iter = iter(dataloader)

    while step < n_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Restart dataloader if we run out of data
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # Compute temperature annealing
        progress = step / n_steps
        temperature = temperature_start * (1 - progress) + temperature_end * progress

        # Training step
        metrics = train_step(
            speaker,
            listener,
            batch,
            speaker_optimizer,
            listener_optimizer,
            speaker_baseline,
            config,
            lambda_speaker,
            entropy_weight,
            length_weight,
            device,
            temperature,
            use_sequence_models,
        )

        step += 1

        # Logging
        if step % log_every == 0:
            logger.info(
                f"Step {step}: Loss={metrics['total_loss']:.4f}, "
                f"Acc={metrics['accuracy']:.4f}, Baseline={metrics['baseline']:.4f}"
            )

        # Save metrics
        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    step,
                    metrics["total_loss"],
                    metrics["listener_loss"],
                    metrics["speaker_loss"],
                    metrics["accuracy"],
                    metrics["baseline"],
                    metrics["avg_message_length"],
                    metrics["message_length_std"],
                ]
            )

        # Save checkpoint
        if step % eval_every == 0:
            checkpoint_path = f"outputs/checkpoints/checkpoint_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "speaker_state_dict": speaker.state_dict(),
                    "listener_state_dict": listener.state_dict(),
                    "speaker_optimizer_state_dict": speaker_optimizer.state_dict(),
                    "listener_optimizer_state_dict": listener_optimizer.state_dict(),
                    "config": config,
                    "metrics": metrics,
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint at step {step}")

    # Save final checkpoint
    final_checkpoint_path = "outputs/checkpoints/checkpoint.pt"
    torch.save(
        {
            "step": step,
            "speaker_state_dict": speaker.state_dict(),
            "listener_state_dict": listener.state_dict(),
            "speaker_optimizer_state_dict": speaker_optimizer.state_dict(),
            "listener_optimizer_state_dict": listener_optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        final_checkpoint_path,
    )

    logger.info(f"Training completed. Final accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Metrics saved to: {metrics_file}")
    logger.info(f"Final checkpoint saved to: {final_checkpoint_path}")
