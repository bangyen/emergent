"""Improved training procedures for emergent language models.

This module implements advanced training techniques including:
- Advanced optimization strategies
- Better loss functions
- Curriculum learning
- Advanced regularization
- Learning rate scheduling
- Early stopping and model selection
"""

import os
import json
import math
from typing import Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from ..core.agents import Speaker, Listener, SpeakerSeq, ListenerSeq
from ..core.improved_agents import (
    ImprovedSpeaker,
    ImprovedListener,
    ImprovedSpeakerSeq,
    ImprovedListenerSeq,
)
from ..core.config import CommunicationConfig
from ..data.data import ReferentialGameDataset
from ..utils.utils import get_logger, get_device, set_seed
from .train import MovingAverage, EarlyStopping

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for improved training."""

    # Model architecture
    use_improved_models: bool = True
    use_sequence_models: bool = False

    # Training parameters
    n_steps: int = 5000
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Advanced optimization
    use_cosine_annealing: bool = True
    use_warmup: bool = True
    warmup_steps: int = 500
    use_ema: bool = True
    ema_decay: float = 0.999

    # Loss function improvements
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    use_label_smoothing: bool = True
    label_smoothing: float = 0.1

    # Regularization
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_steps: int = 1000
    difficulty_schedule: str = "linear"  # "linear", "exponential", "cosine"

    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 100
    early_stopping_min_delta: float = 0.001

    # Evaluation
    eval_every: int = 200
    log_every: int = 50

    # Other parameters
    k: int = 5
    v: int = 16
    message_length: int = 2
    hidden_size: int = 128
    seed: int = 42


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()  # type: ignore[no-any-return]
        elif self.reduction == "sum":
            return focal_loss.sum()  # type: ignore[no-any-return]
        else:
            return focal_loss  # type: ignore[no-any-return]


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing cross-entropy loss."""
        log_prob = F.log_softmax(inputs, dim=1)
        weight = (
            inputs.new_ones(inputs.size()) * self.smoothing / (inputs.size(-1) - 1.0)
        )
        weight.scatter_(-1, targets.unsqueeze(-1), (1.0 - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


class MixUp:
    """MixUp data augmentation."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam  # type: ignore[return-value]


class CurriculumScheduler:
    """Curriculum learning scheduler."""

    def __init__(self, total_steps: int, schedule_type: str = "linear"):
        self.total_steps = total_steps
        self.schedule_type = schedule_type

    def get_difficulty(self, step: int) -> float:
        """Get current difficulty level (0.0 to 1.0)."""
        progress = min(step / self.total_steps, 1.0)

        if self.schedule_type == "linear":
            return progress
        elif self.schedule_type == "exponential":
            return 1 - math.exp(-3 * progress)
        elif self.schedule_type == "cosine":
            return 0.5 * (1 - math.cos(math.pi * progress))
        else:
            return progress


def create_improved_optimizer(
    model: nn.Module, config: TrainingConfig
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """Create improved optimizer with advanced scheduling."""

    # Use AdamW with improved parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Create learning rate scheduler
    if config.use_cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.n_steps, eta_min=config.learning_rate * 0.01
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(  # type: ignore[assignment]
            optimizer, step_size=config.n_steps // 3, gamma=0.1
        )

    # Add warmup if enabled
    if config.use_warmup:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=config.warmup_steps
        )
        scheduler = optim.lr_scheduler.SequentialLR(  # type: ignore[assignment]
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[config.warmup_steps],
        )

    return optimizer, scheduler  # type: ignore[return-value]


def compute_improved_listener_loss(
    listener: Union[Listener, ListenerSeq, ImprovedListener, ImprovedListenerSeq],
    message_tokens: torch.Tensor,
    candidate_objects: torch.Tensor,
    target_indices: torch.Tensor,
    gesture_tokens: Optional[torch.Tensor] = None,
    use_focal_loss: bool = True,
    use_label_smoothing: bool = True,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Compute improved listener loss with advanced techniques."""

    # Get predictions
    if gesture_tokens is not None:
        listener_probs = listener(message_tokens, candidate_objects, gesture_tokens)
    else:
        listener_probs = listener(message_tokens, candidate_objects)

    # Convert to logits for loss computation
    logits = torch.log(listener_probs + 1e-8)

    # Choose loss function
    if use_focal_loss:
        loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        loss = loss_fn(logits, target_indices)
    elif use_label_smoothing:
        loss_fn = LabelSmoothingCrossEntropy(smoothing=label_smoothing)  # type: ignore[assignment]
        loss = loss_fn(logits, target_indices)
    else:
        loss = F.cross_entropy(logits, target_indices)

    return loss  # type: ignore[no-any-return]


def compute_improved_speaker_loss(
    speaker: Union[Speaker, SpeakerSeq, ImprovedSpeaker, ImprovedSpeakerSeq],
    speaker_logits: torch.Tensor,
    rewards: torch.Tensor,
    baseline: float,
    entropy_weight: float = 0.01,
    length_weight: float = 0.0,
    use_entropy_bonus: bool = True,
) -> torch.Tensor:
    """Compute improved speaker loss with better regularization."""

    # Policy gradient loss
    policy_loss = -torch.mean(
        (rewards - baseline) * torch.log(torch.sum(speaker_logits, dim=-1) + 1e-8)
    )

    # Entropy bonus for exploration
    if use_entropy_bonus:
        entropy = -torch.sum(speaker_logits * torch.log(speaker_logits + 1e-8), dim=-1)
        entropy_bonus = -entropy_weight * torch.mean(entropy)
    else:
        entropy_bonus = torch.tensor(0.0)  # type: ignore[assignment]

    # Length penalty
    if length_weight > 0:
        message_lengths = torch.sum(
            torch.argmax(speaker_logits, dim=-1) != 0, dim=1
        ).float()
        length_penalty = length_weight * torch.mean(message_lengths)
    else:
        length_penalty = torch.tensor(0.0)  # type: ignore[assignment]

    total_loss = policy_loss + entropy_bonus + length_penalty
    return total_loss


def improved_train_step(
    speaker: Union[Speaker, SpeakerSeq, ImprovedSpeaker, ImprovedSpeakerSeq],
    listener: Union[Listener, ListenerSeq, ImprovedListener, ImprovedListenerSeq],
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    speaker_optimizer: optim.Optimizer,
    listener_optimizer: optim.Optimizer,
    speaker_baseline: MovingAverage,
    config: TrainingConfig,
    step: int,
    mixup: Optional[MixUp] = None,
    curriculum_scheduler: Optional[CurriculumScheduler] = None,
) -> Dict[str, float]:
    """Improved training step with advanced techniques."""

    scene_tensor, target_indices, candidate_objects = batch
    batch_size = scene_tensor.size(0)

    # Extract target objects
    target_objects = scene_tensor[torch.arange(batch_size), target_indices]

    # Apply curriculum learning if enabled
    if curriculum_scheduler is not None:
        difficulty = curriculum_scheduler.get_difficulty(step)
        # Adjust batch complexity based on difficulty
        if difficulty < 0.5:
            # Use simpler examples
            pass  # For now, just use full batch

    # Apply MixUp if enabled
    if mixup is not None and np.random.random() < 0.5:
        mixup_result = mixup(target_objects, target_indices)
        target_objects, y_a, y_b, lam = mixup_result  # type: ignore[misc]
        use_mixup = True
    else:
        y_a, y_b, lam = target_indices, target_indices, 1.0
        use_mixup = False

    # Speaker generates messages
    if hasattr(speaker, "config") and speaker.config.use_sequence_models:
        speaker_logits, message_tokens = speaker(target_objects)
    else:
        speaker_logits, message_tokens, _, _ = speaker(target_objects)

    # Listener makes predictions
    listener_probs = listener(message_tokens, candidate_objects)
    listener_predictions = torch.argmax(listener_probs, dim=1)

    # Compute rewards
    rewards = (listener_predictions == target_indices).float()

    # Update speaker baseline
    avg_reward = rewards.mean().item()
    speaker_baseline.update(avg_reward)

    # Compute losses
    listener_loss = compute_improved_listener_loss(
        listener,
        message_tokens,
        candidate_objects,
        target_indices,
        use_focal_loss=config.use_focal_loss,
        use_label_smoothing=config.use_label_smoothing,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
        label_smoothing=config.label_smoothing,
    )

    speaker_loss = compute_improved_speaker_loss(
        speaker,
        speaker_logits,
        rewards,
        speaker_baseline.average,
        use_entropy_bonus=True,
    )

    # Handle MixUp loss
    if use_mixup:
        # Compute loss for both original and mixed targets
        listener_loss_a = compute_improved_listener_loss(
            listener,
            message_tokens,
            candidate_objects,
            y_a,
            use_focal_loss=config.use_focal_loss,
            use_label_smoothing=config.use_label_smoothing,
        )
        listener_loss_b = compute_improved_listener_loss(
            listener,
            message_tokens,
            candidate_objects,
            y_b,
            use_focal_loss=config.use_focal_loss,
            use_label_smoothing=config.use_label_smoothing,
        )
        listener_loss = lam * listener_loss_a + (1 - lam) * listener_loss_b

    # Combined loss
    total_loss = listener_loss + speaker_loss

    # Backward pass
    speaker_optimizer.zero_grad()
    listener_optimizer.zero_grad()
    total_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(speaker.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(listener.parameters(), max_norm=1.0)

    # Optimizer step
    speaker_optimizer.step()
    listener_optimizer.step()

    # Compute metrics
    accuracy = rewards.mean().item()
    message_length = torch.mean(
        torch.sum(torch.argmax(speaker_logits, dim=-1) != 0, dim=1).float()
    ).item()

    return {
        "total_loss": total_loss.item(),
        "listener_loss": listener_loss.item(),
        "speaker_loss": speaker_loss.item(),
        "accuracy": accuracy,
        "baseline": speaker_baseline.average,
        "message_length": message_length,
    }


def train_improved_model(config: TrainingConfig) -> Dict[str, Any]:
    """Train model with improved techniques."""

    # Set seed
    set_seed(config.seed)

    # Create output directories
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    # Get device
    device = get_device()
    logger.info(f"Training on device: {device}")

    # Create communication config
    comm_config = CommunicationConfig(
        vocabulary_size=config.v,
        message_length=config.message_length,
        hidden_size=config.hidden_size,
        use_sequence_models=config.use_sequence_models,
        seed=config.seed,
    )

    # Create models
    if config.use_improved_models:
        if config.use_sequence_models:
            speaker = ImprovedSpeakerSeq(comm_config).to(device)
            listener = ImprovedListenerSeq(comm_config).to(device)
            logger.info("Using improved sequence models")
        else:
            speaker = ImprovedSpeaker(comm_config).to(device)  # type: ignore[assignment]
            listener = ImprovedListener(comm_config).to(device)  # type: ignore[assignment]
            logger.info("Using improved regular models")
    else:
        if config.use_sequence_models:
            speaker = SpeakerSeq(comm_config).to(device)  # type: ignore[assignment]
            listener = ListenerSeq(comm_config).to(device)  # type: ignore[assignment]
            logger.info("Using regular sequence models")
        else:
            speaker = Speaker(comm_config).to(device)  # type: ignore[assignment]
            listener = Listener(comm_config).to(device)  # type: ignore[assignment]
            logger.info("Using regular models")

    # Create optimizers
    speaker_optimizer, speaker_scheduler = create_improved_optimizer(speaker, config)
    listener_optimizer, listener_scheduler = create_improved_optimizer(listener, config)

    # Create training components
    speaker_baseline = MovingAverage()
    early_stopping = (
        EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )
        if config.use_early_stopping
        else None
    )

    # Create data augmentation
    mixup = MixUp(alpha=config.mixup_alpha) if config.use_mixup else None
    curriculum_scheduler = (
        CurriculumScheduler(config.curriculum_steps, config.difficulty_schedule)
        if config.use_curriculum
        else None
    )

    # Create dataset
    dataset = ReferentialGameDataset(
        n_scenes=config.n_steps * config.batch_size, k=config.k, seed=config.seed
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Training loop
    best_accuracy = 0.0
    training_logs = []

    logger.info(f"Starting improved training for {config.n_steps} steps...")

    for step in range(1, config.n_steps + 1):
        # Get batch
        batch = next(iter(dataloader))
        batch = [b.to(device) for b in batch]

        # Training step
        metrics = improved_train_step(
            speaker,
            listener,
            batch,  # type: ignore[arg-type]
            speaker_optimizer,
            listener_optimizer,
            speaker_baseline,
            config,
            step,
            mixup,
            curriculum_scheduler,
        )

        # Update learning rates
        speaker_scheduler.step()
        listener_scheduler.step()

        # Logging
        if step % config.log_every == 0:
            logger.info(
                f"Step {step}: Loss={metrics['total_loss']:.4f}, "
                f"Acc={metrics['accuracy']:.4f}, "
                f"Baseline={metrics['baseline']:.4f}"
            )

        # Evaluation and checkpointing
        if step % config.eval_every == 0:
            # Evaluate on validation set
            val_dataset = ReferentialGameDataset(
                n_scenes=1000, k=config.k, seed=config.seed + 1000
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=config.batch_size, shuffle=False
            )

            val_accuracy = 0.0
            val_samples = 0

            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_batch = [b.to(device) for b in val_batch]
                    val_scene, val_targets, val_candidates = val_batch
                    val_batch_size = val_scene.size(0)
                    val_target_objects = val_scene[
                        torch.arange(val_batch_size), val_targets
                    ]

                    # Generate messages
                    if (
                        hasattr(speaker, "config")
                        and speaker.config.use_sequence_models
                    ):
                        _, val_messages = speaker(val_target_objects)
                    else:
                        _, val_messages, _, _ = speaker(val_target_objects)

                    # Get predictions
                    val_probs = listener(val_messages, val_candidates)
                    val_preds = torch.argmax(val_probs, dim=1)

                    # Compute accuracy
                    val_correct = (val_preds == val_targets).sum().item()
                    val_accuracy += val_correct
                    val_samples += val_batch_size

            val_accuracy /= val_samples

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                checkpoint = {
                    "step": step,
                    "speaker_state_dict": speaker.state_dict(),
                    "listener_state_dict": listener.state_dict(),
                    "config": comm_config,
                    "accuracy": val_accuracy,
                    "training_config": config.__dict__,
                }
                torch.save(checkpoint, "outputs/checkpoints/improved_checkpoint.pt")
                logger.info(f"New best model saved with accuracy: {val_accuracy:.4f}")

            # Early stopping
            if early_stopping is not None:
                if early_stopping(val_accuracy):  # type: ignore[call-arg]
                    logger.info(f"Early stopping at step {step}")
                    break

        # Store metrics
        training_logs.append(
            {
                "step": step,
                **metrics,
                "learning_rate": speaker_optimizer.param_groups[0]["lr"],
            }
        )

    # Save final results
    results = {
        "best_accuracy": best_accuracy,
        "final_step": step,
        "training_logs": training_logs,
        "config": config.__dict__,
    }

    with open("outputs/logs/improved_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training completed. Best accuracy: {best_accuracy:.4f}")
    return results


def main() -> None:
    """Main function for improved training."""
    config = TrainingConfig(
        n_steps=5000,
        batch_size=64,
        learning_rate=1e-3,
        use_improved_models=True,
        use_sequence_models=False,
        use_focal_loss=True,
        use_label_smoothing=True,
        use_mixup=True,
        use_curriculum=True,
        use_early_stopping=True,
    )

    results = train_improved_model(config)
    print(f"Training completed with best accuracy: {results['best_accuracy']:.4f}")


if __name__ == "__main__":
    main()
