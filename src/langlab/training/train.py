"""Simplified training module for emergent language experiments.

This module implements the core training loop for referential games where language
emerges through interaction between Speaker and Listener agents.
"""

import os
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..core.agents import Listener, ListenerSeq, Speaker, SpeakerSeq
from ..core.config import CommunicationConfig
from ..data.data import ReferentialGameDataset
from ..utils.utils import get_device, get_logger, set_seed

logger = get_logger(__name__)


class MovingAverage:
    """Simple exponential moving average baseline for REINFORCE."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._average = 0.0
        self.count = 0

    def update(self, reward: float) -> None:
        self.count += 1
        if self.count == 1:
            self._average = reward
        else:
            self._average = (1 - self.alpha) * self._average + self.alpha * reward

    @property
    def average(self) -> float:
        return self._average


def compute_speaker_loss(
    logits: torch.Tensor,
    rewards: torch.Tensor,
    baseline: float,
    entropy_weight: float = 0.01,
) -> torch.Tensor:
    """REINFORCE loss with entropy regularization."""
    log_probs = F.log_softmax(logits, dim=-1)
    sampled_tokens = torch.argmax(logits, dim=-1)

    # Gather log probs of sampled tokens
    log_probs_sampled = log_probs.gather(2, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    total_log_probs = log_probs_sampled.sum(dim=1)

    advantages = rewards - baseline
    reinforce_loss = -(total_log_probs * advantages).mean()

    # Entropy bonus
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    return reinforce_loss - entropy_weight * entropy


def train(
    n_steps: int,
    k: int,
    v: int,
    message_length: int,
    seed: int = 7,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
    hidden_size: int = 128,
    use_sequence_models: bool = False,
    entropy_weight: float = 0.01,
    heldout_pairs: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Core training loop for emergent language."""
    set_seed(seed)
    device = get_device()

    config = CommunicationConfig(
        vocabulary_size=v,
        message_length=message_length,
        hidden_size=hidden_size,
        use_sequence_models=use_sequence_models,
        seed=seed,
    )

    speaker: Any
    listener: Any

    if use_sequence_models:
        speaker = SpeakerSeq(config).to(device)
        listener = ListenerSeq(config).to(device)
    else:
        speaker = Speaker(config).to(device)
        listener = Listener(config).to(device)

    speaker_opt = torch.optim.Adam(speaker.parameters(), lr=learning_rate)
    listener_opt = torch.optim.Adam(listener.parameters(), lr=learning_rate)

    baseline = MovingAverage()
    dataset = ReferentialGameDataset(n_steps * batch_size, k, seed=seed)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    speaker.train()
    listener.train()

    for step, batch in enumerate(dataloader):
        if step >= n_steps:
            break

        scene, targets, _ = batch
        scene, targets = scene.to(device), targets.to(device)

        # Speaker
        target_objs = scene[torch.arange(batch_size), targets]
        speaker_output = speaker(target_objs)
        speaker_logits, message_tokens = speaker_output[0], speaker_output[1]

        # Listener
        listener_probs = listener(message_tokens, scene)
        preds = torch.argmax(listener_probs, dim=1)

        # Rewards and Loss
        rewards = (preds == targets).float()
        baseline.update(rewards.mean().item())

        l_loss = F.cross_entropy(torch.log(listener_probs + 1e-8), targets)
        s_loss = compute_speaker_loss(
            speaker_logits, rewards, baseline.average, entropy_weight
        )

        total_loss = l_loss + s_loss

        # Update
        speaker_opt.zero_grad()
        listener_opt.zero_grad()
        total_loss.backward()
        speaker_opt.step()
        listener_opt.step()

        if step % 100 == 0:
            logger.info(
                f"Step {step}/{n_steps} | Loss: {total_loss.item():.4f} | Acc: {rewards.mean().item():.4f}"
            )

    # Save final model
    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save(
        {
            "speaker_state_dict": speaker.state_dict(),
            "listener_state_dict": listener.state_dict(),
            "config": config,
        },
        "outputs/checkpoints/final_model.pt",
    )
    logger.info("Training complete. Model saved to outputs/checkpoints/final_model.pt")
