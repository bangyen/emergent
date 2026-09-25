"""Simplified training module for emergent language experiments.

This module implements the core training loop for referential games where language
emerges through interaction between Speaker and Listener agents.
"""

import csv
import os
from itertools import islice
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..analysis.eval import accuracy, build_agents
from ..analysis.language import language_metrics
from ..core.config import CommunicationConfig
from ..data.data import (
    ReferentialGameDataset,
    SceneStream,
    make_compositional_splits,
    make_heldout_target_dataset,
)
from ..data.world import World, get_world
from ..utils.utils import get_device, get_logger, set_seed

logger = get_logger(__name__)

# Offset keeping evaluation scenes on a different seed from the training stream.
EVAL_SEED_OFFSET = 1_000_003

LANGUAGE_FIELDS = ["topsim", "posdis", "msg_entropy", "n_messages"]
EVAL_FIELDS = ["iid_acc", "compo_acc", "compo_target_acc"] + LANGUAGE_FIELDS
METRIC_FIELDS = [
    "step",
    "total_loss",
    "listener_loss",
    "speaker_loss",
    "accuracy",
    "baseline",
] + EVAL_FIELDS


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
    tokens: torch.Tensor,
    rewards: torch.Tensor,
    baseline: float,
    entropy_weight: float = 0.01,
) -> torch.Tensor:
    """REINFORCE loss with entropy regularization.

    Args:
        logits: (batch, message_length, vocab) speaker logits.
        tokens: (batch, message_length) tokens the speaker actually sent.
        rewards: (batch,) per-example rewards.
        baseline: Scalar baseline subtracted from rewards.
        entropy_weight: Weight of the entropy bonus.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_sent = log_probs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
    total_log_probs = log_probs_sent.sum(dim=1)

    advantages = rewards - baseline
    reinforce_loss = -(total_log_probs * advantages).mean()

    # Entropy bonus
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    return reinforce_loss - entropy_weight * entropy


def game_step(
    speaker: Any,
    listener: Any,
    speaker_opt: torch.optim.Optimizer,
    listener_opt: torch.optim.Optimizer,
    baseline: MovingAverage,
    scene: torch.Tensor,
    targets: torch.Tensor,
    entropy_weight: float,
) -> Dict[str, float]:
    """Play one batch of the referential game and update both agents."""
    target_objs = scene[torch.arange(scene.size(0)), targets]
    speaker_output = speaker(target_objs)
    listener_output = listener(speaker_output.tokens, scene)

    # Rewards and Loss (baseline excludes the current batch)
    rewards = (listener_output.preds == targets).float()
    l_loss = F.nll_loss(torch.log(listener_output.probs + 1e-8), targets)
    s_loss = compute_speaker_loss(
        speaker_output.logits,
        speaker_output.tokens,
        rewards,
        baseline.average,
        entropy_weight,
    )
    baseline.update(rewards.mean().item())
    total_loss = l_loss + s_loss

    speaker_opt.zero_grad()
    listener_opt.zero_grad()
    total_loss.backward()
    speaker_opt.step()
    listener_opt.step()

    return {
        "total_loss": total_loss.item(),
        "listener_loss": l_loss.item(),
        "speaker_loss": s_loss.item(),
        "accuracy": rewards.mean().item(),
        "baseline": baseline.average,
    }


def build_eval_sets(
    k: int,
    seed: int,
    n_eval: int,
    heldout_pairs: Optional[Sequence[Tuple[str, str]]],
    world: World,
) -> Dict[str, Dataset]:
    """Fixed evaluation sets.

    ``iid`` always; with held-out pairs also ``compo`` (scenes containing a
    held-out object) and ``compo_target`` (the target itself is held out).
    """
    eval_seed = seed + EVAL_SEED_OFFSET
    if heldout_pairs:
        # Scale up so the iid/compo portions (20% each) hold ~n_eval scenes.
        splits = make_compositional_splits(
            5 * n_eval, k, list(heldout_pairs), eval_seed, world=world
        )
        return {
            "iid": splits["iid"],
            "compo": splits["compo"],
            "compo_target": make_heldout_target_dataset(
                n_eval, k, heldout_pairs, eval_seed, world=world
            ),
        }
    return {"iid": ReferentialGameDataset(n_eval, k, seed=eval_seed, world=world)}


def evaluate_agents(
    speaker: Any,
    listener: Any,
    eval_sets: Dict[str, Dataset],
    world: World,
    device: torch.device,
) -> Dict[str, float]:
    """Accuracy on every eval set plus lexicon metrics for the speaker."""
    metrics = {
        f"{name}_acc": accuracy(speaker, listener, ds, device)
        for name, ds in eval_sets.items()
    }
    metrics.update(language_metrics(speaker, world, device))
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    return " | ".join(f"{name}: {value:.3f}" for name, value in metrics.items())


def train(
    n_steps: int = 10000,
    k: int = 5,
    v: int = 16,
    message_length: int = 2,
    seed: int = 7,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
    hidden_size: int = 128,
    use_sequence_models: bool = False,
    entropy_weight: float = 0.01,
    heldout_pairs: Optional[List[Tuple[str, str]]] = None,
    out_dir: str = "outputs",
    eval_every: int = 500,
    n_eval: int = 1000,
    log_every: int = 100,
    world: str = "default",
    hard_distractors: int = 0,
    listener_type: str = "mlp",
) -> Dict[str, float]:
    """Core training loop for emergent language.

    Trains on an endless stream of scenes that never contains held-out objects,
    periodically evaluates on fixed eval sets (see :func:`build_eval_sets`) and
    computes lexicon metrics, writing ``metrics.csv`` and
    ``checkpoints/final_model.pt`` under ``out_dir``.

    Returns:
        Final evaluation metrics, e.g. ``{"iid_acc": ..., "topsim": ...}``.
    """
    device = get_device()
    world_def = get_world(world)
    eval_sets = build_eval_sets(k, seed, n_eval, heldout_pairs, world_def)

    set_seed(seed)
    config = CommunicationConfig(
        vocabulary_size=v,
        message_length=message_length,
        hidden_size=hidden_size,
        object_dim=world_def.dim,
        use_sequence_models=use_sequence_models,
        listener_type=listener_type,
        seed=seed,
    )
    speaker, listener = build_agents(config, device)

    speaker_opt = torch.optim.Adam(speaker.parameters(), lr=learning_rate)
    listener_opt = torch.optim.Adam(listener.parameters(), lr=learning_rate)

    baseline = MovingAverage()
    stream = SceneStream(
        k,
        seed=seed,
        heldout_pairs=heldout_pairs,
        world=world_def,
        hard_distractors=hard_distractors,
    )
    dataloader = DataLoader(stream, batch_size=batch_size)

    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.csv")

    speaker.train()
    listener.train()

    final: Dict[str, float] = {}
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()

        for step, (scene, targets) in enumerate(islice(dataloader, n_steps), 1):
            stats = game_step(
                speaker,
                listener,
                speaker_opt,
                listener_opt,
                baseline,
                scene.to(device),
                targets.to(device),
                entropy_weight,
            )

            is_eval = step % eval_every == 0 or step == n_steps
            if step % log_every == 0 or is_eval:
                row: Dict[str, float] = {"step": step, **stats}
                if is_eval:
                    final = evaluate_agents(
                        speaker, listener, eval_sets, world_def, device
                    )
                    row.update(final)
                writer.writerow(row)
                f.flush()
                logger.info(
                    f"Step {step}/{n_steps} | Loss: {stats['total_loss']:.4f} | "
                    f"Acc: {stats['accuracy']:.3f}"
                    + (f" | {format_metrics(final)}" if is_eval else "")
                )

    ckpt_path = os.path.join(out_dir, "checkpoints", "final_model.pt")
    torch.save(
        {
            "speaker_state_dict": speaker.state_dict(),
            "listener_state_dict": listener.state_dict(),
            "config": config,
            "k": k,
            "heldout_pairs": heldout_pairs,
            "world": world,
            "metrics": final,
        },
        ckpt_path,
    )
    logger.info(f"Training complete. Model saved to {ckpt_path}")
    return final
