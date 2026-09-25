"""Population training with generational turnover (cultural transmission).

A population of ``n_agents`` speakers and ``n_agents`` listeners plays the
referential game in randomly drawn speaker/listener pairs, so a shared
language must emerge across the whole community rather than within one pair.

With ``lifespan > 0``, every ``lifespan`` steps the oldest agent is replaced by
a freshly initialised one (cycling through speakers and listeners). Newcomers
must learn the language from the survivors, which is the iterated-learning
pressure thought to favour compositional, easy-to-learn languages.
"""

import csv
import os
import random
from itertools import islice
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..analysis.eval import accuracy, build_agents
from ..analysis.language import language_metrics, lexicon
from ..core.config import CommunicationConfig
from ..data.data import SceneStream
from ..data.world import get_world
from ..utils.utils import get_device, get_logger, set_seed
from .train import (
    LANGUAGE_FIELDS,
    MovingAverage,
    build_eval_sets,
    format_metrics,
    game_step,
)

logger = get_logger(__name__)

POP_FIELDS = [
    "step",
    "generation",
    "total_loss",
    "accuracy",
    "iid_acc",
    "iid_acc_min",
    "compo_acc",
    "compo_target_acc",
    "agreement",
] + LANGUAGE_FIELDS


class _Agent:
    """A model with its own optimizer (and baseline, for speakers)."""

    def __init__(self, model: Any, learning_rate: float):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.baseline = MovingAverage()


def train_population(
    n_steps: int = 10000,
    k: int = 5,
    v: int = 16,
    message_length: int = 2,
    n_agents: int = 3,
    lifespan: int = 0,
    n_listeners: Optional[int] = None,
    seed: int = 7,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
    hidden_size: int = 128,
    use_sequence_models: bool = False,
    entropy_weight: float = 0.01,
    heldout_pairs: Optional[List[Tuple[str, str]]] = None,
    out_dir: str = "outputs/population",
    eval_every: int = 500,
    n_eval: int = 1000,
    log_every: int = 100,
    world: str = "default",
    hard_distractors: int = 0,
) -> Dict[str, float]:
    """Train a population of agents in random pairings.

    Args:
        n_agents: Number of speakers (and of listeners unless ``n_listeners``).
        lifespan: Every ``lifespan`` steps the oldest agent is replaced
            (speakers and listeners interleaved); 0 disables turnover.
        n_listeners: Number of listeners; e.g. 1 makes all speakers talk to a
            single shared listener.

    Evaluation (run before any replacement at the same step, so each
    generation is measured at its end) plays every speaker against every
    listener: ``iid_acc`` is the mean over all pairs and ``iid_acc_min`` the
    worst pair. ``agreement`` is the fraction of objects on which two speakers
    send the same message, averaged over speaker pairs; language metrics are
    averaged over speakers. ``topsim_initial`` is the TopSim at the first
    evaluation, to compare against the final ``topsim``.

    Returns:
        Final evaluation metrics.
    """
    if n_listeners is None:
        n_listeners = n_agents
    if n_agents < 1 or n_listeners < 1:
        raise ValueError("need at least one speaker and one listener")
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
        seed=seed,
    )

    def new_pair() -> Tuple[_Agent, _Agent]:
        speaker, listener = build_agents(config, device)
        return _Agent(speaker, learning_rate), _Agent(listener, learning_rate)

    pairs = [new_pair() for _ in range(max(n_agents, n_listeners))]
    speakers = [p[0] for p in pairs[:n_agents]]
    listeners = [p[1] for p in pairs[:n_listeners]]
    # Replacement order, oldest first: speaker 0, listener 0, speaker 1, ...
    slots = [
        (role, i)
        for i in range(max(n_agents, n_listeners))
        for role, count in (("speaker", n_agents), ("listener", n_listeners))
        if i < count
    ]

    pairing_rng = random.Random(seed)
    stream = SceneStream(
        k,
        seed=seed,
        heldout_pairs=heldout_pairs,
        world=world_def,
        hard_distractors=hard_distractors,
    )
    dataloader = DataLoader(stream, batch_size=batch_size)

    def run_eval() -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name, ds in eval_sets.items():
            accs = [
                accuracy(s.model, lis.model, ds, device)
                for s in speakers
                for lis in listeners
            ]
            results[f"{name}_acc"] = mean(accs)
            if name == "iid":
                results["iid_acc_min"] = min(accs)
        lexicons = [lexicon(s.model, world_def, device) for s in speakers]
        results["agreement"] = mean(
            [
                mean(float(a == b) for a, b in zip(lexicons[i], lexicons[j]))
                for i in range(len(lexicons))
                for j in range(i + 1, len(lexicons))
            ]
            or [1.0]
        )
        per_speaker = [language_metrics(s.model, world_def, device) for s in speakers]
        for field in LANGUAGE_FIELDS:
            results[field] = mean(m[field] for m in per_speaker)
        return results

    os.makedirs(out_dir, exist_ok=True)
    generation = 0
    final: Dict[str, float] = {}
    topsim_initial: Optional[float] = None
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POP_FIELDS)
        writer.writeheader()

        for step, (scene, targets) in enumerate(islice(dataloader, n_steps), 1):
            s = speakers[pairing_rng.randrange(n_agents)]
            lis = listeners[pairing_rng.randrange(n_listeners)]
            stats = game_step(
                s.model,
                lis.model,
                s.opt,
                lis.opt,
                s.baseline,
                scene.to(device),
                targets.to(device),
                entropy_weight,
            )

            is_eval = step % eval_every == 0 or step == n_steps
            if step % log_every == 0 or is_eval:
                row: Dict[str, float] = {
                    "step": step,
                    "generation": generation,
                    "total_loss": stats["total_loss"],
                    "accuracy": stats["accuracy"],
                }
                if is_eval:
                    final = run_eval()
                    if topsim_initial is None:
                        topsim_initial = final["topsim"]
                    row.update(final)
                writer.writerow(row)
                f.flush()
                logger.info(
                    f"Step {step}/{n_steps} | gen {generation} | "
                    f"Acc: {stats['accuracy']:.3f}"
                    + (f" | {format_metrics(final)}" if is_eval else "")
                )

            if lifespan and step % lifespan == 0 and step < n_steps:
                role, idx = slots[generation % len(slots)]
                fresh_speaker, fresh_listener = new_pair()
                if role == "speaker":
                    speakers[idx] = fresh_speaker
                else:
                    listeners[idx] = fresh_listener
                generation += 1

    if topsim_initial is not None:
        final["topsim_initial"] = topsim_initial
    torch.save(
        {
            "speaker_state_dicts": [s.model.state_dict() for s in speakers],
            "listener_state_dicts": [lis.model.state_dict() for lis in listeners],
            "config": config,
            "k": k,
            "heldout_pairs": heldout_pairs,
            "world": world,
            "metrics": final,
        },
        os.path.join(out_dir, "population.pt"),
    )
    return final
