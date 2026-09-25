"""Upper bound on held-out-target accuracy with a perfectly compositional speaker.

The speaker is replaced by an oracle that sends one token per attribute (the
attribute's value index), so every object -- including held-out ones -- gets a
perfectly compositional message. Only the listener is trained, on the same
held-out-free scene stream as ``langlab train``. If the listener then solves
held-out targets, the generalization gap in the README comes from the learned
speaker's language rather than from the listener.

Usage: python experiments/oracle_listener.py [--listener mlp|dot] [--steps 10000]
"""

import argparse
from itertools import islice
from statistics import mean, stdev

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from langlab.analysis.eval import accuracy
from langlab.core.agents import DotListener, Listener, SpeakerOutput
from langlab.core.config import CommunicationConfig
from langlab.data.data import SceneStream
from langlab.data.world import DEFAULT_WORLD
from langlab.training.train import build_eval_sets

HELDOUT = [("red", "circle")]


class OracleSpeaker(nn.Module):
    """Message position i carries the value index of attribute i."""

    def forward(self, objects: torch.Tensor) -> SpeakerOutput:
        tokens = []
        offset = 0
        for values in DEFAULT_WORLD.attributes.values():
            block = objects[:, offset : offset + len(values)]
            tokens.append(block.argmax(dim=-1))
            offset += len(values)
        tokens_t = torch.stack(tokens, dim=1)
        return SpeakerOutput(logits=F.one_hot(tokens_t, 3).float(), tokens=tokens_t)


def run(seed: int, steps: int, listener_type: str) -> dict:
    torch.manual_seed(seed)
    config = CommunicationConfig(vocabulary_size=3, message_length=3)
    listener_cls = DotListener if listener_type == "dot" else Listener
    speaker, listener = OracleSpeaker(), listener_cls(config)
    opt = torch.optim.Adam(listener.parameters(), lr=2e-4)
    stream = SceneStream(5, seed=seed, heldout_pairs=HELDOUT)
    for scene, targets in islice(DataLoader(stream, batch_size=32), steps):
        tokens = speaker(scene[torch.arange(scene.size(0)), targets]).tokens
        probs = listener(tokens, scene).probs
        loss = F.nll_loss(torch.log(probs + 1e-8), targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    eval_sets = build_eval_sets(5, seed, 1000, HELDOUT, DEFAULT_WORLD)
    device = torch.device("cpu")
    return {
        name: accuracy(speaker, listener, ds, device) for name, ds in eval_sets.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--listener", choices=["mlp", "dot"], default="mlp")
    args = parser.parse_args()

    results = [run(seed, args.steps, args.listener) for seed in args.seeds]
    for name in results[0]:
        values = [r[name] for r in results]
        spread = stdev(values) if len(values) > 1 else 0.0
        print(f"{name}_acc: {100 * mean(values):.1f} ± {100 * spread:.1f}%")


if __name__ == "__main__":
    main()
