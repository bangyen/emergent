"""Structural metrics for emergent languages.

All metrics are computed on the speaker's greedy lexicon: the message it sends
for every object in the world.

- ``topsim``: topographic similarity (Brighton & Kirby, 2006) - Spearman
  correlation between pairwise attribute Hamming distances and pairwise message
  Hamming distances. 1.0 means similar meanings get similar messages.
- ``posdis``: positional disentanglement (Chaabouni et al., 2020) - how much
  each message position specialises in a single attribute.
- ``msg_entropy``: entropy (bits) of the message distribution over objects;
  ``log2(n_objects)`` means every object gets its own message.
- ``n_messages``: number of distinct messages in the lexicon.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import torch

from ..data.world import World

Message = Tuple[int, ...]


def lexicon(speaker: Any, world: World, device: torch.device) -> List[Message]:
    """Greedy message for every object in ``world`` (in ``world.objects`` order)."""
    was_training = speaker.training
    speaker.eval()
    with torch.no_grad():
        encodings = torch.stack([world.encode(o) for o in world.objects]).to(device)
        tokens = speaker(encodings).tokens.cpu()
    speaker.train(was_training)
    return [tuple(int(t) for t in row) for row in tokens]


def _rank(values: Sequence[float]) -> List[float]:
    """Ranks with ties given their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for m in range(i, j + 1):
            ranks[order[m]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation; 0.0 if either input is constant."""
    rx, ry = _rank(x), _rank(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def _hamming(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(x != y for x, y in zip(a, b))


def topographic_similarity(
    meanings: Sequence[Sequence[int]], messages: Sequence[Message]
) -> float:
    pairs = [(i, j) for i in range(len(meanings)) for j in range(i + 1, len(meanings))]
    meaning_d = [float(_hamming(meanings[i], meanings[j])) for i, j in pairs]
    message_d = [float(_hamming(messages[i], messages[j])) for i, j in pairs]
    return spearman(meaning_d, message_d)


def _entropy(items: Sequence[Any]) -> float:
    n = len(items)
    return -sum(c / n * math.log2(c / n) for c in Counter(items).values())


def _mutual_information(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    return _entropy(xs) + _entropy(ys) - _entropy(list(zip(xs, ys)))


def positional_disentanglement(
    meanings: Sequence[Sequence[int]], messages: Sequence[Message]
) -> float:
    """Mean over informative positions of (I_top1 - I_top2) / H(position)."""
    n_attrs = len(meanings[0])
    scores = []
    for pos in range(len(messages[0])):
        symbols = [m[pos] for m in messages]
        h = _entropy(symbols)
        if h == 0:
            continue
        mis = sorted(
            (
                _mutual_information(symbols, [m[a] for m in meanings])
                for a in range(n_attrs)
            ),
            reverse=True,
        )
        second = mis[1] if len(mis) > 1 else 0.0
        scores.append((mis[0] - second) / h)
    return sum(scores) / len(scores) if scores else 0.0


def language_metrics(
    speaker: Any, world: World, device: torch.device
) -> Dict[str, float]:
    """Compute all lexicon metrics for ``speaker`` over ``world``."""
    messages = lexicon(speaker, world, device)
    meanings = [world.attribute_indices(o) for o in world.objects]
    return {
        "topsim": topographic_similarity(meanings, messages),
        "posdis": positional_disentanglement(meanings, messages),
        "msg_entropy": _entropy(messages),
        "n_messages": float(len(set(messages))),
    }
