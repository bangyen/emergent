"""Tests for emergent-language structure metrics."""

import pytest
import torch

from langlab.analysis.language import (
    language_metrics,
    positional_disentanglement,
    spearman,
    topographic_similarity,
)
from langlab.core.agents import Speaker
from langlab.core.config import CommunicationConfig
from langlab.data.world import DEFAULT_WORLD

MEANINGS = [DEFAULT_WORLD.attribute_indices(o) for o in DEFAULT_WORLD.objects]


def test_spearman() -> None:
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0


def test_perfectly_compositional_language() -> None:
    # One position per attribute, symbol = value index.
    messages = [tuple(m) for m in MEANINGS]
    assert topographic_similarity(MEANINGS, messages) == pytest.approx(1.0)
    assert positional_disentanglement(MEANINGS, messages) == pytest.approx(1.0)


def test_holistic_language_scores_low() -> None:
    # Unique arbitrary symbol per object at position 0, constant position 1.
    messages = [((i * 7) % 18, 0) for i in range(18)]
    assert topographic_similarity(MEANINGS, messages) < 0.3
    assert positional_disentanglement(MEANINGS, messages) < 0.3


def test_language_metrics_on_speaker() -> None:
    torch.manual_seed(0)
    speaker = Speaker(CommunicationConfig(vocabulary_size=8, message_length=3))
    speaker.train()
    metrics = language_metrics(speaker, DEFAULT_WORLD, torch.device("cpu"))
    assert set(metrics) == {"topsim", "posdis", "msg_entropy", "n_messages"}
    assert 1 <= metrics["n_messages"] <= 18
    assert speaker.training  # mode restored
