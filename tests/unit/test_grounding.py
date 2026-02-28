import pytest
import torch
from langlab.training.grounding import (
    GroundedEnvironment,
    GroundedSpeaker,
    GroundedListener,
)
from langlab.core.config import CommunicationConfig


@pytest.fixture
def config() -> CommunicationConfig:
    return CommunicationConfig(vocabulary_size=10, message_length=2, hidden_size=32)


def test_grounded_speaker(config: CommunicationConfig) -> None:
    speaker = GroundedSpeaker(config)
    obj_attr = torch.randn(2, 8)
    target_coord = torch.randn(2, 2)

    logits, tokens = speaker(obj_attr, target_coord)

    assert logits.shape == (2, 2, 10)
    assert tokens.shape == (2, 2)


def test_grounded_listener(config: CommunicationConfig) -> None:
    listener = GroundedListener(config, grid_size=5)
    tokens = torch.randint(0, 10, (2, 2))
    grid_obs = torch.randn(2, 5, 5)

    logits = listener(tokens, grid_obs)
    assert logits.shape == (2, 5)


def test_grounded_environment_episode(config: CommunicationConfig) -> None:
    env = GroundedEnvironment(config, grid_size=5, max_steps=5)
    episode = env.create_episode()

    assert hasattr(episode, "success")
    assert isinstance(episode.success, bool)
    assert len(episode.actions) <= 5
