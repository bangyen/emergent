"""Shared pytest fixtures for Language Emergence Lab tests."""

from typing import Any

import pytest

from langlab.core.config import CommunicationConfig


@pytest.fixture
def sample_config() -> CommunicationConfig:
    """Provide a small communication configuration for fast tests."""
    return CommunicationConfig(
        vocabulary_size=10,
        message_length=1,
        hidden_size=32,
        seed=42,
    )


def pytest_configure(config: Any) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
