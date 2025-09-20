"""Basic tests for the template."""

from typing import Any


def test_basic_functionality() -> None:
    """Test basic functionality."""
    assert True


def test_sample_config_fixture(sample_config: Any) -> None:
    """Test that sample_config fixture works."""
    assert hasattr(sample_config, "vocabulary_size")
    assert hasattr(sample_config, "message_length")
    assert hasattr(sample_config, "hidden_size")
    assert sample_config.vocabulary_size == 10
    assert sample_config.message_length == 1
    assert sample_config.hidden_size == 32
