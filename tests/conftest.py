"""
Pytest configuration and shared fixtures.
"""

import pytest


@pytest.fixture
def sample_data() -> dict:
    """Provide sample data for tests."""
    return {"name": "Test User", "age": 30}


@pytest.fixture
def sample_config() -> dict:
    """Provide sample configuration for tests."""
    return {"name": "test-app", "version": "1.0.0"}
