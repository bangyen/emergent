"""Basic tests for the template."""


def test_basic_functionality() -> None:
    """Test basic functionality."""
    assert True


def test_sample_data_fixture(sample_data: dict) -> None:
    """Test that sample_data fixture works."""
    assert "name" in sample_data
    assert sample_data["name"] == "Test User"


def test_sample_config_fixture(sample_config: dict) -> None:
    """Test that sample_config fixture works."""
    assert "name" in sample_config
    assert sample_config["name"] == "test-app"
