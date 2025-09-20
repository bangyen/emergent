"""
Enhanced pytest configuration and shared fixtures for Language Emergence Lab.

This module provides comprehensive fixtures and configuration for testing
the Language Emergence Lab package, including mock data, configurations,
and test utilities.
"""

import pytest
import torch
from typing import Dict, List, Tuple, Any, Generator
from unittest.mock import patch

from langlab.core.config import CommunicationConfig
from langlab.data.world import make_object, sample_scene
from langlab.core.agents import Speaker, Listener
from langlab.data.data import ReferentialGameDataset


@pytest.fixture
def sample_config() -> CommunicationConfig:
    """Provide a standard communication configuration for tests."""
    return CommunicationConfig(
        vocabulary_size=10,
        message_length=1,
        hidden_size=32,  # Smaller for faster tests
        multimodal=False,
        distractors=0,
        pragmatic=False,
        seed=42,
    )


@pytest.fixture
def large_config() -> CommunicationConfig:
    """Provide a larger configuration for integration tests."""
    return CommunicationConfig(
        vocabulary_size=20,
        message_length=2,
        hidden_size=64,
        multimodal=True,
        distractors=2,
        pragmatic=True,
        seed=123,
    )


@pytest.fixture
def sample_object() -> Dict[str, Any]:
    """Provide a sample object for testing."""
    return make_object("red", "circle", "small")  # type: ignore


@pytest.fixture
def sample_scene_data() -> Tuple[List[Dict[str, Any]], int]:
    """Provide sample scene data for testing."""
    return sample_scene(k=3, seed=42)  # type: ignore


@pytest.fixture
def sample_speaker(sample_config: CommunicationConfig) -> Speaker:
    """Provide a Speaker agent for testing."""
    return Speaker(sample_config)


@pytest.fixture
def sample_listener(sample_config: CommunicationConfig) -> Listener:
    """Provide a Listener agent for testing."""
    return Listener(sample_config)


@pytest.fixture
def sample_dataset() -> ReferentialGameDataset:
    """Provide a small dataset for testing."""
    return ReferentialGameDataset(n_scenes=10, k=3, seed=42)


@pytest.fixture
def mock_checkpoint(sample_config: CommunicationConfig) -> Dict[str, Any]:
    """Provide a mock checkpoint for testing."""
    return {
        "step": 1000,
        "speaker_state_dict": {
            "encoder.0.weight": torch.randn(32, 8),
            "encoder.0.bias": torch.randn(32),
        },
        "listener_state_dict": {
            "decoder.0.weight": torch.randn(32, 10),
            "decoder.0.bias": torch.randn(10),
        },
        "config": sample_config,
    }


@pytest.fixture
def sample_training_logs() -> List[Dict[str, float]]:
    """Provide sample training logs for testing."""
    return [
        {"step": 0, "accuracy": 0.1, "entropy": 2.3, "message_length": 1.0},
        {"step": 100, "accuracy": 0.3, "entropy": 2.1, "message_length": 1.0},
        {"step": 200, "accuracy": 0.5, "entropy": 1.8, "message_length": 1.0},
        {"step": 300, "accuracy": 0.7, "entropy": 1.5, "message_length": 1.0},
        {"step": 400, "accuracy": 0.8, "entropy": 1.2, "message_length": 1.0},
    ]


@pytest.fixture
def sample_experiment_results() -> Dict[str, Any]:
    """Provide sample experiment results for testing."""
    return {
        "experiment_id": "test_exp_001",
        "params": {
            "vocabulary_size": 10,
            "channel_noise": 0.05,
            "length_cost": 0.01,
            "message_length": 1,
            "hidden_size": 64,
        },
        "metrics": {
            "train": {"acc": 0.85, "entropy": 1.2},
            "iid": {"acc": 0.82, "entropy": 1.3},
            "compo": {"acc": 0.75, "entropy": 1.4},
        },
        "zipf_slope": -0.8,
        "training_time": 120.5,
        "convergence_step": 800,
    }


@pytest.fixture
def mock_device() -> torch.device:
    """Provide a mock device for testing."""
    return torch.device("cpu")


@pytest.fixture
def sample_message_tokens() -> torch.Tensor:
    """Provide sample message tokens for testing."""
    return torch.tensor([[1, 3, 2], [0, 4, 1]])  # Batch of 2 messages


@pytest.fixture
def sample_object_encodings() -> torch.Tensor:
    """Provide sample object encodings for testing."""
    # 2 objects, each with 8 attributes (3 colors + 3 shapes + 2 sizes)
    return torch.tensor(
        [
            [1, 0, 0, 1, 0, 0, 1, 0],  # red circle small
            [0, 1, 0, 0, 1, 0, 0, 1],  # blue square large
        ]
    )


@pytest.fixture
def sample_scene_tensor() -> torch.Tensor:
    """Provide sample scene tensor for testing."""
    # Scene with 3 objects
    return torch.tensor(
        [
            [1, 0, 0, 1, 0, 0, 1, 0],  # red circle small
            [0, 1, 0, 0, 1, 0, 0, 1],  # blue square large
            [0, 0, 1, 0, 0, 1, 1, 0],  # green triangle small
        ]
    )


@pytest.fixture
def sample_population_config() -> Dict[str, Any]:
    """Provide sample population configuration for testing."""
    return {
        "n_pairs": 3,
        "lifespan": 100,
        "replacement_noise": 0.1,
        "crossplay_prob": 0.2,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "hidden_size": 32,
        "vocabulary_size": 10,
        "message_length": 1,
        "use_sequence_models": False,
        "entropy_weight": 0.01,
        "seed": 42,
    }


@pytest.fixture
def sample_contact_config() -> Dict[str, Any]:
    """Provide sample contact experiment configuration for testing."""
    return {
        "n_pairs": 2,
        "steps_a": 200,
        "steps_b": 200,
        "contact_steps": 100,
        "p_contact": 0.3,
        "k": 3,
        "v": 10,
        "message_length": 1,
        "seed_a": 42,
        "seed_b": 123,
        "log_every": 50,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "hidden_size": 32,
        "use_sequence_models": False,
        "entropy_weight": 0.01,
        "heldout_pairs_a": [("red", "circle")],
        "heldout_pairs_b": [("blue", "square")],
    }


@pytest.fixture
def sample_ablation_params() -> Dict[str, Any]:
    """Provide sample ablation study parameters for testing."""
    return {
        "vocab_sizes": [6, 10],
        "channel_noise_levels": [0.0, 0.05],
        "length_costs": [0.0, 0.01],
        "base_seed": 42,
        "n_steps": 100,
        "k": 3,
        "message_length": 1,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "hidden_size": 32,
        "entropy_weight": 0.01,
    }


@pytest.fixture
def mock_streamlit() -> Generator[Any, None, None]:
    """Mock Streamlit for testing dashboard components."""
    with patch("streamlit.set_page_config"), patch("streamlit.title"), patch(
        "streamlit.markdown"
    ), patch("streamlit.sidebar"), patch("streamlit.selectbox"), patch(
        "streamlit.warning"
    ), patch(
        "streamlit.success"
    ), patch(
        "streamlit.error"
    ), patch(
        "streamlit.metric"
    ), patch(
        "streamlit.plotly_chart"
    ), patch(
        "streamlit.pyplot"
    ) as mock_plt:
        yield mock_plt


@pytest.fixture
def mock_torch_load() -> Generator[Any, None, None]:
    """Mock torch.load for testing checkpoint loading."""
    with patch("torch.load") as mock_load:
        mock_load.return_value = {
            "step": 1000,
            "speaker_state_dict": {},
            "listener_state_dict": {},
            "config": {},
        }
        yield mock_load


@pytest.fixture
def mock_subprocess() -> Generator[Any, None, None]:
    """Mock subprocess for testing CLI commands."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        yield mock_run


@pytest.fixture
def temp_output_dir(tmp_path: Any) -> Any:
    """Provide a temporary output directory for testing."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_zipf_data() -> Dict[str, Any]:
    """Provide sample Zipf analysis data for testing."""
    return {
        "ranks": [1, 2, 3, 4, 5],
        "frequencies": [100, 50, 33, 25, 20],
        "tokens": ["token1", "token2", "token3", "token4", "token5"],
    }


@pytest.fixture
def sample_compositional_data() -> Dict[str, float]:
    """Provide sample compositional analysis data for testing."""
    return {
        "train_accuracy": 0.85,
        "iid_accuracy": 0.82,
        "compo_accuracy": 0.75,
        "generalization_gap": 0.07,
    }


# Test markers for organizing tests
def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "smoke: mark test as a smoke test")
