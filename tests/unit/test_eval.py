import pytest
import torch
import os
import tempfile
from langlab.analysis.eval import evaluate
from langlab.core.config import CommunicationConfig


def test_evaluate_train_split() -> None:
    # Create a dummy model checkpoint
    config = CommunicationConfig(vocabulary_size=10, message_length=2, hidden_size=32)
    speaker_state = {
        "encoder.0.weight": torch.randn(32, 8),
        "encoder.0.bias": torch.zeros(32),
        "encoder.1.weight": torch.ones(32),
        "encoder.1.bias": torch.zeros(32),
        "encoder.4.weight": torch.randn(32, 32),
        "encoder.4.bias": torch.zeros(32),
        "encoder.5.weight": torch.ones(32),
        "encoder.5.bias": torch.zeros(32),
        "encoder.8.weight": torch.randn(32, 32),
        "encoder.8.bias": torch.zeros(32),
        "encoder.9.weight": torch.ones(32),
        "encoder.9.bias": torch.zeros(32),
        "residual_projection.weight": torch.randn(32, 8),
        "residual_projection.bias": torch.zeros(32),
        "output_layers.0.0.weight": torch.randn(32, 32),
        "output_layers.0.0.bias": torch.zeros(32),
        "output_layers.0.1.weight": torch.ones(32),
        "output_layers.0.1.bias": torch.zeros(32),
        "output_layers.0.4.weight": torch.randn(16, 32),
        "output_layers.0.4.bias": torch.zeros(16),
        "output_layers.0.5.weight": torch.ones(16),
        "output_layers.0.5.bias": torch.zeros(16),
        "output_layers.0.8.weight": torch.randn(10, 16),
        "output_layers.0.8.bias": torch.zeros(10),
        "output_layers.1.0.weight": torch.randn(32, 32),
        "output_layers.1.0.bias": torch.zeros(32),
        "output_layers.1.1.weight": torch.ones(32),
        "output_layers.1.1.bias": torch.zeros(32),
        "output_layers.1.4.weight": torch.randn(16, 32),
        "output_layers.1.4.bias": torch.zeros(16),
        "output_layers.1.5.weight": torch.ones(16),
        "output_layers.1.5.bias": torch.zeros(16),
        "output_layers.1.8.weight": torch.randn(10, 16),
        "output_layers.1.8.bias": torch.zeros(10),
    }

    # Minimal listener state
    listener_state = {
        "message_encoder.0.weight": torch.randn(32, 20),
        "message_encoder.0.bias": torch.zeros(32),
        "message_encoder.1.weight": torch.ones(32),
        "message_encoder.1.bias": torch.zeros(32),
        "message_encoder.4.weight": torch.randn(32, 32),
        "message_encoder.4.bias": torch.zeros(32),
        "message_encoder.5.weight": torch.ones(32),
        "message_encoder.5.bias": torch.zeros(32),
        "message_residual_proj.weight": torch.randn(32, 20),
        "message_residual_proj.bias": torch.zeros(32),
        "object_residual_proj.weight": torch.randn(32, 8),
        "object_residual_proj.bias": torch.zeros(32),
        "object_encoder.0.weight": torch.randn(32, 8),
        "object_encoder.0.bias": torch.zeros(32),
        "object_encoder.1.weight": torch.ones(32),
        "object_encoder.1.bias": torch.zeros(32),
        "object_encoder.4.weight": torch.randn(32, 32),
        "object_encoder.4.bias": torch.zeros(32),
        "object_encoder.5.weight": torch.ones(32),
        "object_encoder.5.bias": torch.zeros(32),
        "scorer.0.weight": torch.randn(32, 64),
        "scorer.0.bias": torch.zeros(32),
        "scorer.1.weight": torch.ones(32),
        "scorer.1.bias": torch.zeros(32),
        "scorer.4.weight": torch.randn(32, 32),
        "scorer.4.bias": torch.zeros(32),
        "scorer.5.weight": torch.ones(32),
        "scorer.5.bias": torch.zeros(32),
        "scorer.8.weight": torch.randn(16, 32),
        "scorer.8.bias": torch.zeros(16),
        "scorer.9.weight": torch.ones(16),
        "scorer.9.bias": torch.zeros(16),
        "scorer.12.weight": torch.randn(1, 16),
        "scorer.12.bias": torch.zeros(1),
    }

    checkpoint = {
        "config": config,
        "speaker_state_dict": speaker_state,
        "listener_state_dict": listener_state,
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save(checkpoint, tmp.name)
        tmp_path = tmp.name

    try:
        results = evaluate(tmp_path, split="train", n_scenes=10, k=5, batch_size=2)
        assert "acc" in results
        assert 0.0 <= results["acc"] <= 1.0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_evaluate_invalid_split() -> None:
    # Create a valid checkpoint structure to avoid loading errors before split check
    config = CommunicationConfig(vocabulary_size=10, message_length=2, hidden_size=32)
    from langlab.core.agents import Speaker, Listener

    speaker = Speaker(config)
    listener = Listener(config)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save(
            {
                "config": config,
                "speaker_state_dict": speaker.state_dict(),
                "listener_state_dict": listener.state_dict(),
            },
            tmp.name,
        )
        tmp_path = tmp.name

    try:
        with pytest.raises(ValueError, match="Unsupported split"):
            evaluate(tmp_path, split="invalid")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
