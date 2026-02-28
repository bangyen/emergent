from click.testing import CliRunner
from langlab.apps.cli import main
import torch
import os
import tempfile
from langlab.core.config import CommunicationConfig


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0


def test_cli_sample() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["sample", "--k", "3", "--seed", "42"])
    assert result.exit_code == 0
    assert "Target object index:" in result.output


def test_cli_train_minimal() -> None:
    runner = CliRunner()
    # Run a very short training session
    result = runner.invoke(main, ["train", "--steps", "2", "--batch-size", "2"])
    assert result.exit_code == 0
    assert "Training completed successfully!" in result.output


def test_cli_eval_fail_missing_ckpt() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["eval"])
    assert result.exit_code != 0


def test_cli_eval_success() -> None:
    # Need a temporary checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        config = CommunicationConfig(
            vocabulary_size=10, message_length=2, hidden_size=32
        )
        # We need a proper state dict for Speaker and Listener since eval init them
        # This is complex to mock fully in a CLI test without creating a real model,
        # but we can try to save a minimal one if we have access to agents.
        from langlab.core.agents import Speaker, Listener

        speaker = Speaker(config)
        listener = Listener(config)
        torch.save(
            {
                "config": config,
                "speaker_state_dict": speaker.state_dict(),
                "listener_state_dict": listener.state_dict(),
            },
            tmp.name,
        )
        tmp_path = tmp.name

    runner = CliRunner()
    try:
        # Using train split to avoid needing heldout_pairs
        result = runner.invoke(main, ["eval", "--ckpt", tmp_path, "--split", "train"])
        assert result.exit_code == 0
        assert "Evaluation Results:" in result.output
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
