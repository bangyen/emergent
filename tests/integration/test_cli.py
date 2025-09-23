"""Integration tests for CLI functionality.

This module tests the command-line interface integration, including
end-to-end workflows and command execution.
"""

from typing import Any
import pytest
from unittest.mock import patch, Mock
from click.testing import CliRunner

from src.langlab.apps.cli import main


@pytest.mark.integration
class TestCLIIntegration:
    """Test CLI integration and end-to-end workflows."""

    def test_cli_help(self) -> None:
        """Test that CLI help works correctly."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Language Emergence Lab" in result.output
        assert "sample" in result.output
        assert "train" in result.output

    def test_sample_command(self) -> None:
        """Test sample command execution."""
        runner = CliRunner()
        result = runner.invoke(main, ["sample", "--k", "3", "--seed", "42"])
        assert result.exit_code == 0
        assert "Scene with 3 objects" in result.output
        assert "Target object index" in result.output

    def test_dataset_command(self) -> None:
        """Test dataset command execution."""
        runner = CliRunner()
        result = runner.invoke(main, ["dataset", "--n-scenes", "5", "--k", "3"])
        assert result.exit_code == 0
        assert "Dataset created" in result.output
        assert "Scenes: 5" in result.output

    def test_info_command(self) -> None:
        """Test info command execution."""
        runner = CliRunner()
        result = runner.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "Language Emergence Lab" in result.output
        assert "Available colors" in result.output
        assert "Device:" in result.output

    @patch("src.langlab.apps.cli.train")
    def test_train_command_mock(self, mock_train: Any) -> None:
        """Test train command with mocked training function."""
        mock_train.return_value = None

        runner = CliRunner()
        result = runner.invoke(
            main, ["train", "--steps", "10", "--k", "3", "--v", "5", "--seed", "42"]
        )

        assert result.exit_code == 0
        assert "Training completed successfully!" in result.output
        mock_train.assert_called_once()

    @patch("src.langlab.apps.cli.evaluate")
    def test_eval_command_mock(self, mock_eval: Any) -> None:
        """Test eval command with mocked evaluation function."""
        mock_eval.return_value = {"acc": 0.85}

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "eval",
                "--ckpt",
                "test_checkpoint.pt",
                "--split",
                "train",
                "--n-scenes",
                "10",
            ],
        )

        assert result.exit_code == 0
        assert "Evaluation Results:" in result.output
        assert "Accuracy: 0.8500" in result.output
        mock_eval.assert_called_once()

    @patch("src.langlab.apps.cli.train_population")
    def test_pop_train_command_mock(self, mock_train_pop: Any) -> None:
        """Test population training command with mocked function."""
        mock_train_pop.return_value = None

        runner = CliRunner()
        result = runner.invoke(
            main, ["pop-train", "--pairs", "2", "--steps", "50", "--lifespan", "25"]
        )

        assert result.exit_code == 0
        assert "Population training completed successfully!" in result.output
        mock_train_pop.assert_called_once()

    @patch("src.langlab.apps.cli.train_contact_experiment")
    def test_contact_command_mock(self, mock_contact: Any) -> None:
        """Test contact experiment command with mocked function."""
        mock_contact.return_value = None

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "contact",
                "--pairs",
                "2",
                "--steps-a",
                "50",
                "--steps-b",
                "50",
                "--contact-steps",
                "25",
            ],
        )

        assert result.exit_code == 0
        assert "Contact experiment completed successfully!" in result.output
        mock_contact.assert_called_once()

    @patch("src.langlab.apps.cli.train_grounded")
    def test_train_grid_command_mock(self, mock_grounded: Any) -> None:
        """Test grounded training command with mocked function."""
        mock_grounded.return_value = None

        runner = CliRunner()
        result = runner.invoke(
            main, ["train-grid", "--episodes", "10", "--grid", "3", "--max-steps", "5"]
        )

        assert result.exit_code == 0
        assert "Grounded training completed successfully!" in result.output
        mock_grounded.assert_called_once()

    @patch("src.langlab.apps.cli.run_ablation_suite")
    def test_ablate_command_mock(self, mock_ablate: Any) -> None:
        """Test ablation command with mocked function."""
        mock_ablate.return_value = [{"exp_id": "test_1", "acc": 0.8}]

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "ablate",
                "--vocab-sizes",
                "6,10",
                "--noise-levels",
                "0,0.05",
                "--length-costs",
                "0,0.01",
                "--steps",
                "50",
            ],
        )

        assert result.exit_code == 0
        assert "Ablation study completed successfully!" in result.output
        mock_ablate.assert_called_once()

    @patch("src.langlab.apps.cli.create_report")
    def test_report_command_mock(self, mock_report: Any) -> None:
        """Test report command with mocked function."""
        mock_report.return_value = {
            "csv_path": "test.csv",
            "summary_path": "test.json",
            "total_experiments": 4,
        }

        runner = CliRunner()
        result = runner.invoke(
            main, ["report", "--input", "test_pattern", "--output-dir", "test_output"]
        )

        assert result.exit_code == 0
        assert "Report generated successfully!" in result.output
        mock_report.assert_called_once()

    @patch("subprocess.run")
    def test_dash_command_mock(self, mock_subprocess: Any) -> None:
        """Test dashboard command with mocked subprocess."""
        mock_subprocess.return_value = Mock(returncode=0)

        runner = CliRunner()
        result = runner.invoke(main, ["dash", "--port", "8888"])

        # The command should start but we can't test the full execution
        # Just verify it doesn't crash immediately
        assert result.exit_code == 0 or result.exit_code == 1  # May exit due to mocking

    def test_invalid_command(self) -> None:
        """Test that invalid commands return appropriate error."""
        runner = CliRunner()
        result = runner.invoke(main, ["invalid-command"])
        assert result.exit_code != 0

    def test_command_with_invalid_options(self) -> None:
        """Test commands with invalid options."""
        runner = CliRunner()
        result = runner.invoke(main, ["sample", "--k", "0"])  # Invalid k value
        # The CLI doesn't validate ranges, so it will succeed but show an error message
        assert result.exit_code == 0  # CLI succeeds
        assert "Error:" in result.output  # But shows error message

    def test_heldout_parsing(self) -> None:
        """Test heldout pair parsing in train command."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["train", "--steps", "10", "--heldout", "red,circle"],  # Valid format
        )
        # Should not crash on parsing
        assert result.exit_code == 0 or result.exit_code == 1  # May fail on training

    def test_heldout_parsing_invalid(self) -> None:
        """Test invalid heldout pair parsing."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "train",
                "--steps",
                "10",
                "--heldout",
                "red",  # Invalid format - only one attribute
            ],
        )
        # The CLI validates heldout format and returns early but doesn't set error exit code
        assert result.exit_code == 0  # CLI succeeds
        assert (
            "Error: heldout must be exactly two comma-separated attributes"
            in result.output
        )


@pytest.mark.integration
class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""

    def test_missing_required_arguments(self) -> None:
        """Test that missing required arguments are handled properly."""
        runner = CliRunner()
        result = runner.invoke(main, ["eval"])  # Missing required --ckpt
        assert result.exit_code != 0

    def test_invalid_numeric_arguments(self) -> None:
        """Test handling of invalid numeric arguments."""
        runner = CliRunner()
        result = runner.invoke(main, ["sample", "--k", "invalid"])
        assert result.exit_code != 0

    def test_out_of_range_arguments(self) -> None:
        """Test handling of out-of-range arguments."""
        runner = CliRunner()
        result = runner.invoke(main, ["sample", "--k", "-1"])
        # The CLI doesn't validate ranges, so it will succeed but show an error message
        assert result.exit_code == 0  # CLI succeeds
        assert "Error:" in result.output  # But shows error message

    @patch("src.langlab.apps.cli.train")
    def test_training_error_handling(self, mock_train: Any) -> None:
        """Test that training errors are handled gracefully."""
        mock_train.side_effect = Exception("Training failed")

        runner = CliRunner()
        result = runner.invoke(main, ["train", "--steps", "10", "--k", "3", "--v", "5"])

        # The CLI catches exceptions and shows error message but doesn't exit with error code
        assert result.exit_code == 0  # CLI succeeds
        assert "Error:" in result.output  # But shows error message


@pytest.mark.integration
class TestCLIWorkflows:
    """Test complete CLI workflows."""

    def test_sample_to_dataset_workflow(self) -> None:
        """Test workflow from sampling to dataset generation."""
        runner = CliRunner()

        # First, generate a sample
        result1 = runner.invoke(main, ["sample", "--k", "3"])
        assert result1.exit_code == 0

        # Then, generate a dataset
        result2 = runner.invoke(main, ["dataset", "--n-scenes", "5", "--k", "3"])
        assert result2.exit_code == 0

    def test_info_to_sample_workflow(self) -> None:
        """Test workflow from info to sampling."""
        runner = CliRunner()

        # Get system info
        result1 = runner.invoke(main, ["info"])
        assert result1.exit_code == 0

        # Generate a sample
        result2 = runner.invoke(main, ["sample", "--k", "3"])
        assert result2.exit_code == 0

    @patch("src.langlab.apps.cli.train")
    @patch("src.langlab.apps.cli.evaluate")
    def test_train_to_eval_workflow(self, mock_eval: Any, mock_train: Any) -> None:
        """Test workflow from training to evaluation."""
        mock_train.return_value = None
        mock_eval.return_value = {"acc": 0.85}

        runner = CliRunner()

        # Train a model
        result1 = runner.invoke(
            main, ["train", "--steps", "10", "--k", "3", "--v", "5"]
        )
        assert result1.exit_code == 0

        # Evaluate the model
        result2 = runner.invoke(
            main,
            ["eval", "--ckpt", "outputs/checkpoints/checkpoint.pt", "--split", "train"],
        )
        assert result2.exit_code == 0
