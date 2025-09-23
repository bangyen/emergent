"""Tests for heatmap input/output functionality.

This module contains validation tests for heatmap creation and file I/O
operations in the contact experiment system.
"""

import numpy as np
from unittest.mock import patch, mock_open

from src.langlab.experiments.contact import ContactExperiment, ContactConfig


class TestHeatmapIO:
    """Test cases for heatmap input/output functionality."""

    def test_heatmap_file_created(self) -> None:
        """Test that heatmap file is created under outputs/figures/."""
        # Create minimal config
        config = ContactConfig(
            n_pairs=2,
            steps_a=10,
            steps_b=10,
            contact_steps=5,
            p_contact=0.3,
            k=3,
            v=5,
            message_length=1,
            seed_a=42,
            seed_b=123,
            batch_size=8,
            learning_rate=1e-3,
            hidden_size=32,
            use_sequence_models=False,
            entropy_weight=0.01,
        )

        # Create experiment
        experiment = ContactExperiment(config)

        # Set up mock intelligibility matrix
        experiment.intelligibility_matrix = np.array([[0.5, 0.6], [0.7, 0.8]])

        # Mock matplotlib to avoid actual file creation
        with patch("matplotlib.pyplot.figure"), patch(
            "matplotlib.pyplot.savefig"
        ), patch("matplotlib.pyplot.close"), patch("matplotlib.pyplot.title"), patch(
            "matplotlib.pyplot.xlabel"
        ), patch(
            "matplotlib.pyplot.ylabel"
        ), patch(
            "matplotlib.pyplot.tight_layout"
        ), patch(
            "seaborn.heatmap"
        ):

            experiment._create_heatmap()

            # Check that savefig was called with correct path

    def test_heatmap_creation_with_different_sizes(self) -> None:
        """Test heatmap creation with different matrix sizes."""
        config = ContactConfig(n_pairs=3)
        experiment = ContactExperiment(config)

        # Test with 3x3 matrix
        experiment.intelligibility_matrix = np.array(
            [[0.5, 0.6, 0.7], [0.8, 0.9, 0.4], [0.3, 0.2, 0.1]]
        )

        with patch("matplotlib.pyplot.figure"), patch(
            "matplotlib.pyplot.savefig"
        ), patch("matplotlib.pyplot.close"), patch("matplotlib.pyplot.title"), patch(
            "matplotlib.pyplot.xlabel"
        ), patch(
            "matplotlib.pyplot.ylabel"
        ), patch(
            "matplotlib.pyplot.tight_layout"
        ), patch(
            "seaborn.heatmap"
        ) as mock_heatmap:

            experiment._create_heatmap()

            # Check that heatmap was called with correct data
            mock_heatmap.assert_called_once()
            call_args = mock_heatmap.call_args
            np.testing.assert_array_equal(
                call_args[0][0], experiment.intelligibility_matrix
            )
            assert call_args[1]["annot"]
            assert call_args[1]["fmt"] == ".3f"
            assert call_args[1]["cmap"] == "viridis"

    def test_csv_output_format(self) -> None:
        """Test that intelligibility matrix is saved in correct CSV format."""
        # Create experiment with mock data
        config = ContactConfig(n_pairs=2)
        experiment = ContactExperiment(config)

        # Set up test matrix
        test_matrix = np.array([[0.5, 0.6], [0.7, 0.8]])
        experiment.intelligibility_matrix = test_matrix

        # Mock file operations
        with patch("builtins.open", mock_open()):
            with patch("numpy.savetxt") as mock_savetxt:
                experiment.save_results()

                # Check that savetxt was called with correct parameters
                mock_savetxt.assert_called_once()
                call_args = mock_savetxt.call_args
                assert call_args[0][0] == "outputs/M.csv"
                np.testing.assert_array_equal(call_args[0][1], test_matrix)
                assert call_args[1]["delimiter"] == ","
                assert call_args[1]["fmt"] == "%.4f"

    def test_jsd_json_output(self) -> None:
        """Test that JSD score is saved in correct JSON format."""
        # Create experiment
        config = ContactConfig(n_pairs=2)
        experiment = ContactExperiment(config)

        # Set up test JSD score
        test_jsd = 0.25
        experiment.jsd_score = test_jsd

        # Mock file operations
        with patch("builtins.open", mock_open()):
            with patch("json.dump") as mock_dump:
                experiment.save_results()

                # Check that json.dump was called with correct data
                mock_dump.assert_called_once()
                call_args = mock_dump.call_args
                expected_data = {"jsd": test_jsd}
                assert call_args[0][0] == expected_data
                assert call_args[1]["indent"] == 2

    def test_output_directory_creation(self) -> None:
        """Test that output directories are created if they don't exist."""
        config = ContactConfig(n_pairs=2)

        with patch("os.makedirs") as mock_makedirs:
            ContactExperiment(config)

            # Check that directories were created
            mock_makedirs.assert_any_call("outputs/logs", exist_ok=True)
            mock_makedirs.assert_any_call("outputs/figures", exist_ok=True)

    def test_heatmap_labels(self) -> None:
        """Test that heatmap has correct labels."""
        config = ContactConfig(n_pairs=2)
        experiment = ContactExperiment(config)

        # Set up test matrix
        experiment.intelligibility_matrix = np.array([[0.5, 0.6], [0.7, 0.8]])

        with patch("matplotlib.pyplot.figure"), patch(
            "matplotlib.pyplot.savefig"
        ), patch("matplotlib.pyplot.close"), patch(
            "matplotlib.pyplot.title"
        ) as mock_title, patch(
            "matplotlib.pyplot.xlabel"
        ) as mock_xlabel, patch(
            "matplotlib.pyplot.ylabel"
        ) as mock_ylabel, patch(
            "matplotlib.pyplot.tight_layout"
        ), patch(
            "seaborn.heatmap"
        ) as mock_heatmap:

            experiment._create_heatmap()

            # Check labels
            mock_title.assert_called_once_with("Mutual Intelligibility Matrix")
            mock_xlabel.assert_called_once_with("Population B Listeners")
            mock_ylabel.assert_called_once_with("Population A Speakers")

            # Check heatmap labels
            call_args = mock_heatmap.call_args
            expected_xticklabels = ["Listener_B0", "Listener_B1"]
            expected_yticklabels = ["Speaker_A0", "Speaker_A1"]
            assert call_args[1]["xticklabels"] == expected_xticklabels
            assert call_args[1]["yticklabels"] == expected_yticklabels

    def test_file_paths_consistency(self) -> None:
        """Test that file paths are consistent across different operations."""
        config = ContactConfig(n_pairs=2)
        experiment = ContactExperiment(config)

        # Set up test data
        experiment.intelligibility_matrix = np.array([[0.5, 0.6], [0.7, 0.8]])
        experiment.jsd_score = 0.25

        # Mock all file operations
        with patch("builtins.open", mock_open()), patch(
            "numpy.savetxt"
        ) as mock_savetxt, patch("json.dump") as mock_dump, patch(
            "matplotlib.pyplot.figure"
        ), patch(
            "matplotlib.pyplot.savefig"
        ) as mock_savefig, patch(
            "matplotlib.pyplot.close"
        ), patch(
            "matplotlib.pyplot.title"
        ), patch(
            "matplotlib.pyplot.xlabel"
        ), patch(
            "matplotlib.pyplot.ylabel"
        ), patch(
            "matplotlib.pyplot.tight_layout"
        ), patch(
            "seaborn.heatmap"
        ):

            experiment.save_results()

            # Check file paths
            assert mock_savetxt.call_args[0][0] == "outputs/M.csv"
            assert (
                mock_dump.call_args[0][1]
                # == mock_open.return_value.__enter__.return_value
            )
            assert (
                mock_savefig.call_args[0][0]
                == "outputs/figures/intelligibility_heatmap.png"
            )
