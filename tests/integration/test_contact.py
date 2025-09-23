"""Tests for contact experiments and intelligibility measurement.

This module contains validation tests for the contact experiment functionality,
including mutual intelligibility matrix calculation and Jensen-Shannon divergence.
"""

import numpy as np
import torch

# import pytest
from unittest.mock import Mock, patch

from src.langlab.experiments.contact import (
    ContactExperiment,
    ContactConfig,
    train_contact_experiment,
)


class TestContactExperiments:
    """Test cases for contact experiment functionality."""

    def test_mutual_intelligibility_matrix_shape(self) -> None:
        """Test that mutual intelligibility matrix has correct shape [pairs_A, pairs_B]."""
        # Create minimal config
        config = ContactConfig(
            n_pairs=3,
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

        # Mock populations with correct number of pairs
        experiment.population_a = Mock()
        experiment.population_b = Mock()
        experiment.population_a.pairs = [Mock() for _ in range(3)]
        experiment.population_b.pairs = [Mock() for _ in range(3)]

        # Mock the intelligibility measurement
        with patch.object(
            experiment, "_measure_pair_intelligibility", return_value=0.5
        ), patch.object(
            experiment, "_collect_message_distribution", return_value=[]
        ), patch.object(
            experiment,
            "_messages_to_distribution",
            return_value=np.array([0.1, 0.2, 0.3, 0.2, 0.2]),
        ):
            experiment.measure_intelligibility()

        # Check matrix shape
        assert experiment.intelligibility_matrix is not None
        assert experiment.intelligibility_matrix.shape == (3, 3)
        assert experiment.intelligibility_matrix.dtype == np.float64

    def test_jsd_in_range(self) -> None:
        """Test that Jensen-Shannon divergence is in range [0, 1]."""
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

        # Mock populations
        experiment.population_a = Mock()
        experiment.population_b = Mock()
        experiment.population_a.pairs = [Mock() for _ in range(2)]
        experiment.population_b.pairs = [Mock() for _ in range(2)]

        # Mock intelligibility measurement
        experiment.intelligibility_matrix = np.array([[0.5, 0.6], [0.7, 0.8]])

        # Mock message collection and JSD computation
        with patch.object(experiment, "_collect_message_distribution") as mock_collect:
            mock_collect.return_value = [torch.tensor([[1, 2, 3]])]

            with patch.object(experiment, "_messages_to_distribution") as mock_dist:
                mock_dist.side_effect = [
                    np.array([0.1, 0.2, 0.3, 0.2, 0.2]),  # Population A
                    np.array([0.2, 0.1, 0.2, 0.3, 0.2]),  # Population B
                ]

                jsd = experiment._compute_jsd()

        # Check JSD is in valid range
        assert 0.0 <= jsd <= 1.0
        assert isinstance(jsd, float)

    def test_jensen_shannon_divergence_properties(self) -> None:
        """Test JSD mathematical properties."""
        experiment = ContactExperiment(ContactConfig())

        # Test identical distributions (JSD = 0)
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.2, 0.3, 0.5])
        jsd_identical = experiment._jensen_shannon_divergence(p, q)
        assert abs(jsd_identical) < 1e-9  # Should be approximately 0

        # Test maximally different distributions
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        jsd_different = experiment._jensen_shannon_divergence(p, q)
        assert jsd_different > 0.5  # Should be high

        # Test symmetry
        jsd_pq = experiment._jensen_shannon_divergence(p, q)
        jsd_qp = experiment._jensen_shannon_divergence(q, p)
        assert abs(jsd_pq - jsd_qp) < 1e-10  # Should be symmetric

    def test_messages_to_distribution(self) -> None:
        """Test conversion of messages to probability distribution."""
        experiment = ContactExperiment(ContactConfig(v=5))

        # Create mock messages
        messages = [
            torch.tensor([[0, 1, 2]]),  # Batch 1
            torch.tensor([[1, 2, 3]]),  # Batch 2
        ]

        # Convert to distribution
        distribution = experiment._messages_to_distribution(messages, vocab_size=5)

        # Check properties
        assert len(distribution) == 5
        assert abs(np.sum(distribution) - 1.0) < 1e-10  # Should sum to 1
        assert np.all(distribution >= 0)  # All probabilities >= 0

        # Check specific values
        expected_counts = np.array([1, 2, 2, 1, 0])  # Counts for tokens 0,1,2,3,4
        expected_dist = expected_counts / np.sum(expected_counts)
        np.testing.assert_array_almost_equal(distribution, expected_dist)

    def test_contact_config_validation(self) -> None:
        """Test ContactConfig parameter validation."""
        # Valid config should work
        config = ContactConfig(
            n_pairs=2,
            steps_a=100,
            steps_b=100,
            contact_steps=50,
            p_contact=0.5,
            k=3,
            v=5,
            message_length=1,
            seed_a=42,
            seed_b=123,
        )
        assert config.n_pairs == 2
        assert config.p_contact == 0.5

        # Test invalid p_contact - these should not raise ValueError since ContactConfig doesn't validate
        # The validation happens in PopulationConfig, not ContactConfig
        config_invalid = ContactConfig(p_contact=-0.1)
        assert config_invalid.p_contact == -0.1

        config_invalid2 = ContactConfig(p_contact=1.1)
        assert config_invalid2.p_contact == 1.1


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
        ) as mock_savefig, patch("matplotlib.pyplot.close"), patch(
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

            experiment._create_heatmap()

            # Check that savefig was called with correct path
            mock_savefig.assert_called_once()
            call_args = mock_savefig.call_args
            assert call_args[0][0] == "outputs/figures/intelligibility_heatmap.png"

    def test_csv_output_format(self) -> None:
        """Test that intelligibility matrix is saved in correct CSV format."""
        # Create experiment with mock data
        config = ContactConfig(n_pairs=2)
        experiment = ContactExperiment(config)

        # Set up test matrix
        test_matrix = np.array([[0.5, 0.6], [0.7, 0.8]])
        experiment.intelligibility_matrix = test_matrix

        # Mock file operations
        with patch("builtins.open", create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

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
        with patch("builtins.open", create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            with patch("json.dump") as mock_dump:
                experiment.save_results()

                # Check that json.dump was called with correct data
                mock_dump.assert_called_once()
                call_args = mock_dump.call_args
                expected_data = {"jsd": test_jsd}
                assert call_args[0][0] == expected_data
                assert call_args[0][1] == mock_file
                assert call_args[1]["indent"] == 2


class TestContactIntegration:
    """Integration tests for contact experiment workflow."""

    @patch("src.langlab.experiments.contact.ContactExperiment")
    def test_train_contact_experiment_calls(self, mock_experiment_class: Mock) -> None:
        """Test that train_contact_experiment calls all required methods."""
        # Mock the experiment instance
        mock_experiment = Mock()
        mock_experiment_class.return_value = mock_experiment

        # Call the function
        train_contact_experiment(
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
        )

        # Check that all required methods were called
        mock_experiment.train_stage_a.assert_called_once()
        mock_experiment.train_stage_b.assert_called_once()
        mock_experiment.measure_intelligibility.assert_called_once()
        mock_experiment.save_results.assert_called_once()

    def test_contact_config_defaults(self) -> None:
        """Test that ContactConfig has sensible defaults."""
        config = ContactConfig()

        # Check default values
        assert config.n_pairs == 4
        assert config.steps_a == 4000
        assert config.steps_b == 4000
        assert config.contact_steps == 2000
        assert config.p_contact == 0.3
        assert config.k == 5
        assert config.v == 10
        assert config.message_length == 1
        assert config.seed_a == 42
        assert config.seed_b == 123
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3
        assert config.hidden_size == 64
        assert not config.use_sequence_models
        assert config.entropy_weight == 0.01
        assert config.heldout_pairs_a is None
        assert config.heldout_pairs_b is None
