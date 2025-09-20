"""Integration tests for training workflows.

This module tests complete training workflows, including agent training,
population dynamics, and contact experiments.
"""

import pytest
from unittest.mock import patch

from langlab.agents import Speaker, Listener, SpeakerSeq, ListenerSeq
from langlab.data import ReferentialGameDataset
from langlab.train import train, MovingAverage
from langlab.population import train_population, PopulationManager
from langlab.contact import train_contact_experiment


@pytest.mark.integration
@pytest.mark.slow
class TestTrainingIntegration:
    """Test complete training workflows."""

    def test_basic_training_workflow(self, sample_config, temp_output_dir):
        """Test basic Speaker-Listener training workflow."""
        # Set up temporary output directory
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            # Mock the actual training to avoid long execution
            with patch("langlab.train.train_step") as mock_train_step:
                mock_train_step.return_value = {
                    "speaker_loss": 0.5,
                    "listener_loss": 0.3,
                    "accuracy": 0.7,
                    "entropy": 1.2,
                    "baseline": 0.6,
                }

                # Run training
                train(
                    n_steps=10,
                    k=3,
                    v=sample_config.vocabulary_size,
                    message_length=sample_config.message_length,
                    seed=42,
                    log_every=5,
                    eval_every=10,
                    batch_size=8,
                    learning_rate=1e-3,
                    hidden_size=sample_config.hidden_size,
                    use_sequence_models=False,
                    entropy_weight=0.01,
                    length_weight=0.0,
                    multimodal=False,
                    distractors=0,
                )

                # Verify training step was called
                assert mock_train_step.call_count > 0

    def test_sequence_model_training(self, large_config, temp_output_dir):
        """Test training with sequence models."""
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            with patch("langlab.train.train_step") as mock_train_step:
                mock_train_step.return_value = {
                    "speaker_loss": 0.4,
                    "listener_loss": 0.2,
                    "accuracy": 0.8,
                    "entropy": 1.0,
                    "baseline": 0.7,
                }

                train(
                    n_steps=10,
                    k=3,
                    v=large_config.vocabulary_size,
                    message_length=large_config.message_length,
                    seed=42,
                    use_sequence_models=True,
                    entropy_weight=0.01,
                )

                assert mock_train_step.call_count > 0

    def test_multimodal_training(self, large_config, temp_output_dir):
        """Test training with multimodal communication."""
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            with patch("langlab.train.train_step") as mock_train_step:
                mock_train_step.return_value = {
                    "speaker_loss": 0.6,
                    "listener_loss": 0.4,
                    "accuracy": 0.6,
                    "entropy": 1.5,
                    "baseline": 0.5,
                }

                train(
                    n_steps=10,
                    k=3,
                    v=large_config.vocabulary_size,
                    message_length=large_config.message_length,
                    seed=42,
                    multimodal=True,
                    entropy_weight=0.01,
                )

                assert mock_train_step.call_count > 0

    def test_pragmatic_training(self, large_config, temp_output_dir):
        """Test training with pragmatic inference."""
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            with patch("langlab.train.train_step") as mock_train_step:
                mock_train_step.return_value = {
                    "speaker_loss": 0.7,
                    "listener_loss": 0.5,
                    "accuracy": 0.5,
                    "entropy": 1.8,
                    "baseline": 0.4,
                }

                train(
                    n_steps=10,
                    k=3,
                    v=large_config.vocabulary_size,
                    message_length=large_config.message_length,
                    seed=42,
                    distractors=2,
                    entropy_weight=0.01,
                )

                assert mock_train_step.call_count > 0

    def test_compositional_training(self, sample_config, temp_output_dir):
        """Test training with compositional splits."""
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            with patch("langlab.train.train_step") as mock_train_step:
                mock_train_step.return_value = {
                    "speaker_loss": 0.5,
                    "listener_loss": 0.3,
                    "accuracy": 0.7,
                    "entropy": 1.2,
                    "baseline": 0.6,
                }

                train(
                    n_steps=10,
                    k=3,
                    v=sample_config.vocabulary_size,
                    message_length=sample_config.message_length,
                    seed=42,
                    heldout_pairs=[("red", "circle")],
                    entropy_weight=0.01,
                )

                assert mock_train_step.call_count > 0


@pytest.mark.integration
@pytest.mark.slow
class TestPopulationIntegration:
    """Test population dynamics and cultural transmission."""

    def test_population_training_workflow(
        self, sample_population_config, temp_output_dir
    ):
        """Test population training workflow."""
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            with patch("langlab.population.train_step") as mock_train_step:
                mock_train_step.return_value = {
                    "speaker_loss": 0.5,
                    "listener_loss": 0.3,
                    "accuracy": 0.7,
                    "entropy": 1.2,
                    "baseline": 0.6,
                }

                train_population(
                    n_steps=20,
                    n_pairs=sample_population_config["n_pairs"],
                    lifespan=sample_population_config["lifespan"],
                    crossplay_prob=sample_population_config["crossplay_prob"],
                    replacement_noise=sample_population_config["replacement_noise"],
                    k=3,
                    v=sample_population_config["vocabulary_size"],
                    message_length=sample_population_config["message_length"],
                    seed=sample_population_config["seed"],
                    log_every=10,
                    batch_size=sample_population_config["batch_size"],
                    learning_rate=sample_population_config["learning_rate"],
                    hidden_size=sample_population_config["hidden_size"],
                    use_sequence_models=sample_population_config["use_sequence_models"],
                    entropy_weight=sample_population_config["entropy_weight"],
                )

                assert mock_train_step.call_count > 0

    def test_population_manager_workflow(self, sample_config):
        """Test PopulationManager workflow."""
        # Create a small population manager
        manager = PopulationManager(
            n_pairs=2,
            lifespan=50,
            replacement_noise=0.1,
            crossplay_prob=0.2,
            config=sample_config,
            seed=42,
        )

        # Test population initialization
        assert len(manager.population) == 2
        assert all(pair.age == 0 for pair in manager.population)

        # Test aging
        manager.age_population()
        assert all(pair.age == 1 for pair in manager.population)

        # Test replacement (age beyond lifespan)
        for _ in range(51):  # Age beyond lifespan
            manager.age_population()

        # Should trigger replacement
        manager.replace_old_agents()
        assert all(pair.age == 0 for pair in manager.population)

    def test_crossplay_interactions(self, sample_config):
        """Test cross-pair interactions in population."""
        manager = PopulationManager(
            n_pairs=3,
            lifespan=100,
            replacement_noise=0.1,
            crossplay_prob=0.5,  # High crossplay probability
            config=sample_config,
            seed=42,
        )

        # Test crossplay selection
        crossplay_pairs = manager.select_crossplay_pairs()
        assert len(crossplay_pairs) <= 3  # Should not exceed number of pairs

        # Test that crossplay pairs are different
        if len(crossplay_pairs) > 1:
            for i, pair1 in enumerate(crossplay_pairs):
                for j, pair2 in enumerate(crossplay_pairs):
                    if i != j:
                        assert pair1 != pair2


@pytest.mark.integration
@pytest.mark.slow
class TestContactIntegration:
    """Test contact experiments between populations."""

    def test_contact_experiment_workflow(self, sample_contact_config, temp_output_dir):
        """Test contact experiment workflow."""
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            with patch("langlab.contact.train_step") as mock_train_step:
                mock_train_step.return_value = {
                    "speaker_loss": 0.5,
                    "listener_loss": 0.3,
                    "accuracy": 0.7,
                    "entropy": 1.2,
                    "baseline": 0.6,
                }

                train_contact_experiment(
                    n_pairs=sample_contact_config["n_pairs"],
                    steps_a=sample_contact_config["steps_a"],
                    steps_b=sample_contact_config["steps_b"],
                    contact_steps=sample_contact_config["contact_steps"],
                    p_contact=sample_contact_config["p_contact"],
                    k=sample_contact_config["k"],
                    v=sample_contact_config["v"],
                    message_length=sample_contact_config["message_length"],
                    seed_a=sample_contact_config["seed_a"],
                    seed_b=sample_contact_config["seed_b"],
                    log_every=sample_contact_config["log_every"],
                    batch_size=sample_contact_config["batch_size"],
                    learning_rate=sample_contact_config["learning_rate"],
                    hidden_size=sample_contact_config["hidden_size"],
                    use_sequence_models=sample_contact_config["use_sequence_models"],
                    entropy_weight=sample_contact_config["entropy_weight"],
                    heldout_pairs_a=sample_contact_config["heldout_pairs_a"],
                    heldout_pairs_b=sample_contact_config["heldout_pairs_b"],
                )

                assert mock_train_step.call_count > 0

    def test_contact_experiment_config(self, sample_contact_config):
        """Test contact experiment configuration."""
        from langlab.contact import ContactConfig

        config = ContactConfig(
            n_pairs=sample_contact_config["n_pairs"],
            steps_a=sample_contact_config["steps_a"],
            steps_b=sample_contact_config["steps_b"],
            contact_steps=sample_contact_config["contact_steps"],
            p_contact=sample_contact_config["p_contact"],
            k=sample_contact_config["k"],
            v=sample_contact_config["v"],
            message_length=sample_contact_config["message_length"],
            seed_a=sample_contact_config["seed_a"],
            seed_b=sample_contact_config["seed_b"],
            log_every=sample_contact_config["log_every"],
            batch_size=sample_contact_config["batch_size"],
            learning_rate=sample_contact_config["learning_rate"],
            hidden_size=sample_contact_config["hidden_size"],
            use_sequence_models=sample_contact_config["use_sequence_models"],
            entropy_weight=sample_contact_config["entropy_weight"],
            heldout_pairs_a=sample_contact_config["heldout_pairs_a"],
            heldout_pairs_b=sample_contact_config["heldout_pairs_b"],
        )

        assert config.n_pairs == sample_contact_config["n_pairs"]
        assert config.steps_a == sample_contact_config["steps_a"]
        assert config.steps_b == sample_contact_config["steps_b"]
        assert config.contact_steps == sample_contact_config["contact_steps"]
        assert config.p_contact == sample_contact_config["p_contact"]


@pytest.mark.integration
class TestTrainingComponents:
    """Test individual training components integration."""

    def test_moving_average_baseline(self):
        """Test MovingAverage baseline functionality."""
        baseline = MovingAverage(window_size=5)

        # Test initial state
        assert baseline.average == 0.0

        # Test updates
        rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
        for reward in rewards:
            baseline.update(reward)

        # Test window behavior
        assert (
            abs(baseline.average - 0.3) < 1e-6
        )  # Average of [0.1, 0.2, 0.3, 0.4, 0.5]

        # Test window overflow
        baseline.update(0.6)
        assert (
            abs(baseline.average - 0.4) < 1e-6
        )  # Average of [0.2, 0.3, 0.4, 0.5, 0.6]

    def test_agent_initialization_integration(self, sample_config):
        """Test agent initialization in training context."""
        speaker = Speaker(sample_config)
        listener = Listener(sample_config)

        # Test that agents can be created
        assert speaker is not None
        assert listener is not None

        # Test that agents have correct architecture
        assert speaker.config.vocabulary_size == sample_config.vocabulary_size
        assert listener.config.vocabulary_size == sample_config.vocabulary_size

    def test_dataset_integration(self, sample_config):
        """Test dataset integration with training."""
        dataset = ReferentialGameDataset(n_scenes=5, k=3, seed=42)

        # Test dataset properties
        assert len(dataset) == 5

        # Test data loading
        scene_tensor, target_idx, candidates = dataset[0]
        assert scene_tensor.shape[0] == 3  # k objects
        assert scene_tensor.shape[1] == 8  # TOTAL_ATTRIBUTES
        assert 0 <= target_idx < 3
        assert len(candidates) == 3

    def test_sequence_model_integration(self, large_config):
        """Test sequence model integration."""
        speaker_seq = SpeakerSeq(large_config)
        listener_seq = ListenerSeq(large_config)

        # Test that sequence models can be created
        assert speaker_seq is not None
        assert listener_seq is not None

        # Test that they have sequence-specific components
        assert hasattr(speaker_seq, "gru")
        assert hasattr(listener_seq, "gru")
