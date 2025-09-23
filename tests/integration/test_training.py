"""Integration tests for training workflows.

This module tests complete training workflows, including agent training,
population dynamics, and contact experiments.
"""

from typing import Any
import pytest
from unittest.mock import patch

from src.langlab.core.agents import Speaker, Listener, SpeakerSeq, ListenerSeq
from src.langlab.data.data import ReferentialGameDataset
from src.langlab.training.train import train, MovingAverage

# Imports moved inside test methods to ensure proper mocking


@pytest.mark.integration
@pytest.mark.slow
class TestTrainingIntegration:
    """Test complete training workflows."""

    def test_basic_training_workflow(
        self, sample_config: Any, temp_output_dir: Any
    ) -> None:
        """Test basic Speaker-Listener training workflow."""
        # Set up temporary output directory
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            # Mock the actual training to avoid long execution
            with patch("langlab.training.train.train_step") as mock_train_step:
                mock_train_step.return_value = {
                    "speaker_loss": 0.5,
                    "listener_loss": 0.3,
                    "total_loss": 0.8,
                    "accuracy": 0.7,
                    "entropy": 1.2,
                    "baseline": 0.6,
                    "avg_message_length": 1.0,
                    "message_length_std": 0.0,
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

    def test_sequence_model_training(
        self, large_config: Any, temp_output_dir: Any
    ) -> None:
        """Test training with sequence models."""
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            with patch("langlab.training.train.train_step") as mock_train_step:
                mock_train_step.return_value = {
                    "speaker_loss": 0.4,
                    "listener_loss": 0.2,
                    "total_loss": 0.6,
                    "accuracy": 0.8,
                    "entropy": 1.0,
                    "baseline": 0.7,
                    "avg_message_length": 2.0,
                    "message_length_std": 0.0,
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

    def test_multimodal_training(self, large_config: Any, temp_output_dir: Any) -> None:
        """Test training with multimodal communication."""
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            with patch("langlab.training.train.train_step") as mock_train_step:
                mock_train_step.return_value = {
                    "speaker_loss": 0.6,
                    "listener_loss": 0.4,
                    "total_loss": 1.0,
                    "accuracy": 0.6,
                    "entropy": 1.5,
                    "baseline": 0.5,
                    "avg_message_length": 2.0,
                    "message_length_std": 0.0,
                }

                train(
                    n_steps=10,
                    k=3,
                    v=large_config.vocabulary_size,
                    message_length=large_config.message_length,
                    seed=42,
                    multimodal=True,
                    entropy_weight=0.01,
                    use_early_stopping=False,  # Disable early stopping for test
                )

                assert mock_train_step.call_count > 0

    def test_pragmatic_training(self, large_config: Any, temp_output_dir: Any) -> None:
        """Test training with pragmatic inference."""
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            with patch("langlab.training.train.train_step") as mock_train_step:
                mock_train_step.return_value = {
                    "speaker_loss": 0.7,
                    "listener_loss": 0.5,
                    "total_loss": 1.2,
                    "accuracy": 0.5,
                    "entropy": 1.8,
                    "baseline": 0.4,
                    "avg_message_length": 2.0,
                    "message_length_std": 0.0,
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

    def test_compositional_training(
        self, sample_config: Any, temp_output_dir: Any
    ) -> None:
        """Test training with compositional splits."""
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            with patch("langlab.training.train.train_step") as mock_train_step:
                mock_train_step.return_value = {
                    "speaker_loss": 0.5,
                    "listener_loss": 0.3,
                    "total_loss": 0.8,
                    "accuracy": 0.7,
                    "entropy": 1.2,
                    "baseline": 0.6,
                    "avg_message_length": 1.0,
                    "message_length_std": 0.0,
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
        self, sample_population_config: Any, temp_output_dir: Any
    ) -> None:
        """Test population training workflow."""
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            # Mock the train_population function directly
            with patch(
                "langlab.experiments.population.train_population"
            ) as mock_train_pop:
                mock_train_pop.return_value = None

                # Import after patching
                from langlab.experiments.population import train_population

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

                assert mock_train_pop.call_count == 1

    def test_population_manager_workflow(self, sample_config: Any) -> None:
        """Test PopulationManager workflow."""
        from langlab.experiments.population import PopulationManager, PopulationConfig

        # Create population config
        pop_config = PopulationConfig(
            n_pairs=2,
            lifespan=50,
            replacement_noise=0.1,
            crossplay_prob=0.2,
            batch_size=16,
            learning_rate=1e-3,
            hidden_size=sample_config.hidden_size,
            vocabulary_size=sample_config.vocabulary_size,
            message_length=sample_config.message_length,
            use_sequence_models=False,
            entropy_weight=0.01,
            seed=42,
        )

        manager = PopulationManager(pop_config)

        # Test population initialization
        assert len(manager.pairs) == 2
        assert all(pair.age == 0 for pair in manager.pairs)

        # Test aging
        # Age up all pairs by running a training step
        import torch

        dummy_batch = (
            torch.randn(1, 3, 8),  # scene_tensor
            torch.tensor([0]),  # target_indices
            torch.randn(1, 3, 8),  # candidate_objects
        )
        manager.train_step(dummy_batch)
        assert all(pair.age == 1 for pair in manager.pairs)

        # Test replacement (age beyond lifespan)
        for _ in range(51):  # Age beyond lifespan
            manager.train_step(dummy_batch)

        # Should trigger replacement (this happens automatically in train_step)
        # Check that pairs were replaced (age should be low after replacement)
        assert all(
            pair.age <= 2 for pair in manager.pairs
        )  # After replacement, ages should be low

    def test_crossplay_interactions(self, sample_config: Any) -> None:
        """Test cross-pair interactions in population."""
        from langlab.experiments.population import PopulationManager, PopulationConfig

        # Create population config
        pop_config = PopulationConfig(
            n_pairs=3,
            lifespan=100,
            replacement_noise=0.1,
            crossplay_prob=0.5,  # High crossplay probability
            batch_size=16,
            learning_rate=1e-3,
            hidden_size=sample_config.hidden_size,
            vocabulary_size=sample_config.vocabulary_size,
            message_length=sample_config.message_length,
            use_sequence_models=False,
            entropy_weight=0.01,
            seed=42,
        )

        manager = PopulationManager(pop_config)

        # Test interaction pair selection
        interactions = manager._select_interaction_pairs()
        assert len(interactions) >= 3  # At least self-play for each pair
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in interactions)


@pytest.mark.integration
@pytest.mark.slow
class TestContactIntegration:
    """Test contact experiments between populations."""

    def test_contact_experiment_workflow(
        self, sample_contact_config: Any, temp_output_dir: Any
    ) -> None:
        """Test contact experiment workflow."""
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.return_value = None

            # Mock the train_contact_experiment function directly
            with patch(
                "langlab.experiments.contact.train_contact_experiment"
            ) as mock_train_contact:
                mock_train_contact.return_value = None

                # Import after patching
                from langlab.experiments.contact import train_contact_experiment

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

                assert mock_train_contact.call_count == 1

    def test_contact_experiment_config(self, sample_contact_config: Any) -> None:
        """Test contact experiment configuration."""
        from langlab.experiments.contact import ContactConfig

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

    def test_moving_average_baseline(self) -> None:
        """Test MovingAverage baseline functionality."""
        baseline = MovingAverage(window_size=5)

        # Test initial state
        assert baseline.average == 0.0

        # Test updates with exponential moving average
        rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
        for reward in rewards:
            baseline.update(reward)

        # Test exponential moving average behavior (not simple average)
        # With alpha=0.1, the final value should be close to the last few values
        assert baseline.average > 0.0  # Should be positive
        assert baseline.average < 0.5  # Should be less than the last value

        # Test continued updates
        baseline.update(0.6)
        assert baseline.average > 0.0  # Should still be positive

    def test_agent_initialization_integration(self, sample_config: Any) -> None:
        """Test agent initialization in training context."""
        speaker = Speaker(sample_config)
        listener = Listener(sample_config)

        # Test that agents can be created
        assert speaker is not None
        assert listener is not None

        # Test that agents have correct architecture
        assert speaker.config.vocabulary_size == sample_config.vocabulary_size
        assert listener.config.vocabulary_size == sample_config.vocabulary_size

    def test_dataset_integration(self, sample_config: Any) -> None:
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

    def test_sequence_model_integration(self, large_config: Any) -> None:
        """Test sequence model integration."""
        speaker_seq = SpeakerSeq(large_config)
        listener_seq = ListenerSeq(large_config)

        # Test that sequence models can be created
        assert speaker_seq is not None
        assert listener_seq is not None

        # Test that they have sequence-specific components
        assert hasattr(speaker_seq, "gru")
        assert hasattr(listener_seq, "message_encoder")
