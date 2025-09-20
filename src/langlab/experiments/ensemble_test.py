"""Test ensemble methods for improved model performance.

This script trains multiple models with different seeds and tests
ensemble performance against individual models.
"""

import os
import sys
import subprocess
from typing import List, Dict
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from langlab.core.ensemble import create_ensemble_from_checkpoints
from langlab.core.config import CommunicationConfig
from langlab.analysis.eval import evaluate


def train_multiple_models(
    num_models: int = 3,
    steps: int = 1000,
    k: int = 5,
    v: int = 24,
    message_length: int = 2,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    hidden_size: int = 128,
) -> List[str]:
    """Train multiple models with different seeds.

    Args:
        num_models: Number of models to train.
        steps: Number of training steps.
        k: Number of objects per scene.
        v: Vocabulary size.
        message_length: Message length.
        batch_size: Batch size.
        learning_rate: Learning rate.
        hidden_size: Hidden size.

    Returns:
        List of checkpoint paths.
    """
    checkpoint_paths = []

    for i in range(num_models):
        seed = 7 + i  # Different seeds for each model
        checkpoint_path = f"outputs/checkpoints/ensemble_model_{i}.pt"

        print(f"Training model {i+1}/{num_models} with seed {seed}")

        # Train model
        cmd = [
            "python",
            "-m",
            "src.langlab.apps.cli",
            "train",
            "--steps",
            str(steps),
            "--k",
            str(k),
            "--v",
            str(v),
            "--l",
            str(message_length),
            "--seed",
            str(seed),
            "--log-every",
            "100",
            "--eval-every",
            "200",
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(learning_rate),
            "--hidden-size",
            str(hidden_size),
            "--early-stopping",
            "--early-stopping-patience",
            "20",
            "--early-stopping-min-delta",
            "0.01",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Copy checkpoint to ensemble-specific name
            import shutil

            shutil.copy("outputs/checkpoints/checkpoint.pt", checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
            print(f"Model {i+1} trained successfully")
        else:
            print(f"Error training model {i+1}: {result.stderr}")

    return checkpoint_paths


def test_ensemble_performance(
    checkpoint_paths: List[str],
    config: CommunicationConfig,
    n_scenes: int = 1000,
    k: int = 5,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Test ensemble performance against individual models.

    Args:
        checkpoint_paths: List of checkpoint paths.
        config: Communication configuration.
        n_scenes: Number of scenes for evaluation.
        k: Number of objects per scene.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary of accuracies for each model and ensemble.
    """
    results = {}

    # Test individual models
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"Evaluating individual model {i+1}")
        result = evaluate(
            checkpoint_path,
            split="train",
            n_scenes=n_scenes,
            k=k,
            batch_size=batch_size,
        )
        results[f"model_{i+1}"] = result["acc"]
        print(f"Model {i+1} accuracy: {result['acc']:.4f}")

    # Test ensemble
    if len(checkpoint_paths) > 1:
        print("Evaluating ensemble model")
        try:
            ensemble_speaker, ensemble_listener = create_ensemble_from_checkpoints(
                checkpoint_paths, config
            )

            # Create evaluation dataset
            from langlab.data.data import ReferentialGameDataset
            from torch.utils.data import DataLoader

            dataset = ReferentialGameDataset(n_scenes, k, seed=42)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Evaluate ensemble
            ensemble_speaker.eval()
            ensemble_listener.eval()

            total_correct = 0
            total_samples = 0

            with torch.no_grad():
                for batch in dataloader:
                    scene_tensor, target_indices, candidate_objects = batch

                    # Extract target objects
                    batch_size = scene_tensor.size(0)
                    target_objects = scene_tensor[
                        torch.arange(batch_size), target_indices
                    ]

                    # Generate messages
                    _, message_tokens, _, _ = ensemble_speaker(target_objects)

                    # Make predictions
                    listener_probs = ensemble_listener(
                        message_tokens, candidate_objects
                    )
                    listener_predictions = torch.argmax(listener_probs, dim=1)

                    # Count correct predictions
                    correct = (listener_predictions == target_indices).sum().item()
                    total_correct += correct
                    total_samples += batch_size

            ensemble_accuracy = (
                total_correct / total_samples if total_samples > 0 else 0.0
            )
            results["ensemble"] = ensemble_accuracy
            print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")

        except Exception as e:
            print(f"Error evaluating ensemble: {e}")
            results["ensemble"] = 0.0

    return results


def main() -> None:
    """Main function to run ensemble experiment."""
    print("Starting ensemble experiment")

    # Configuration
    config = CommunicationConfig(
        vocabulary_size=24, message_length=2, hidden_size=128, seed=7
    )

    # Train multiple models
    checkpoint_paths = train_multiple_models(
        num_models=3,
        steps=1000,
        k=5,
        v=24,
        message_length=2,
        batch_size=32,
        learning_rate=5e-4,
        hidden_size=128,
    )

    if not checkpoint_paths:
        print("No models were trained successfully")
        return

    # Test ensemble performance
    results = test_ensemble_performance(
        checkpoint_paths, config, n_scenes=1000, k=5, batch_size=32
    )

    # Print results
    print("\n" + "=" * 50)
    print("ENSEMBLE EXPERIMENT RESULTS")
    print("=" * 50)

    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f} ({accuracy*100:.1f}%)")

    if "ensemble" in results and len(results) > 1:
        individual_avg = sum(v for k, v in results.items() if k != "ensemble") / (
            len(results) - 1
        )
        ensemble_improvement = results["ensemble"] - individual_avg
        print(
            f"\nEnsemble improvement: {ensemble_improvement:.4f} ({ensemble_improvement*100:.1f}%)"
        )

    print("=" * 50)


if __name__ == "__main__":
    main()
