"""Simplified Command-line interface for the Language Emergence Lab.

This module provides a strictly essential CLI for training and evaluating
emergent language models in referential games.
"""

import click
from typing import Optional

from ..data.world import sample_scene
from ..utils.utils import get_logger
from ..training.train import train as train_model
from ..analysis.eval import evaluate as evaluate_model

logger = get_logger(__name__)


@click.group()
@click.version_option()
def main() -> None:
    """Language Emergence Lab - Minimalist core.

    Focused strictly on training and evaluating Speaker/Listener agents.
    """
    pass


@main.command()
@click.option("--k", default=3, help="Number of objects in the scene")
@click.option("--seed", default=42, help="Random seed for reproducible generation")
def sample(k: int, seed: int) -> None:
    """Generate and display a sample scene."""

    scene_objects, target_idx = sample_scene(k, seed)
    click.echo(f"\nTarget object index: {target_idx}")
    for i, obj in enumerate(scene_objects):
        click.echo(f"  {i}: {obj}")


@main.command()
@click.option("--steps", default=10000, help="Number of training steps")
@click.option("--k", default=5, help="Number of objects per scene")
@click.option("--v", default=16, help="Vocabulary size")
@click.option(
    "--l", "--message-length", "message_length", default=2, help="Message length"
)
@click.option("--seed", default=7, help="Random seed")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--learning-rate", default=2e-4, help="Learning rate")
@click.option("--hidden-size", default=128, help="Hidden dimension size")
@click.option("--use-sequence-models", is_flag=True, help="Use sequence-aware models")
def train(
    steps: int,
    k: int,
    v: int,
    message_length: int,
    seed: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    use_sequence_models: bool,
) -> None:
    """Train Speaker and Listener agents for emergent language."""
    train_model(
        n_steps=steps,
        k=k,
        v=v,
        message_length=message_length,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        use_sequence_models=use_sequence_models,
    )
    click.echo("Training completed successfully!")


@main.command()
@click.option("--ckpt", required=True, help="Path to model checkpoint")
@click.option("--split", default="iid", help="Data split to evaluate (train/iid/compo)")
@click.option(
    "--heldout", default=None, help="Comma-separated held-out attribute pairs"
)
def eval(ckpt: str, split: str, heldout: Optional[str]) -> None:
    """Evaluate model performance on specified data split."""
    heldout_pairs = None
    if heldout:
        pairs = heldout.split(",")
        heldout_pairs = [(pairs[0].strip(), pairs[1].strip())]

    results = evaluate_model(model_path=ckpt, split=split, heldout_pairs=heldout_pairs)
    click.echo(f"Evaluation Results: {results}")


if __name__ == "__main__":
    main()
