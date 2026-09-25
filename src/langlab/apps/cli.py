"""Simplified Command-line interface for the Language Emergence Lab.

This module provides a strictly essential CLI for training and evaluating
emergent language models in referential games.
"""

import click
from typing import List, Optional, Tuple

from ..data.world import sample_scene
from ..utils.utils import get_logger
from ..training.train import train as train_model
from ..analysis.eval import evaluate as evaluate_model

logger = get_logger(__name__)

HELDOUT_HELP = (
    "Held-out attribute pairs as comma-separated values taken two at a time, "
    "e.g. 'red,circle' or 'red,circle,blue,square'"
)


def parse_heldout(value: Optional[str]) -> Optional[List[Tuple[str, str]]]:
    """Parse 'a,b,c,d' into [(a, b), (c, d)]."""
    if not value:
        return None
    attrs = [a.strip() for a in value.split(",") if a.strip()]
    if len(attrs) % 2:
        raise click.BadParameter("expected an even number of attributes")
    return list(zip(attrs[::2], attrs[1::2]))


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
@click.option("--entropy-weight", default=0.01, help="Speaker entropy bonus weight")
@click.option("--heldout", default=None, help=HELDOUT_HELP)
@click.option("--eval-every", default=500, help="Evaluate every N steps")
@click.option(
    "--out-dir", default="outputs", help="Directory for metrics and checkpoints"
)
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
    entropy_weight: float,
    heldout: Optional[str],
    eval_every: int,
    out_dir: str,
) -> None:
    """Train Speaker and Listener agents for emergent language."""
    try:
        heldout_pairs = parse_heldout(heldout)
    except click.BadParameter as e:
        raise click.BadParameter(str(e), param_hint="--heldout")
    metrics = train_model(
        n_steps=steps,
        k=k,
        v=v,
        message_length=message_length,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        use_sequence_models=use_sequence_models,
        entropy_weight=entropy_weight,
        heldout_pairs=heldout_pairs,
        out_dir=out_dir,
        eval_every=eval_every,
    )
    click.echo("Training completed successfully!")
    for name, value in metrics.items():
        click.echo(f"  {name}: {value:.3f}")


@main.command()
@click.option("--ckpt", required=True, help="Path to model checkpoint")
@click.option("--split", default="iid", help="Data split to evaluate (train/iid/compo)")
@click.option(
    "--heldout", default=None, help=HELDOUT_HELP + " (defaults to the checkpoint's)"
)
def eval(ckpt: str, split: str, heldout: Optional[str]) -> None:
    """Evaluate model performance on specified data split."""
    try:
        heldout_pairs = parse_heldout(heldout)
    except click.BadParameter as e:
        raise click.BadParameter(str(e), param_hint="--heldout")

    results = evaluate_model(model_path=ckpt, split=split, heldout_pairs=heldout_pairs)
    click.echo(f"Evaluation Results: {results}")


@main.command()
@click.option("--metrics", default="outputs/metrics.csv", help="Path to metrics.csv")
@click.option("--out", default="outputs/training_curve.png", help="Output image path")
def plot(metrics: str, out: str) -> None:
    """Plot a training curve from metrics.csv (needs the 'analysis' extra)."""
    try:
        from ..analysis.analysis import plot_training_curve
    except ImportError as e:
        raise click.ClickException(
            f"{e}. Install plotting dependencies with: pip install 'langlab[analysis]'"
        )
    plot_training_curve(metrics, out)
    click.echo(f"Saved {out}")


if __name__ == "__main__":
    main()
