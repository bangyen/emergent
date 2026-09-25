"""Command-line interface for the Language Emergence Lab."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import click

from ..analysis.eval import SPLITS
from ..analysis.eval import evaluate as evaluate_model
from ..data.world import WORLDS, sample_scene
from ..training.train import train as train_model
from ..utils.utils import get_logger

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


def _heldout(value: Optional[str]) -> Optional[List[Tuple[str, str]]]:
    try:
        return parse_heldout(value)
    except click.BadParameter as e:
        raise click.BadParameter(str(e), param_hint="--heldout")


def training_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """Options shared by ``train`` and ``pop-train``."""
    options = [
        click.option("--steps", "n_steps", default=10000, help="Training steps"),
        click.option("--k", default=5, help="Number of objects per scene"),
        click.option("--v", default=16, help="Vocabulary size"),
        click.option(
            "--l",
            "--message-length",
            "message_length",
            default=2,
            help="Message length",
        ),
        click.option("--seed", default=7, help="Random seed"),
        click.option("--batch-size", default=32, help="Batch size"),
        click.option("--learning-rate", default=2e-4, help="Learning rate"),
        click.option("--hidden-size", default=128, help="Hidden dimension size"),
        click.option(
            "--use-sequence-models", is_flag=True, help="Use GRU sequence agents"
        ),
        click.option("--entropy-weight", default=0.01, help="Speaker entropy bonus"),
        click.option("--heldout", default=None, help=HELDOUT_HELP),
        click.option(
            "--world",
            type=click.Choice(sorted(WORLDS)),
            default="default",
            help="Attribute space: default (18 objects) or large (225 objects)",
        ),
        click.option(
            "--hard-distractors",
            default=0,
            help="Distractors per training scene that share an attribute with the target",
        ),
        click.option(
            "--listener-type",
            type=click.Choice(["mlp", "dot"]),
            default="mlp",
            help="Listener architecture; 'dot' is additive and generalizes better",
        ),
        click.option("--eval-every", default=500, help="Evaluate every N steps"),
    ]
    for option in reversed(options):
        f = option(f)
    return f


def _echo_metrics(metrics: Dict[str, float]) -> None:
    click.echo("Training completed successfully!")
    for name, value in metrics.items():
        click.echo(f"  {name}: {value:.3f}")


@main.command()
@training_options
@click.option(
    "--out-dir", default="outputs", help="Directory for metrics and checkpoints"
)
def train(heldout: Optional[str], **kwargs: Any) -> None:
    """Train a Speaker/Listener pair."""
    _echo_metrics(train_model(heldout_pairs=_heldout(heldout), **kwargs))


@main.command(name="pop-train")
@training_options
@click.option("--agents", "n_agents", default=3, help="Speakers (and listeners)")
@click.option(
    "--lifespan",
    default=0,
    help="Replace the oldest agent every N steps (0 = no turnover)",
)
@click.option(
    "--listeners",
    "n_listeners",
    type=int,
    default=None,
    help="Number of listeners (default: same as --agents)",
)
@click.option("--out-dir", default="outputs/population", help="Output directory")
def pop_train(heldout: Optional[str], **kwargs: Any) -> None:
    """Train a population of agents in random pairings (cultural transmission)."""
    from ..training.population import train_population

    _echo_metrics(train_population(heldout_pairs=_heldout(heldout), **kwargs))


@main.command()
@click.option("--ckpt", required=True, help="Path to model checkpoint")
@click.option(
    "--split", type=click.Choice(SPLITS), default="iid", help="Data split to evaluate"
)
@click.option(
    "--heldout", default=None, help=HELDOUT_HELP + " (defaults to the checkpoint's)"
)
@click.option(
    "--pragmatic", is_flag=True, help="Use an RSA pragmatic listener (MLP agents)"
)
@click.option(
    "--num-distractors",
    type=int,
    default=None,
    help="Distractors per scene for --split distractor (default k-1)",
)
def eval(
    ckpt: str,
    split: str,
    heldout: Optional[str],
    pragmatic: bool,
    num_distractors: Optional[int],
) -> None:
    """Evaluate model performance on specified data split."""
    results = evaluate_model(
        model_path=ckpt,
        split=split,
        heldout_pairs=_heldout(heldout),
        pragmatic=pragmatic,
        num_distractors=num_distractors,
    )
    click.echo(f"Evaluation Results: {results}")


@main.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@click.option("--out-dir", default=None, help="Default: outputs/sweeps/<config name>")
@click.option("--jobs", default=1, help="Runs to execute in parallel")
@click.option(
    "--readme",
    default=None,
    help="Write the summary table into this file's results block",
)
def sweep(
    config: str, out_dir: Optional[str], jobs: int, readme: Optional[str]
) -> None:
    """Run a JSON sweep config over variants, grid points and seeds."""
    import json
    import os

    from ..training.sweep import run_sweep

    with open(config) as f:
        cfg = json.load(f)
    if out_dir is None:
        name = os.path.splitext(os.path.basename(config))[0]
        out_dir = os.path.join("outputs", "sweeps", name)
    click.echo(run_sweep(cfg, out_dir, jobs=jobs, readme=readme))


@main.command()
@click.argument("sweep_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--readme", default=None, help="README to update")
def report(sweep_dir: str, readme: Optional[str]) -> None:
    """Re-aggregate a finished sweep into summary.csv / summary.md."""
    from ..analysis.report import create_report

    click.echo(create_report(sweep_dir, readme=readme))


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
