"""Command-line interface for the Language Emergence Lab.

This module provides a CLI for running experiments and exploring the
referential game framework from the command line.
"""

import click

from .world import sample_scene, COLORS, SHAPES, SIZES
from .data import ReferentialGameDataset
from .utils import get_logger, get_device


logger = get_logger(__name__)


@click.group()
@click.version_option()
def main() -> None:
    """Language Emergence Lab - studying proto-language in referential games.

    This CLI provides tools for exploring referential games and proto-language
    emergence in multi-agent systems.
    """
    pass


@main.command()
@click.option("--k", default=3, help="Number of objects in the scene")
@click.option("--seed", default=42, help="Random seed for reproducible generation")
def sample(k: int, seed: int) -> None:
    """Generate and display a sample scene."""
    logger.info(f"Generating scene with {k} objects (seed={seed})")

    try:
        scene_objects, target_idx = sample_scene(k, seed)

        click.echo(f"\nScene with {k} objects:")
        for i, obj in enumerate(scene_objects):
            marker = " (TARGET)" if i == target_idx else ""
            click.echo(f"  {i}: {obj['color']} {obj['size']} {obj['shape']}{marker}")

        click.echo(f"\nTarget object index: {target_idx}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option("--n-scenes", default=100, help="Number of scenes to generate")
@click.option("--k", default=3, help="Number of objects per scene")
@click.option("--seed", default=42, help="Random seed for reproducible generation")
def dataset(n_scenes: int, k: int, seed: int) -> None:
    """Generate and analyze a referential game dataset."""
    logger.info(
        f"Creating dataset with {n_scenes} scenes, {k} objects each (seed={seed})"
    )

    try:
        dataset = ReferentialGameDataset(n_scenes, k, seed)

        click.echo("\nDataset created:")
        click.echo(f"  Scenes: {len(dataset)}")
        click.echo(f"  Objects per scene: {k}")
        click.echo(f"  Object encoding dimension: {dataset[0][0].shape[1]}")

        # Show first scene
        scene_tensor, target_idx, candidates = dataset[0]
        click.echo("\nFirst scene:")
        click.echo(f"  Scene tensor shape: {scene_tensor.shape}")
        click.echo(f"  Target index: {target_idx}")

        # Analyze target distribution
        target_indices = [dataset[i][1] for i in range(min(10, len(dataset)))]
        click.echo(f"  Target indices (first 10): {target_indices}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)


@main.command()
def info() -> None:
    """Display information about the Language Emergence Lab."""
    click.echo("Language Emergence Lab")
    click.echo("====================")
    click.echo(f"Available colors: {', '.join(COLORS)}")
    click.echo(f"Available shapes: {', '.join(SHAPES)}")
    click.echo(f"Available sizes: {', '.join(SIZES)}")
    click.echo(f"Total possible objects: {len(COLORS) * len(SHAPES) * len(SIZES)}")
    click.echo(f"Object encoding dimension: {len(COLORS) + len(SHAPES) + len(SIZES)}")
    click.echo(f"Device: {get_device()}")


if __name__ == "__main__":
    main()
