"""Command-line interface for the Language Emergence Lab.

This module provides a CLI for running experiments and exploring the
referential game framework from the command line.
"""

import click
from typing import Optional

from ..data.world import sample_scene, COLORS, SHAPES, SIZES
from ..data.data import ReferentialGameDataset
from ..utils.utils import get_logger, get_device
from ..training.train import train
from ..analysis.eval import evaluate
from ..experiments.population import train_population
from ..experiments.contact import train_contact_experiment
from ..training.train_grounded import train_grounded
from ..experiments.ablate import run_ablation_suite
from ..analysis.report import create_report


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
@click.option("--steps", default=10000, help="Number of training steps")
@click.option("--k", default=5, help="Number of objects per scene")
@click.option("--v", default=24, help="Vocabulary size")
@click.option(
    "--l", "--message-length", "message_length", default=2, help="Message length"
)
@click.option("--seed", default=7, help="Random seed")
@click.option("--log-every", default=100, help="Logging frequency")
@click.option("--eval-every", default=500, help="Checkpoint frequency")
@click.option("--lambda-speaker", default=1.0, help="Speaker loss weight")
@click.option("--batch-size", default=64, help="Batch size")
@click.option("--learning-rate", default=5e-4, help="Learning rate")
@click.option("--hidden-size", default=128, help="Hidden dimension size")
@click.option(
    "--use-sequence-models",
    is_flag=True,
    help="Use sequence-aware models (SpeakerSeq/ListenerSeq)",
)
@click.option(
    "--entropy-weight", default=0.05, help="Weight for entropy bonus regularization"
)
@click.option(
    "--length-weight", default=0.0, help="Weight for length cost regularization"
)
@click.option(
    "--heldout",
    default=None,
    help="Comma-separated held-out attribute pairs (e.g., 'blue,triangle')",
)
@click.option(
    "--multimodal",
    default=0,
    help="Enable multimodal communication with gestures (0=disabled, 1=enabled)",
)
@click.option(
    "--distractors",
    default=0,
    help="Number of distractor objects for pragmatic inference",
)
@click.option(
    "--temperature-start",
    default=2.0,
    help="Starting temperature for Gumbel-Softmax sampling",
)
@click.option(
    "--temperature-end",
    default=0.5,
    help="Ending temperature for Gumbel-Softmax sampling",
)
def train_cmd(
    steps: int,
    k: int,
    v: int,
    message_length: int,
    seed: int,
    log_every: int,
    eval_every: int,
    lambda_speaker: float,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    use_sequence_models: bool,
    entropy_weight: float,
    length_weight: float,
    heldout: Optional[str],
    multimodal: int,
    distractors: int,
    temperature_start: float,
    temperature_end: float,
) -> None:
    """Train Speaker and Listener agents for emergent language."""
    logger.info(
        f"Starting training: steps={steps}, k={k}, v={v}, message_length={message_length}, seed={seed}"
    )
    if use_sequence_models:
        logger.info("Using sequence-aware models with autoregressive generation")
    if multimodal:
        logger.info("Using multimodal communication with gestures")
    if distractors > 0:
        logger.info(f"Using {distractors} distractor objects for pragmatic inference")

    # Parse heldout pairs
    heldout_pairs = None
    if heldout:
        pairs = heldout.split(",")
        if len(pairs) != 2:
            click.echo(
                "Error: heldout must be exactly two comma-separated attributes",
                err=True,
            )
            return
        heldout_pairs = [(pairs[0].strip(), pairs[1].strip())]
        logger.info(f"Using compositional splits with heldout pairs: {heldout_pairs}")

    try:
        train(
            n_steps=steps,
            k=k,
            v=v,
            message_length=message_length,
            seed=seed,
            log_every=log_every,
            eval_every=eval_every,
            lambda_speaker=lambda_speaker,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            use_sequence_models=use_sequence_models,
            entropy_weight=entropy_weight,
            length_weight=length_weight,
            heldout_pairs=heldout_pairs,
            multimodal=bool(multimodal),
            distractors=distractors,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
        )
        click.echo("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option("--ckpt", required=True, help="Path to model checkpoint")
@click.option(
    "--split", default="compo", help="Data split to evaluate (train/iid/compo)"
)
@click.option(
    "--heldout",
    default="blue,triangle",
    help="Comma-separated held-out attribute pairs",
)
@click.option("--n-scenes", default=1000, help="Number of scenes for evaluation")
@click.option("--k", default=5, help="Number of objects per scene")
@click.option("--batch-size", default=32, help="Batch size for evaluation")
def eval_cmd(
    ckpt: str,
    split: str,
    heldout: str,
    n_scenes: int,
    k: int,
    batch_size: int,
) -> None:
    """Evaluate model performance on specified data split."""
    logger.info(f"Evaluating model {ckpt} on {split} split")

    # Parse heldout pairs
    pairs = heldout.split(",")
    if len(pairs) != 2:
        click.echo(
            "Error: heldout must be exactly two comma-separated attributes", err=True
        )
        return
    heldout_pairs = [(pairs[0].strip(), pairs[1].strip())]

    try:
        results = evaluate(
            model_path=ckpt,
            split=split,
            heldout_pairs=heldout_pairs,
            n_scenes=n_scenes,
            k=k,
            batch_size=batch_size,
        )

        click.echo("\nEvaluation Results:")
        click.echo(f"  Split: {split}")
        click.echo(f"  Accuracy: {results['acc']:.4f}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option("--pairs", default=5, help="Number of agent pairs in the population")
@click.option("--lifespan", default=1000, help="Maximum age before agent replacement")
@click.option("--steps", default=10000, help="Number of training steps")
@click.option("--crossplay", default=0.1, help="Probability of cross-pair interactions")
@click.option(
    "--replacement-noise",
    default=0.1,
    help="Standard deviation of Gaussian noise for new agents",
)
@click.option("--k", default=5, help="Number of objects per scene")
@click.option("--v", default=10, help="Vocabulary size")
@click.option(
    "--l", "--message-length", "message_length", default=1, help="Message length"
)
@click.option("--seed", default=42, help="Random seed")
@click.option("--log-every", default=100, help="Logging frequency")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--learning-rate", default=1e-3, help="Learning rate")
@click.option("--hidden-size", default=64, help="Hidden dimension size")
@click.option("--use-sequence-models", is_flag=True, help="Use sequence-aware models")
@click.option(
    "--entropy-weight", default=0.01, help="Weight for entropy bonus regularization"
)
def pop_train(
    pairs: int,
    lifespan: int,
    steps: int,
    crossplay: float,
    replacement_noise: float,
    k: int,
    v: int,
    message_length: int,
    seed: int,
    log_every: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    use_sequence_models: bool,
    entropy_weight: float,
) -> None:
    """Train a population of agent pairs for cultural transmission studies."""
    logger.info(
        f"Starting population training: pairs={pairs}, lifespan={lifespan}, "
        f"steps={steps}, crossplay={crossplay}, seed={seed}"
    )

    if use_sequence_models:
        logger.info("Using sequence-aware models with autoregressive generation")

    try:
        train_population(
            n_steps=steps,
            n_pairs=pairs,
            lifespan=lifespan,
            crossplay_prob=crossplay,
            replacement_noise=replacement_noise,
            k=k,
            v=v,
            message_length=message_length,
            seed=seed,
            log_every=log_every,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            use_sequence_models=use_sequence_models,
            entropy_weight=entropy_weight,
        )
        click.echo("Population training completed successfully!")

    except Exception as e:
        logger.error(f"Population training failed: {e}")
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option("--pairs", default=4, help="Number of agent pairs per population")
@click.option(
    "--steps-a",
    default=4000,
    help="Number of training steps for Stage A (separate training)",
)
@click.option(
    "--steps-b",
    default=4000,
    help="Number of training steps for Stage B (contact phase)",
)
@click.option("--contact-steps", default=2000, help="Number of contact phase steps")
@click.option(
    "--p-contact", default=0.3, help="Probability of cross-population interactions"
)
@click.option("--k", default=5, help="Number of objects per scene")
@click.option("--v", default=10, help="Vocabulary size")
@click.option(
    "--l", "--message-length", "message_length", default=1, help="Message length"
)
@click.option("--seed-a", default=42, help="Random seed for Population A")
@click.option("--seed-b", default=123, help="Random seed for Population B")
@click.option("--log-every", default=100, help="Logging frequency")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--learning-rate", default=1e-3, help="Learning rate")
@click.option("--hidden-size", default=64, help="Hidden dimension size")
@click.option("--use-sequence-models", is_flag=True, help="Use sequence-aware models")
@click.option(
    "--entropy-weight", default=0.01, help="Weight for entropy bonus regularization"
)
@click.option(
    "--heldout-a", default="blue,triangle", help="Held-out pairs for Population A"
)
@click.option(
    "--heldout-b", default="red,circle", help="Held-out pairs for Population B"
)
def contact(
    pairs: int,
    steps_a: int,
    steps_b: int,
    contact_steps: int,
    p_contact: float,
    k: int,
    v: int,
    message_length: int,
    seed_a: int,
    seed_b: int,
    log_every: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    use_sequence_models: bool,
    entropy_weight: float,
    heldout_a: str,
    heldout_b: str,
) -> None:
    """Train two populations separately, then bring them into contact and measure intelligibility."""
    logger.info(
        f"Starting contact experiment: pairs={pairs}, steps_a={steps_a}, "
        f"steps_b={steps_b}, contact_steps={contact_steps}, p_contact={p_contact}"
    )

    # Parse heldout pairs
    def parse_heldout(heldout_str: str) -> list:
        pairs = heldout_str.split(",")
        if len(pairs) != 2:
            raise ValueError("heldout must be exactly two comma-separated attributes")
        return [(pairs[0].strip(), pairs[1].strip())]

    try:
        heldout_pairs_a = parse_heldout(heldout_a)
        heldout_pairs_b = parse_heldout(heldout_b)

        logger.info(f"Population A heldout pairs: {heldout_pairs_a}")
        logger.info(f"Population B heldout pairs: {heldout_pairs_b}")

        train_contact_experiment(
            n_pairs=pairs,
            steps_a=steps_a,
            steps_b=steps_b,
            contact_steps=contact_steps,
            p_contact=p_contact,
            k=k,
            v=v,
            message_length=message_length,
            seed_a=seed_a,
            seed_b=seed_b,
            log_every=log_every,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            use_sequence_models=use_sequence_models,
            entropy_weight=entropy_weight,
            heldout_pairs_a=heldout_pairs_a,
            heldout_pairs_b=heldout_pairs_b,
        )
        click.echo("Contact experiment completed successfully!")

    except Exception as e:
        logger.error(f"Contact experiment failed: {e}")
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option("--port", default=8888, help="Port to run the dashboard on")
@click.option("--host", default="localhost", help="Host to run the dashboard on")
def dash(port: int, host: str) -> None:
    """Launch the interactive Streamlit dashboard for visualizing language emergence."""
    import subprocess
    import sys
    import os

    # Get the path to the app.py file
    app_path = os.path.join(os.path.dirname(__file__), "app.py")

    if not os.path.exists(app_path):
        click.echo("Error: Dashboard app not found", err=True)
        return

    logger.info(f"Launching dashboard on {host}:{port}")
    click.echo("Launching Language Emergence Dashboard...")
    click.echo(f"Dashboard will be available at: http://{host}:{port}")
    click.echo("Press Ctrl+C to stop the dashboard")

    try:
        # Launch Streamlit
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            app_path,
            "--server.headless",
            "true",
            "--server.port",
            str(port),
            "--server.address",
            host,
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch dashboard: {e}")
        click.echo(
            "Error: Failed to launch dashboard. Make sure Streamlit is installed.",
            err=True,
        )
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped.")


@main.command()
@click.option("--episodes", default=500, help="Number of training episodes")
@click.option("--grid", default=5, help="Size of the grid world")
@click.option(
    "--l", "--message-length", "message_length", default=3, help="Message length"
)
@click.option("--v", default=12, help="Vocabulary size")
@click.option("--seed", default=3, help="Random seed")
@click.option("--max-steps", default=15, help="Maximum steps per episode")
@click.option("--hidden-size", default=64, help="Hidden layer size")
@click.option("--learning-rate", default=1e-3, help="Learning rate")
@click.option("--entropy-weight", default=0.01, help="Weight for entropy bonus")
@click.option("--log-every", default=50, help="Logging frequency")
@click.option("--eval-every", default=100, help="Evaluation frequency")
@click.option("--use-curriculum", is_flag=True, help="Use curriculum learning")
def train_grid(
    episodes: int,
    grid: int,
    message_length: int,
    v: int,
    seed: int,
    max_steps: int,
    hidden_size: int,
    learning_rate: float,
    entropy_weight: float,
    log_every: int,
    eval_every: int,
    use_curriculum: bool,
) -> None:
    """Train grounded Speaker and Listener agents in grid world navigation."""
    logger.info(
        f"Starting grounded training: episodes={episodes}, grid_size={grid}, "
        f"vocab={v}, message_length={message_length}, seed={seed}"
    )

    if use_curriculum:
        logger.info("Using curriculum learning with progressive difficulty")

    try:
        train_grounded(
            episodes=episodes,
            grid_size=grid,
            max_steps=max_steps,
            vocabulary_size=v,
            message_length=message_length,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            entropy_weight=entropy_weight,
            log_every=log_every,
            eval_every=eval_every,
            seed=seed,
            use_curriculum=use_curriculum,
        )
        click.echo("Grounded training completed successfully!")

    except Exception as e:
        logger.error(f"Grounded training failed: {e}")
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option("--runs", default=6, help="Number of runs to perform (currently unused)")
@click.option(
    "--vocab-sizes", default="6,12,24", help="Comma-separated vocabulary sizes"
)
@click.option(
    "--noise-levels", default="0,0.05,0.1", help="Comma-separated channel noise levels"
)
@click.option(
    "--length-costs", default="0,0.01,0.05", help="Comma-separated length cost weights"
)
@click.option("--steps", default=2000, help="Number of training steps per experiment")
@click.option("--k", default=5, help="Number of objects per scene")
@click.option("--message-length", default=1, help="Message length")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--learning-rate", default=1e-3, help="Learning rate")
@click.option("--hidden-size", default=64, help="Hidden dimension size")
@click.option("--entropy-weight", default=0.01, help="Entropy bonus weight")
@click.option("--seed", default=42, help="Base random seed")
def ablate(
    runs: int,
    vocab_sizes: str,
    noise_levels: str,
    length_costs: str,
    steps: int,
    k: int,
    message_length: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    entropy_weight: float,
    seed: int,
) -> None:
    """Run ablation studies across parameter configurations."""
    logger.info("Starting ablation study suite")

    # Parse parameter lists
    try:
        vocab_list = [int(x.strip()) for x in vocab_sizes.split(",")]
        noise_list = [float(x.strip()) for x in noise_levels.split(",")]
        length_list = [float(x.strip()) for x in length_costs.split(",")]
    except ValueError as e:
        click.echo(f"Error parsing parameters: {e}", err=True)
        return

    click.echo("Running ablation study with:")
    click.echo(f"  Vocabulary sizes: {vocab_list}")
    click.echo(f"  Noise levels: {noise_list}")
    click.echo(f"  Length costs: {length_list}")
    click.echo(
        f"  Total experiments: {len(vocab_list) * len(noise_list) * len(length_list)}"
    )

    try:
        results = run_ablation_suite(
            runs=runs,
            vocab_sizes=vocab_list,
            channel_noise_levels=noise_list,
            length_costs=length_list,
            base_seed=seed,
            n_steps=steps,
            k=k,
            message_length=message_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            entropy_weight=entropy_weight,
        )

        click.echo("Ablation study completed successfully!")
        click.echo("Results saved to outputs/experiments/")
        click.echo(f"Total experiments completed: {len(results)}")

    except Exception as e:
        logger.error(f"Ablation study failed: {e}")
        click.echo(f"Error: {e}", err=True)


@main.command()
@click.option(
    "--input",
    required=True,
    help="Input pattern for experiment results (e.g., 'outputs/experiments/**/metrics.json')",
)
@click.option(
    "--output-dir", default="outputs/summary", help="Output directory for report files"
)
@click.option("--no-charts", is_flag=True, help="Skip generating charts")
def report(input: str, output_dir: str, no_charts: bool) -> None:
    """Generate ablation study report from experiment results."""
    logger.info(f"Creating report from: {input}")

    try:
        report_info = create_report(
            input_pattern=input,
            output_dir=output_dir,
            create_charts=not no_charts,
        )

        if "error" in report_info:
            click.echo(f"Error: {report_info['error']}", err=True)
            return

        click.echo("Report generated successfully!")
        click.echo(f"CSV file: {report_info['csv_path']}")
        click.echo(f"Summary: {report_info['summary_path']}")

        if not no_charts:
            click.echo("Charts generated:")
            if "accuracy_chart" in report_info:
                click.echo(f"  Accuracy bars: {report_info['accuracy_chart']}")
            if "compo_chart" in report_info:
                click.echo(f"  Compositional bars: {report_info['compo_chart']}")
            if "heatmap" in report_info:
                click.echo(f"  Heatmap: {report_info['heatmap']}")

        click.echo(f"Total experiments processed: {report_info['total_experiments']}")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
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
