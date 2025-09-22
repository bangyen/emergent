"""Experiment tracking implementation for Language Emergence Lab.

This module provides experiment tracking using MLflow, allowing researchers to
monitor training progress, log hyperparameters, and track metrics across different experiments.
"""

import os
from typing import Dict, Any, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass

from ..utils.utils import get_logger

if TYPE_CHECKING:
    from mlflow.entities import Run

logger = get_logger(__name__)


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking.

    This class defines the configuration parameters for experiment tracking,
    including project settings and logging preferences.
    """

    project_name: str = "langlab-emergent"
    experiment_name: Optional[str] = None
    tags: Optional[list] = None
    notes: Optional[str] = None
    log_frequency: int = 100
    log_artifacts: bool = True

    # MLflow specific settings
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None


class ExperimentTracker:
    """MLflow experiment tracker.

    This class provides a simple interface for experiment tracking using MLflow,
    allowing seamless logging of hyperparameters, metrics, and artifacts during training.
    """

    def __init__(self, config: TrackingConfig):
        """Initialize the experiment tracker.

        Args:
            config: Tracking configuration specifying settings.
        """
        self.config = config
        self.mlflow_run: Optional["Run"] = None
        self._initialized = False

        # Initialize tracking
        self._init_tracking()

    def _init_tracking(self) -> None:
        """Initialize MLflow tracking."""
        try:
            import mlflow
            import mlflow.pytorch

            # Set tracking URI if specified
            if self.config.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

            # Set experiment name
            experiment_name = (
                self.config.mlflow_experiment_name or self.config.project_name
            )
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                else:
                    experiment_id = experiment.experiment_id
            except Exception:
                # Fallback to default experiment
                experiment_id = "0"

            # Start MLflow run
            self.mlflow_run = mlflow.start_run(
                experiment_id=experiment_id, run_name=self.config.experiment_name
            )

            if self.mlflow_run and hasattr(self.mlflow_run, "info"):
                logger.info(f"MLflow run initialized: {self.mlflow_run.info.run_id}")
            else:
                logger.info("MLflow run initialized successfully")

            self._initialized = True
            logger.info("Experiment tracking initialized with MLflow")

        except ImportError:
            logger.warning("MLflow not available, skipping MLflow initialization")
            self._initialized = False
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            logger.warning("Continuing without experiment tracking...")
            self._initialized = False

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to MLflow.

        Args:
            params: Dictionary of hyperparameters to log.
        """
        if not self._initialized:
            return

        try:
            import mlflow

            mlflow.log_params(params)
            logger.debug(f"Logged parameters: {list(params.keys())}")

        except Exception as e:
            logger.warning(f"Failed to log parameters: {e}")

    def log_metrics(
        self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None
    ) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log.
            step: Optional step number for the metrics.
        """
        if not self._initialized:
            return

        try:
            import mlflow

            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged metrics at step {step}: {list(metrics.keys())}")

        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact to MLflow.

        Args:
            file_path: Path to the file to log as artifact.
            artifact_path: Optional path within the artifact store.
        """
        if not self._initialized or not self.config.log_artifacts:
            return

        try:
            import mlflow

            mlflow.log_artifact(file_path, artifact_path)
            logger.debug(f"Logged artifact: {file_path}")

        except Exception as e:
            logger.warning(f"Failed to log artifact {file_path}: {e}")

    def log_model(self, model_path: str, model_name: str = "model") -> None:
        """Log a trained model to MLflow.

        Args:
            model_path: Path to the saved model.
            model_name: Name for the model in the tracking system.
        """
        if not self._initialized or not self.config.log_artifacts:
            return

        try:
            import mlflow
            import mlflow.pytorch

            mlflow.pytorch.log_model(
                pytorch_model=model_path,
                artifact_path=model_name,
                registered_model_name=model_name,
            )

            logger.info(f"Logged model: {model_path}")

        except Exception as e:
            logger.warning(f"Failed to log model {model_path}: {e}")

    def log_figure(self, figure: Any, name: str, step: Optional[int] = None) -> None:
        """Log a matplotlib figure to MLflow.

        Args:
            figure: Matplotlib figure object.
            name: Name for the figure in the tracking system.
            step: Optional step number for the figure.
        """
        if not self._initialized:
            return

        try:
            import mlflow
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                figure.savefig(tmp.name, dpi=150, bbox_inches="tight")
                mlflow.log_artifact(tmp.name, f"figures/{name}.png")
                os.unlink(tmp.name)

            logger.debug(f"Logged figure: {name}")

        except Exception as e:
            logger.warning(f"Failed to log figure {name}: {e}")

    def finish(self) -> None:
        """Finish the experiment tracking session."""
        if not self._initialized:
            return

        try:
            import mlflow

            mlflow.end_run()
            logger.info("Experiment tracking session finished")

        except Exception as e:
            logger.warning(f"Failed to finish experiment tracking: {e}")

    def __enter__(self) -> "ExperimentTracker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.finish()


def get_tracker(
    project_name: str = "langlab-emergent",
    experiment_name: Optional[str] = None,
    **kwargs: Any,
) -> ExperimentTracker:
    """Create an experiment tracker with the specified configuration.

    This is a convenience function for creating experiment trackers with
    common configurations.

    Args:
        project_name: Name of the project for tracking.
        experiment_name: Name of the experiment run.
        **kwargs: Additional configuration parameters.

    Returns:
        Configured ExperimentTracker instance.
    """
    config = TrackingConfig(
        project_name=project_name,
        experiment_name=experiment_name,
        **kwargs,
    )

    return ExperimentTracker(config)
