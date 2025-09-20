"""Utility functions for seeding, device selection, and logging.

This module provides essential utilities for reproducible experiments and
device management in the Language Emergence Lab.
"""

import logging
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments.

    This function ensures that all random number generators (Python's random,
    NumPy, and PyTorch) use the same seed for reproducible results across
    different runs of the same experiment.

    Args:
        seed: The seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the appropriate device for tensor operations.

    This function automatically selects the best available device for computation,
    preferring CUDA if available and requested.

    Args:
        prefer_cuda: Whether to prefer CUDA if available.

    Returns:
        A PyTorch device object (cuda or cpu).
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger for the Language Emergence Lab.

    This function creates a logger with consistent formatting suitable for
    research experiments and debugging.

    Args:
        name: The name of the logger (typically __name__).
        level: The logging level (default: INFO).

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger
