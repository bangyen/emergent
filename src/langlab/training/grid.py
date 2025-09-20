"""Simple grid creation utilities for grounded training.

This module provides utilities for creating simple grid environments
for grounded language learning experiments.
"""

from typing import Optional
from ..experiments.grid import Grid


def create_simple_grid(
    size: int = 5,
    max_steps: int = 20,
    seed: Optional[int] = None,
) -> Grid:
    """Create a simple grid environment for grounded training.

    Args:
        size: Size of the square grid.
        max_steps: Maximum steps per episode.
        seed: Random seed for reproducible generation.

    Returns:
        Grid environment instance.
    """
    return Grid(
        size=size,
        walls=None,  # No walls for simple grid
        start_pos=(0, 0),  # Start at top-left
        target_pos=(size - 1, size - 1),  # Target at bottom-right
        max_steps=max_steps,
        seed=seed,
    )
