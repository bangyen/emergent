"""Tests for training grid utilities.

This module tests the simple grid creation utilities used in grounded training.
"""

from langlab.training.grid import create_simple_grid
from langlab.experiments.grid import Grid


class TestCreateSimpleGrid:
    """Test the create_simple_grid function."""

    def test_create_simple_grid_default_params(self) -> None:
        """Test creating a simple grid with default parameters."""
        grid = create_simple_grid()

        assert isinstance(grid, Grid)
        assert grid.size == 5
        assert grid.max_steps == 20
        assert grid.start_pos == (0, 0)
        assert grid.target_pos == (4, 4)
        assert grid.walls == []  # Empty list, not None

    def test_create_simple_grid_custom_size(self) -> None:
        """Test creating a simple grid with custom size."""
        grid = create_simple_grid(size=10)

        assert isinstance(grid, Grid)
        assert grid.size == 10
        assert grid.start_pos == (0, 0)
        assert grid.target_pos == (9, 9)

    def test_create_simple_grid_custom_max_steps(self) -> None:
        """Test creating a simple grid with custom max steps."""
        grid = create_simple_grid(max_steps=50)

        assert isinstance(grid, Grid)
        assert grid.max_steps == 50

    def test_create_simple_grid_with_seed(self) -> None:
        """Test creating a simple grid with a random seed."""
        grid = create_simple_grid(seed=42)

        assert isinstance(grid, Grid)
        # Note: Grid doesn't store the seed as an attribute

    def test_create_simple_grid_all_params(self) -> None:
        """Test creating a simple grid with all parameters specified."""
        grid = create_simple_grid(size=8, max_steps=30, seed=123)

        assert isinstance(grid, Grid)
        assert grid.size == 8
        assert grid.max_steps == 30
        assert grid.start_pos == (0, 0)
        assert grid.target_pos == (7, 7)
        assert grid.walls == []  # Empty list, not None

    def test_create_simple_grid_positioning(self) -> None:
        """Test that start and target positions are correctly set."""
        for size in [3, 5, 7, 10]:
            grid = create_simple_grid(size=size)
            assert grid.start_pos == (0, 0)
            assert grid.target_pos == (size - 1, size - 1)

    def test_create_simple_grid_no_walls(self) -> None:
        """Test that simple grids have no walls."""
        grid = create_simple_grid(size=6)
        assert grid.walls == []  # Empty list, not None

    def test_create_simple_grid_reproducibility(self) -> None:
        """Test that grids with the same seed are reproducible."""
        grid1 = create_simple_grid(size=5, seed=42)
        grid2 = create_simple_grid(size=5, seed=42)

        # Both grids should have identical properties
        assert grid1.size == grid2.size
        assert grid1.max_steps == grid2.max_steps
        assert grid1.start_pos == grid2.start_pos
        assert grid1.target_pos == grid2.target_pos
        assert grid1.walls == grid2.walls
