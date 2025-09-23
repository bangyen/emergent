"""Smoke tests for the Streamlit dashboard application.

This module contains basic smoke tests to ensure the dashboard
application can be imported and basic functions work correctly.
"""

import pytest
from unittest.mock import patch
from typing import Any


class TestAppImports:
    """Test cases for importing the dashboard application."""

    def test_app_imports(self) -> None:
        """Test that importing the app module does not raise exceptions."""
        try:
            from src.langlab.apps.app import main

            assert callable(main)
        except ImportError as e:
            pytest.fail(f"Failed to import app module: {e}")
        except Exception as e:
            pytest.fail(f"Unexpected error importing app module: {e}")

    def test_app_exposes_main_function(self) -> None:
        """Test that the app module exposes a main() function."""
        from src.langlab.apps.app import main

        assert callable(main)
        assert main.__name__ == "main"

    def test_app_imports_dependencies(self) -> None:
        """Test that the app module can import all its dependencies."""
        import importlib.util

        # Test external dependencies
        dependencies = [
            "streamlit",
            "pandas",
            "numpy",
            "matplotlib.pyplot",
            "seaborn",
            "torch",
        ]

        for dep in dependencies:
            spec = importlib.util.find_spec(dep)
            assert spec is not None, f"Dependency {dep} not found"

        # Test internal modules can be imported
        modules_to_test = [
            "src.langlab.analysis",
            "src.langlab.data.world",
            "src.langlab.core.agents",
            "src.langlab.core.config",
            "src.langlab.utils",
        ]

        for module_name in modules_to_test:
            spec = importlib.util.find_spec(module_name)
            assert spec is not None, f"Module {module_name} not found"


class TestAppFunctions:
    """Test cases for dashboard application functions."""

    def test_load_checkpoint_function_exists(self) -> None:
        """Test that the load_checkpoint function exists and is callable."""
        from src.langlab.apps.app import load_checkpoint

        assert callable(load_checkpoint)

    def test_load_checkpoint_with_nonexistent_file(self) -> None:
        """Test load_checkpoint with nonexistent file."""
        from src.langlab.apps.app import load_checkpoint

        result = load_checkpoint("nonexistent_file.pt")
        assert result is None

    @patch("torch.load")
    def test_load_checkpoint_with_mock_data(self, mock_torch_load: Any) -> None:
        """Test load_checkpoint with mock torch data."""
        from src.langlab.apps.app import load_checkpoint

        # Mock successful checkpoint loading
        mock_checkpoint = {
            "step": 1000,
            "speaker_state_dict": {},
            "listener_state_dict": {},
        }
        mock_torch_load.return_value = mock_checkpoint

        result = load_checkpoint("test_checkpoint.pt")

        assert result is not None
        assert result["step"] == 1000
        mock_torch_load.assert_called_once()

    def test_create_accuracy_plot_function_exists(self) -> None:
        """Test that create_accuracy_plot function exists."""
        from src.langlab.apps.app import create_accuracy_plot

        assert callable(create_accuracy_plot)

    def test_create_entropy_plot_function_exists(self) -> None:
        """Test that create_entropy_plot function exists."""
        from src.langlab.apps.app import create_entropy_plot

        assert callable(create_entropy_plot)

    def test_create_zipf_plot_function_exists(self) -> None:
        """Test that create_zipf_plot function exists."""
        from src.langlab.apps.app import create_zipf_plot

        assert callable(create_zipf_plot)

    def test_analyze_token_distribution_function_exists(self) -> None:
        """Test that analyze_token_distribution function exists."""
        from src.langlab.apps.app import analyze_token_distribution

        assert callable(analyze_token_distribution)

    def test_compute_zipf_slope_function_exists(self) -> None:
        """Test that compute_zipf_slope function exists."""
        from src.langlab.apps.app import compute_zipf_slope

        assert callable(compute_zipf_slope)


class TestAppIntegration:
    """Integration tests for the dashboard application."""

    def test_main_function_signature(self) -> None:
        """Test that main function has correct signature."""
        from src.langlab.apps.app import main

        # Check that main can be called without arguments
        # (We can't actually run it in tests, but we can check the signature)
        import inspect

        sig = inspect.signature(main)
        assert len(sig.parameters) == 0  # main() takes no arguments

    def test_app_module_structure(self) -> None:
        """Test that the app module has expected structure."""
        import src.langlab.apps.app as app_module

        # Check that expected functions exist
        expected_functions = [
            "main",
            "load_checkpoint",
            "create_accuracy_plot",
            "create_entropy_plot",
            "create_zipf_plot",
            "analyze_token_distribution",
            "compute_zipf_slope",
        ]

        for func_name in expected_functions:
            assert hasattr(app_module, func_name), f"Missing function: {func_name}"
            assert callable(
                getattr(app_module, func_name)
            ), f"Not callable: {func_name}"

    def test_app_can_handle_empty_data(self) -> None:
        """Test that app functions can handle empty data gracefully."""
        import pandas as pd
        from src.langlab.apps.app import (
            create_accuracy_plot,
            create_entropy_plot,
            create_zipf_plot,
        )

        empty_df = pd.DataFrame()

        # These should not raise exceptions with empty data
        try:
            create_accuracy_plot(empty_df)
            create_entropy_plot(empty_df)
            create_zipf_plot([])  # Empty token list
        except Exception as e:
            pytest.fail(f"Functions should handle empty data gracefully: {e}")

    def test_app_can_handle_missing_columns(self) -> None:
        """Test that app functions can handle DataFrames with missing columns."""
        import pandas as pd
        from src.langlab.apps.app import (
            create_accuracy_plot,
            create_entropy_plot,
        )

        # DataFrame with wrong columns
        wrong_df = pd.DataFrame(
            {"wrong_column": [1, 2, 3], "another_wrong_column": [4, 5, 6]}
        )

        # These should not raise exceptions with wrong columns
        try:
            create_accuracy_plot(wrong_df)
            create_entropy_plot(wrong_df)
        except Exception as e:
            pytest.fail(f"Functions should handle missing columns gracefully: {e}")


class TestAppDependencies:
    """Test cases for dashboard application dependencies."""

    def test_streamlit_available(self) -> None:
        """Test that Streamlit is available."""
        try:
            import streamlit as st

            assert hasattr(st, "set_page_config")
            assert hasattr(st, "title")
            assert hasattr(st, "plotly_chart")
        except ImportError:
            pytest.fail("Streamlit is not available")

    def test_matplotlib_available(self) -> None:
        """Test that matplotlib is available."""
        try:
            import matplotlib.pyplot as plt

            assert hasattr(plt, "figure")
            assert hasattr(plt, "plot")
            assert hasattr(plt, "show")
        except ImportError:
            pytest.fail("Matplotlib is not available")

    def test_pandas_available(self) -> None:
        """Test that pandas is available."""
        try:
            import pandas as pd

            assert hasattr(pd, "DataFrame")
            assert hasattr(pd, "read_csv")
        except ImportError:
            pytest.fail("Pandas is not available")

    def test_numpy_available(self) -> None:
        """Test that numpy is available."""
        try:
            import numpy as np

            assert hasattr(np, "array")
            assert hasattr(np, "random")
        except ImportError:
            pytest.fail("Numpy is not available")
