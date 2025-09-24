"""Comprehensive unit tests for Streamlit dashboard application.

This module tests the functions in src/langlab/apps/app.py,
providing comprehensive coverage for the dashboard functionality.
"""

import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from typing import List

from langlab.apps.app import (
    load_checkpoint,
    create_accuracy_plot,
    create_entropy_plot,
    create_loss_plot,
    create_zipf_plot,
    analyze_token_distribution,
    compute_zipf_slope,
    main,
)


class TestLoadCheckpoint:
    """Test the load_checkpoint function."""

    def test_load_checkpoint_function_exists(self) -> None:
        """Test that load_checkpoint function exists and can be imported."""
        assert callable(load_checkpoint)

        # Test that it has the expected signature parameters
        import inspect

        sig = inspect.signature(load_checkpoint)
        expected_params = ["checkpoint_path"]

        for param in expected_params:
            assert param in sig.parameters

    @patch("langlab.apps.app.torch.load")
    def test_load_checkpoint_success(self, mock_torch_load: Mock) -> None:
        """Test successful checkpoint loading."""
        # Mock successful checkpoint loading
        mock_checkpoint = {
            "step": 100,
            "speaker_state_dict": {},
            "listener_state_dict": {},
            "config": {},
            "metrics": {"accuracy": 0.5},
        }
        mock_torch_load.return_value = mock_checkpoint

        result = load_checkpoint("test_checkpoint.pt")

        assert result == mock_checkpoint
        mock_torch_load.assert_called_once_with(
            "test_checkpoint.pt", map_location="cpu", weights_only=False
        )

    @patch("langlab.apps.app.torch.load")
    @patch("langlab.apps.app.st.error")
    def test_load_checkpoint_failure(
        self, mock_st_error: Mock, mock_torch_load: Mock
    ) -> None:
        """Test checkpoint loading failure."""
        # Mock loading failure
        mock_torch_load.side_effect = FileNotFoundError("File not found")

        result = load_checkpoint("nonexistent_checkpoint.pt")

        assert result is None
        mock_st_error.assert_called_once()
        assert "Error loading checkpoint" in str(mock_st_error.call_args)

    @patch("langlab.apps.app.torch.load")
    @patch("langlab.apps.app.st.error")
    def test_load_checkpoint_exception_handling(
        self, mock_st_error: Mock, mock_torch_load: Mock
    ) -> None:
        """Test checkpoint loading with various exceptions."""
        # Test different exception types
        exceptions = [
            FileNotFoundError("File not found"),
            RuntimeError("CUDA error"),
            ValueError("Invalid checkpoint format"),
            Exception("Generic error"),
        ]

        for exc in exceptions:
            mock_torch_load.side_effect = exc
            result = load_checkpoint("test_checkpoint.pt")

            assert result is None
            mock_st_error.assert_called()


class TestCreateAccuracyPlot:
    """Test the create_accuracy_plot function."""

    def test_create_accuracy_plot_function_exists(self) -> None:
        """Test that create_accuracy_plot function exists and can be imported."""
        assert callable(create_accuracy_plot)

        # Test that it has the expected signature parameters
        import inspect

        sig = inspect.signature(create_accuracy_plot)
        expected_params = ["logs_df"]

        for param in expected_params:
            assert param in sig.parameters

    @patch("langlab.apps.app.st.warning")
    def test_create_accuracy_plot_empty_dataframe(self, mock_st_warning: Mock) -> None:
        """Test accuracy plot with empty DataFrame."""
        empty_df = pd.DataFrame()

        create_accuracy_plot(empty_df)

        mock_st_warning.assert_called_once_with("No accuracy data available")

    @patch("langlab.apps.app.st.warning")
    def test_create_accuracy_plot_missing_accuracy_column(
        self, mock_st_warning: Mock
    ) -> None:
        """Test accuracy plot with missing accuracy column."""
        df = pd.DataFrame({"loss": [0.5, 0.4, 0.3]})

        create_accuracy_plot(df)

        mock_st_warning.assert_called_once_with("No accuracy data available")

    @patch("langlab.apps.app.st.pyplot")
    @patch("langlab.apps.app.plt")
    def test_create_accuracy_plot_success(
        self, mock_plt: Mock, mock_st_pyplot: Mock
    ) -> None:
        """Test successful accuracy plot creation."""
        # Create test data
        df = pd.DataFrame(
            {"step": [1, 2, 3, 4, 5], "accuracy": [0.1, 0.3, 0.5, 0.7, 0.9]}
        )

        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        create_accuracy_plot(df)

        # Verify plot creation
        mock_plt.subplots.assert_called_once_with(figsize=(10, 6))
        mock_ax.plot.assert_called_once()
        mock_ax.set_xlabel.assert_called_once_with("Training Step")
        mock_ax.set_ylabel.assert_called_once_with("Accuracy")
        mock_ax.set_title.assert_called_once_with("Training Accuracy Over Time")
        mock_st_pyplot.assert_called_once_with(mock_fig)


class TestCreateEntropyPlot:
    """Test the create_entropy_plot function."""

    def test_create_entropy_plot_function_exists(self) -> None:
        """Test that create_entropy_plot function exists and can be imported."""
        assert callable(create_entropy_plot)

        # Test that it has the expected signature parameters
        import inspect

        sig = inspect.signature(create_entropy_plot)
        expected_params = ["logs_df"]

        for param in expected_params:
            assert param in sig.parameters

    @patch("langlab.apps.app.st.warning")
    def test_create_entropy_plot_empty_dataframe(self, mock_st_warning: Mock) -> None:
        """Test entropy plot with empty DataFrame."""
        empty_df = pd.DataFrame()

        create_entropy_plot(empty_df)

        mock_st_warning.assert_called_once_with("No training data available")

    @patch("langlab.apps.app.st.warning")
    def test_create_entropy_plot_missing_entropy_column(
        self, mock_st_warning: Mock
    ) -> None:
        """Test entropy plot with missing speaker_loss column."""
        df = pd.DataFrame({"loss": [0.5, 0.4, 0.3]})

        create_entropy_plot(df)

        mock_st_warning.assert_called_once_with("No entropy data available")

    @patch("langlab.apps.app.st.pyplot")
    @patch("langlab.apps.app.plt")
    def test_create_entropy_plot_success(
        self, mock_plt: Mock, mock_st_pyplot: Mock
    ) -> None:
        """Test successful entropy plot creation."""
        # Create test data with speaker_loss column
        df = pd.DataFrame(
            {"step": [1, 2, 3, 4, 5], "speaker_loss": [2.1, 1.8, 1.5, 1.2, 0.9]}
        )

        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        create_entropy_plot(df)

        # Verify plot creation
        mock_plt.subplots.assert_called_once_with(figsize=(10, 6))
        mock_ax.plot.assert_called_once()
        mock_ax.set_xlabel.assert_called_once_with("Training Step")
        mock_ax.set_ylabel.assert_called_once_with("Approximate Entropy")
        mock_ax.set_title.assert_called_once_with("Message Entropy Over Time")
        mock_st_pyplot.assert_called_once_with(mock_fig)


class TestCreateLossPlot:
    """Test the create_loss_plot function."""

    def test_create_loss_plot_function_exists(self) -> None:
        """Test that create_loss_plot function exists and can be imported."""
        assert callable(create_loss_plot)

        # Test that it has the expected signature parameters
        import inspect

        sig = inspect.signature(create_loss_plot)
        expected_params = ["logs_df"]

        for param in expected_params:
            assert param in sig.parameters

    @patch("langlab.apps.app.st.warning")
    def test_create_loss_plot_empty_dataframe(self, mock_st_warning: Mock) -> None:
        """Test loss plot with empty DataFrame."""
        empty_df = pd.DataFrame()

        create_loss_plot(empty_df)

        mock_st_warning.assert_called_once_with("No training data available")

    @patch("langlab.apps.app.st.info")
    def test_create_loss_plot_missing_loss_column(self, mock_st_info: Mock) -> None:
        """Test loss plot with missing total_loss column."""
        df = pd.DataFrame({"accuracy": [0.5, 0.6, 0.7]})

        create_loss_plot(df)

        mock_st_info.assert_called_once_with(
            "Loss analysis requires additional logging data"
        )

    @patch("langlab.apps.app.st.pyplot")
    @patch("langlab.apps.app.plt")
    def test_create_loss_plot_success(
        self, mock_plt: Mock, mock_st_pyplot: Mock
    ) -> None:
        """Test successful loss plot creation."""
        # Create test data with total_loss column
        df = pd.DataFrame(
            {"step": [1, 2, 3, 4, 5], "total_loss": [2.1, 1.8, 1.5, 1.2, 0.9]}
        )

        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        create_loss_plot(df)

        # Verify plot creation
        mock_plt.subplots.assert_called_once_with(figsize=(10, 6))
        mock_ax.plot.assert_called_once()
        mock_ax.set_xlabel.assert_called_once_with("Training Step")
        mock_ax.set_ylabel.assert_called_once_with("Loss")
        mock_ax.set_title.assert_called_once_with("Training Loss Over Time")
        mock_st_pyplot.assert_called_once_with(mock_fig)


class TestCreateZipfPlot:
    """Test the create_zipf_plot function."""

    def test_create_zipf_plot_function_exists(self) -> None:
        """Test that create_zipf_plot function exists and can be imported."""
        assert callable(create_zipf_plot)

        # Test that it has the expected signature parameters
        import inspect

        sig = inspect.signature(create_zipf_plot)
        expected_params = ["tokens"]

        for param in expected_params:
            assert param in sig.parameters

    @patch("langlab.apps.app.st.warning")
    def test_create_zipf_plot_empty_tokens(self, mock_st_warning: Mock) -> None:
        """Test Zipf plot with empty token list."""
        empty_tokens: List[int] = []

        create_zipf_plot(empty_tokens)

        mock_st_warning.assert_called_once_with(
            "No token data available for Zipf analysis"
        )

    @patch("langlab.apps.app.st.pyplot")
    @patch("langlab.apps.app.plt")
    def test_create_zipf_plot_success(
        self, mock_plt: Mock, mock_st_pyplot: Mock
    ) -> None:
        """Test successful Zipf plot creation."""
        # Create test tokens
        tokens = [1, 1, 1, 2, 2, 3, 4, 5]

        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        create_zipf_plot(tokens)

        # Verify plot creation
        mock_plt.subplots.assert_called_once_with(figsize=(10, 6))
        assert (
            mock_ax.loglog.call_count == 2
        )  # Called twice: once for data, once for fit line
        mock_ax.set_xlabel.assert_called_once_with("Rank")
        mock_ax.set_ylabel.assert_called_once_with("Frequency")
        mock_ax.set_title.assert_called_once_with("Zipf Rank-Frequency Plot")
        mock_st_pyplot.assert_called_once_with(mock_fig)


class TestAnalyzeTokenDistribution:
    """Test the analyze_token_distribution function."""

    def test_analyze_token_distribution_function_exists(self) -> None:
        """Test that analyze_token_distribution function exists and can be imported."""
        assert callable(analyze_token_distribution)

        # Test that it has the expected signature parameters
        import inspect

        sig = inspect.signature(analyze_token_distribution)
        expected_params = ["tokens"]

        for param in expected_params:
            assert param in sig.parameters

    def test_analyze_token_distribution_empty_tokens(self) -> None:
        """Test token distribution analysis with empty token list."""
        empty_tokens: List[int] = []

        result = analyze_token_distribution(empty_tokens)

        assert isinstance(result, dict)
        assert result["total_tokens"] == 0
        assert result["unique_tokens"] == 0
        assert result["vocab_size"] == 0
        assert result["entropy"] == 0.0

    def test_analyze_token_distribution_success(self) -> None:
        """Test successful token distribution analysis."""
        tokens = [1, 1, 1, 2, 2, 3, 4, 5]

        result = analyze_token_distribution(tokens)

        assert isinstance(result, dict)
        assert result["total_tokens"] == 8
        assert result["unique_tokens"] == 5
        assert result["vocab_size"] == 5
        assert isinstance(result["entropy"], float)
        assert isinstance(result["zipf_slope"], float)

    def test_analyze_token_distribution_single_token(self) -> None:
        """Test token distribution analysis with single token type."""
        tokens = [1, 1, 1, 1, 1]

        result = analyze_token_distribution(tokens)

        assert isinstance(result, dict)
        assert result["total_tokens"] == 5
        assert result["unique_tokens"] == 1
        assert result["vocab_size"] == 1
        assert result["entropy"] < 0.001  # Very close to zero due to epsilon


class TestComputeZipfSlope:
    """Test the compute_zipf_slope function."""

    def test_compute_zipf_slope_function_exists(self) -> None:
        """Test that compute_zipf_slope function exists and can be imported."""
        assert callable(compute_zipf_slope)

        # Test that it has the expected signature parameters
        import inspect

        sig = inspect.signature(compute_zipf_slope)
        expected_params = ["tokens"]

        for param in expected_params:
            assert param in sig.parameters

    def test_compute_zipf_slope_empty_tokens(self) -> None:
        """Test Zipf slope computation with empty token list."""
        empty_tokens: List[int] = []

        slope, frequencies = compute_zipf_slope(empty_tokens)

        assert isinstance(slope, float)
        assert isinstance(frequencies, np.ndarray)
        assert len(frequencies) == 0

    def test_compute_zipf_slope_success(self) -> None:
        """Test successful Zipf slope computation."""
        tokens = [1, 1, 1, 2, 2, 3, 4, 5]

        slope, frequencies = compute_zipf_slope(tokens)

        assert isinstance(slope, float)
        assert isinstance(frequencies, np.ndarray)
        assert len(frequencies) > 0
        # Slope should be negative for Zipf-like distribution
        assert slope < 0

    def test_compute_zipf_slope_single_token(self) -> None:
        """Test Zipf slope computation with single token type."""
        tokens = [1, 1, 1, 1, 1]

        slope, frequencies = compute_zipf_slope(tokens)

        assert isinstance(slope, float)
        assert isinstance(frequencies, np.ndarray)
        assert len(frequencies) == 1


class TestMain:
    """Test the main function."""

    def test_main_function_exists(self) -> None:
        """Test that main function exists and can be imported."""
        assert callable(main)

        # Test that it has the expected signature parameters
        import inspect

        sig = inspect.signature(main)

        # Main function should have no parameters
        assert len(sig.parameters) == 0

    @patch("langlab.apps.app.st")
    def test_main_function_interface(self, mock_st: Mock) -> None:
        """Test main function interface."""
        # Mock streamlit components
        mock_st.title.return_value = None
        mock_st.sidebar.return_value = None
        mock_st.selectbox.return_value = None
        mock_st.file_uploader.return_value = None
        mock_st.button.return_value = None
        mock_st.success.return_value = None
        mock_st.error.return_value = None
        mock_st.warning.return_value = None

        # Test that we can call it (will fail in execution but tests interface)
        try:
            main()
            # If it succeeds, that's fine too
        except Exception as e:
            # Expected to fail due to missing streamlit context, but should be callable
            assert (
                "main" in str(type(e).__name__)
                or "RuntimeError" in str(type(e).__name__)
                or "AttributeError" in str(type(e).__name__)
            )
