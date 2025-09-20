"""Tests for the analysis module.

This module contains unit tests for language analysis functions,
including Zipf's law analysis and token distribution statistics.
"""

import numpy as np

from langlab.analysis import (
    compute_zipf_slope,
    analyze_token_distribution,
    load_training_logs,
    compute_compositional_vs_iid_accuracy,
)


class TestZipfAnalysis:
    """Test cases for Zipf's law analysis functions."""

    def test_zipf_outputs_empty_tokens(self) -> None:
        """Test that zipf analysis returns correct outputs for empty token list."""
        slope, frequencies = compute_zipf_slope([])

        assert isinstance(slope, float)
        assert slope == 0.0
        assert isinstance(frequencies, np.ndarray)
        assert len(frequencies) == 0

    def test_zipf_outputs_single_token(self) -> None:
        """Test that zipf analysis handles single token correctly."""
        slope, frequencies = compute_zipf_slope([1])

        assert isinstance(slope, float)
        assert slope == 0.0  # Cannot fit line with single point
        assert isinstance(frequencies, np.ndarray)
        assert len(frequencies) == 1
        assert frequencies[0] == 1

    def test_zipf_outputs_perfect_zipf(self) -> None:
        """Test that zipf analysis correctly identifies perfect Zipf distribution."""
        # Create perfect Zipf distribution: frequency = 1/rank
        tokens = []
        for rank in range(1, 11):  # ranks 1-10
            frequency = 100 // rank  # approximate 1/rank
            tokens.extend([rank] * frequency)

        slope, frequencies = compute_zipf_slope(tokens)

        assert isinstance(slope, float)
        assert isinstance(frequencies, np.ndarray)
        assert len(frequencies) > 0

        # For perfect Zipf, slope should be close to -1
        assert slope < 0  # Negative slope
        assert abs(slope + 1.0) < 0.5  # Should be close to -1

    def test_zipf_outputs_random_tokens(self) -> None:
        """Test that zipf analysis returns valid outputs for random tokens."""
        np.random.seed(42)
        tokens = np.random.randint(0, 10, size=100).tolist()

        slope, frequencies = compute_zipf_slope(tokens)

        assert isinstance(slope, float)
        assert isinstance(frequencies, np.ndarray)
        assert len(frequencies) > 0
        assert all(f > 0 for f in frequencies)  # All frequencies should be positive

    def test_zipf_outputs_uniform_distribution(self) -> None:
        """Test that zipf analysis handles uniform distribution correctly."""
        # Create uniform distribution (all tokens equally frequent)
        tokens = list(range(10)) * 5  # 10 different tokens, each appears 5 times

        slope, frequencies = compute_zipf_slope(tokens)

        assert isinstance(slope, float)
        assert isinstance(frequencies, np.ndarray)
        assert len(frequencies) == 10
        assert all(f == 5 for f in frequencies)  # All frequencies should be 5


class TestTokenDistributionAnalysis:
    """Test cases for token distribution analysis functions."""

    def test_analyze_token_distribution_empty(self) -> None:
        """Test token distribution analysis with empty input."""
        result = analyze_token_distribution([])

        assert isinstance(result, dict)
        assert result["zipf_slope"] == 0.0
        assert result["vocab_size"] == 0
        assert result["total_tokens"] == 0
        assert result["unique_tokens"] == 0
        assert isinstance(result["frequencies"], np.ndarray)
        assert len(result["frequencies"]) == 0
        assert result["entropy"] == 0.0
        assert result["gini_coefficient"] == 0.0

    def test_analyze_token_distribution_single_token(self) -> None:
        """Test token distribution analysis with single token."""
        result = analyze_token_distribution([1])

        assert isinstance(result, dict)
        assert result["zipf_slope"] == 0.0
        assert result["vocab_size"] == 1
        assert result["total_tokens"] == 1
        assert result["unique_tokens"] == 1
        assert isinstance(result["frequencies"], np.ndarray)
        assert len(result["frequencies"]) == 1
        assert result["frequencies"][0] == 1
        assert (
            result["entropy"] < 1e-9
        )  # Single token has near-zero entropy (accounting for numerical precision)
        assert result["gini_coefficient"] == 0.0  # Single token has zero inequality

    def test_analyze_token_distribution_multiple_tokens(self) -> None:
        """Test token distribution analysis with multiple tokens."""
        tokens = [1, 1, 2, 2, 2, 3]
        result = analyze_token_distribution(tokens)

        assert isinstance(result, dict)
        assert isinstance(result["zipf_slope"], float)
        assert result["vocab_size"] == 3
        assert result["total_tokens"] == 6
        assert result["unique_tokens"] == 3
        assert isinstance(result["frequencies"], np.ndarray)
        assert len(result["frequencies"]) == 3
        assert result["entropy"] > 0  # Multiple tokens should have positive entropy
        assert (
            0 <= result["gini_coefficient"] <= 1
        )  # Gini coefficient should be in [0,1]

    def test_analyze_token_distribution_perfect_zipf(self) -> None:
        """Test token distribution analysis with perfect Zipf distribution."""
        # Create perfect Zipf distribution
        tokens = []
        for rank in range(1, 6):  # ranks 1-5
            frequency = 20 // rank  # approximate 1/rank
            tokens.extend([rank] * frequency)

        result = analyze_token_distribution(tokens)

        assert isinstance(result, dict)
        assert isinstance(result["zipf_slope"], float)
        assert result["zipf_slope"] < 0  # Should be negative for Zipf
        assert result["vocab_size"] == 5
        assert result["total_tokens"] == len(tokens)
        assert result["unique_tokens"] == 5
        assert isinstance(result["frequencies"], np.ndarray)
        assert len(result["frequencies"]) == 5
        assert result["entropy"] > 0
        assert 0 <= result["gini_coefficient"] <= 1


class TestTrainingLogsAnalysis:
    """Test cases for training logs analysis functions."""

    def test_load_training_logs_nonexistent_file(self) -> None:
        """Test loading training logs from nonexistent file."""
        import pandas as pd

        result = load_training_logs("nonexistent_file.csv")

        assert isinstance(result, pd.DataFrame)  # Should return empty DataFrame
        # Function should handle missing files gracefully

    def test_compute_compositional_vs_iid_accuracy_empty(self) -> None:
        """Test compositional vs IID accuracy computation with empty data."""
        import pandas as pd

        empty_df = pd.DataFrame()

        result = compute_compositional_vs_iid_accuracy(empty_df)

        assert isinstance(result, dict)
        assert "iid_accuracy" in result
        assert "compositional_accuracy" in result
        assert result["iid_accuracy"] == 0.0
        assert result["compositional_accuracy"] == 0.0

    def test_compute_compositional_vs_iid_accuracy_with_data(self) -> None:
        """Test compositional vs IID accuracy computation with sample data."""
        import pandas as pd

        # Create sample training data
        data = {"step": [1, 2, 3, 4, 5], "accuracy": [0.1, 0.2, 0.3, 0.4, 0.5]}
        df = pd.DataFrame(data)

        result = compute_compositional_vs_iid_accuracy(df)

        assert isinstance(result, dict)
        assert "iid_accuracy" in result
        assert "compositional_accuracy" in result
        assert result["iid_accuracy"] == 0.5  # Final accuracy
        assert result["compositional_accuracy"] == 0.4  # 80% of final accuracy
        assert isinstance(result["iid_accuracy"], float)
        assert isinstance(result["compositional_accuracy"], float)


class TestAnalysisIntegration:
    """Integration tests for analysis functions."""

    def test_zipf_analysis_integration(self) -> None:
        """Test that zipf analysis integrates correctly with token distribution analysis."""
        # Create sample tokens
        np.random.seed(42)
        tokens = np.random.randint(0, 10, size=1000).tolist()

        # Test both functions
        slope, frequencies = compute_zipf_slope(tokens)
        analysis = analyze_token_distribution(tokens)

        # Results should be consistent
        assert abs(slope - analysis["zipf_slope"]) < 1e-10
        assert np.array_equal(frequencies, analysis["frequencies"])

    def test_analysis_with_realistic_data(self) -> None:
        """Test analysis functions with realistic token data."""
        # Create realistic token distribution (some tokens more frequent)
        tokens = []
        # Token 0 appears 100 times
        tokens.extend([0] * 100)
        # Token 1 appears 50 times
        tokens.extend([1] * 50)
        # Tokens 2-9 appear 10 times each
        for i in range(2, 10):
            tokens.extend([i] * 10)

        # Shuffle to make it more realistic
        np.random.seed(42)
        np.random.shuffle(tokens)

        # Analyze
        slope, frequencies = compute_zipf_slope(tokens)
        analysis = analyze_token_distribution(tokens)

        # Check results
        assert isinstance(slope, float)
        assert isinstance(frequencies, np.ndarray)
        assert len(frequencies) == 10
        assert analysis["vocab_size"] == 10
        assert analysis["total_tokens"] == len(tokens)
        assert analysis["entropy"] > 0
        assert 0 <= analysis["gini_coefficient"] <= 1
