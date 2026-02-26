"""Tests for the analysis module.

This module contains unit tests for language analysis functions,
focusing on training log loading and processing.
"""

from langlab.analysis.analysis import load_training_logs


class TestTrainingLogsAnalysis:
    """Test cases for training logs analysis functions."""

    def test_load_training_logs_nonexistent_file(self) -> None:
        """Test loading training logs from nonexistent file."""
        import pandas as pd

        result = load_training_logs("nonexistent_file.csv")

        assert isinstance(result, pd.DataFrame)  # Should return empty DataFrame
        assert len(result) == 0  # Should be empty
        # Function should handle missing files gracefully

    def test_load_training_logs_with_valid_data(self) -> None:
        """Test loading training logs with valid CSV data."""
        import pandas as pd
        import tempfile
        import os

        # Create a temporary CSV file with sample data
        sample_data = """step,total_loss,listener_loss,speaker_loss,accuracy,baseline
0,1.5,0.8,0.7,0.3,0.5
1,1.2,0.6,0.6,0.5,0.6
2,1.0,0.4,0.6,0.7,0.7
3,0.8,0.3,0.5,0.8,0.8
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_data)
            temp_file_path = f.name

        try:
            result = load_training_logs(temp_file_path)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 4  # 4 rows of data
            assert "step" in result.columns
            assert "total_loss" in result.columns
            assert "accuracy" in result.columns

            # Check that numeric conversion worked
            assert result["step"].dtype in [int, float]
            assert result["total_loss"].dtype == float
            assert result["accuracy"].dtype == float

            # Check specific values
            assert result.iloc[0]["step"] == 0
            assert result.iloc[0]["accuracy"] == 0.3
            assert result.iloc[-1]["accuracy"] == 0.8

        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    def test_load_training_logs_with_missing_columns(self) -> None:
        """Test loading training logs with some missing columns."""
        import pandas as pd
        import tempfile
        import os

        # Create a CSV with only some of the expected columns
        sample_data = """step,accuracy
0,0.3
1,0.5
2,0.7
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_data)
            temp_file_path = f.name

        try:
            result = load_training_logs(temp_file_path)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert "step" in result.columns
            assert "accuracy" in result.columns
            assert result["step"].dtype in [int, float]
            assert result["accuracy"].dtype == float

        finally:
            os.unlink(temp_file_path)

    def test_load_training_logs_with_episode_column(self) -> None:
        """Test loading training logs with episode column (grid training)."""
        import pandas as pd
        import tempfile
        import os

        # Create a CSV with episode column instead of step
        sample_data = """episode,total_loss,accuracy
0,1.5,0.3
1,1.2,0.5
2,1.0,0.7
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_data)
            temp_file_path = f.name

        try:
            result = load_training_logs(temp_file_path)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert "episode" in result.columns
            assert "accuracy" in result.columns
            assert result["episode"].dtype in [int, float]
            assert result["accuracy"].dtype == float

        finally:
            os.unlink(temp_file_path)

    def test_load_training_logs_with_invalid_numeric_data(self) -> None:
        """Test loading training logs with invalid numeric data."""
        import pandas as pd
        import tempfile
        import os

        # Create a CSV with some invalid numeric values
        sample_data = """step,total_loss,accuracy
0,1.5,0.3
1,invalid,0.5
2,1.0,not_a_number
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_data)
            temp_file_path = f.name

        try:
            result = load_training_logs(temp_file_path)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3

            # Invalid values should be converted to NaN
            assert pd.isna(result.iloc[1]["total_loss"])
            assert pd.isna(result.iloc[2]["accuracy"])
            # Valid values should remain
            assert result.iloc[0]["total_loss"] == 1.5
            assert result.iloc[0]["accuracy"] == 0.3

        finally:
            os.unlink(temp_file_path)
