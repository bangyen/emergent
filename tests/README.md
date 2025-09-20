# 🧪 Testing Guide for Language Emergence Lab

This document provides a comprehensive guide to the testing infrastructure and organization in the Language Emergence Lab project.

## 📁 Test Organization

The test suite is organized into three main categories:

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── pytest.ini              # Pytest configuration
├── unit/                    # Unit tests for individual components
│   ├── test_agents.py       # Agent architecture tests
│   ├── test_world.py        # World generation tests
│   ├── test_data.py         # Dataset tests
│   ├── test_config.py       # Configuration tests
│   ├── test_channel.py      # Communication channel tests
│   ├── test_utils.py        # Utility function tests
│   ├── test_grid.py         # Grid world tests
│   ├── test_grounding.py    # Grounded learning tests
│   ├── test_analysis.py     # Analysis function tests
│   ├── test_pragmatics.py   # Pragmatic inference tests
│   ├── test_multimodal.py   # Multimodal communication tests
│   ├── test_sequences.py    # Sequence model tests
│   ├── test_splits.py       # Data splitting tests
│   ├── test_entropy_bonus.py # Entropy bonus tests
│   ├── test_loss_shapes.py  # Loss function tests
│   ├── test_heatmap_io.py   # Heatmap I/O tests
│   └── test_vocab_stats.py  # Vocabulary statistics tests
├── integration/             # Integration tests for workflows
│   ├── test_cli.py          # CLI integration tests
│   ├── test_training.py     # Training workflow tests
│   ├── test_e2e.py          # End-to-end workflow tests
│   ├── test_app_smoke.py    # Dashboard smoke tests
│   ├── test_train_smoke.py  # Training smoke tests
│   ├── test_population.py   # Population dynamics tests
│   ├── test_contact.py      # Contact experiment tests
│   ├── test_eval.py         # Evaluation workflow tests
│   ├── test_ablate.py       # Ablation study tests
│   ├── test_report.py       # Report generation tests
│   └── test_reproducibility.py # Reproducibility tests
└── fixtures/                # Test fixtures and utilities
    └── (future test data files)
```

## 🏷️ Test Markers

Tests are organized using pytest markers:

- **`@pytest.mark.unit`**: Unit tests for individual components
- **`@pytest.mark.integration`**: Integration tests for workflows
- **`@pytest.mark.slow`**: Slow-running tests (training, large experiments)
- **`@pytest.mark.gpu`**: Tests requiring GPU (future)
- **`@pytest.mark.smoke`**: Smoke tests for basic functionality

## 🚀 Running Tests

### Basic Commands

```bash
# Run all tests
make test

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration

# Run slow tests
make test-slow

# Run tests with coverage
make test-coverage

# Run all checks (format, lint, type, test)
make test-all
```

### Advanced Commands

```bash
# Run specific test file
python -m pytest tests/unit/test_agents.py

# Run tests with specific marker
python -m pytest -m "unit and not slow"

# Run tests with verbose output
python -m pytest -v

# Run tests in parallel (if pytest-xdist installed)
python -m pytest -n auto

# Run tests and stop on first failure
python -m pytest -x

# Run tests matching pattern
python -m pytest -k "test_speaker"
```

## 🔧 Test Configuration

### Coverage Requirements

- **Minimum coverage**: 80%
- **Coverage reports**: HTML, XML, and terminal output
- **Coverage exclusions**: Test files, `__init__.py` files

### Test Data and Fixtures

The `conftest.py` file provides comprehensive fixtures:

- **`sample_config`**: Standard communication configuration
- **`large_config`**: Larger configuration for integration tests
- **`sample_object`**: Sample object for testing
- **`sample_scene_data`**: Sample scene data
- **`sample_speaker/listener`**: Pre-configured agents
- **`sample_dataset`**: Small dataset for testing
- **`mock_checkpoint`**: Mock checkpoint data
- **`sample_training_logs`**: Sample training metrics
- **`temp_output_dir`**: Temporary output directory

## 📊 Test Categories

### Unit Tests

Unit tests focus on individual components in isolation:

- **Agent Architecture**: Speaker, Listener, sequence models
- **World Generation**: Object creation, scene sampling, encoding
- **Data Processing**: Dataset creation, loading, splitting
- **Analysis Functions**: Zipf analysis, compositional metrics
- **Utility Functions**: Device detection, seeding, logging

### Integration Tests

Integration tests verify component interactions:

- **CLI Integration**: Command-line interface workflows
- **Training Workflows**: Complete training pipelines
- **Population Dynamics**: Cultural transmission experiments
- **Contact Experiments**: Cross-population interactions
- **Evaluation Workflows**: Model evaluation and analysis
- **End-to-End Workflows**: Complete experiment pipelines

### Smoke Tests

Smoke tests verify basic functionality:

- **Import Tests**: All modules can be imported
- **Basic Functionality**: Core functions work without errors
- **Configuration**: Default configurations are valid
- **Data Generation**: Basic data generation works

## 🎯 Test Quality Guidelines

### Test Design Principles

1. **Isolation**: Tests should be independent and not rely on each other
2. **Deterministic**: Tests should produce consistent results
3. **Fast**: Unit tests should run quickly (< 1 second each)
4. **Clear**: Test names and assertions should be self-documenting
5. **Comprehensive**: Cover both happy path and edge cases

### Mocking Strategy

- **External Dependencies**: Mock file I/O, network calls, subprocess calls
- **Slow Operations**: Mock training steps, large computations
- **Random Operations**: Use fixed seeds for reproducibility
- **System Dependencies**: Mock device detection, environment variables

### Test Data Management

- **Fixtures**: Use pytest fixtures for reusable test data
- **Temporary Files**: Use `tmp_path` for temporary file operations
- **Mock Data**: Use realistic but minimal test data
- **Cleanup**: Ensure tests clean up after themselves

## 🔍 Debugging Tests

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Path Issues**: Use absolute paths or proper relative paths
3. **Mock Issues**: Verify mock setup and teardown
4. **Seed Issues**: Use consistent seeds for reproducibility

### Debugging Commands

```bash
# Run single test with debug output
python -m pytest tests/unit/test_agents.py::TestSpeaker::test_forward -v -s

# Run tests with print statements
python -m pytest tests/unit/test_agents.py -s

# Run tests with pdb debugger
python -m pytest tests/unit/test_agents.py --pdb

# Run tests with coverage debugging
python -m pytest --cov=src --cov-report=term-missing --cov-report=html
```

## 📈 Coverage Analysis

### Coverage Reports

- **Terminal**: `--cov-report=term-missing`
- **HTML**: `--cov-report=html:htmlcov`
- **XML**: `--cov-report=xml`

### Coverage Targets

- **Overall**: ≥ 80%
- **Critical Modules**: ≥ 90% (agents, training, evaluation)
- **Utility Modules**: ≥ 70% (acceptable for simple utilities)

### Coverage Exclusions

```python
# In pyproject.toml
[tool.coverage.run]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__init__.py",
    "*/conftest.py"
]
```

## 🚀 Continuous Integration

### GitHub Actions (Future)

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: make init
      - name: Run tests
        run: make test-coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## 📝 Writing New Tests

### Test Template

```python
"""Tests for module_name.

Brief description of what this module tests.
"""

import pytest
import torch
from unittest.mock import patch, Mock

from langlab.module_name import function_name


@pytest.mark.unit
class TestFunctionName:
    """Test cases for function_name."""

    def test_basic_functionality(self, sample_config):
        """Test basic functionality of function_name."""
        # Arrange
        input_data = create_test_data()
        
        # Act
        result = function_name(input_data, sample_config)
        
        # Assert
        assert result is not None
        assert result.shape == expected_shape

    def test_edge_case(self, sample_config):
        """Test edge case handling."""
        with pytest.raises(ValueError, match="Expected error message"):
            function_name(invalid_input, sample_config)

    @patch('langlab.module_name.external_function')
    def test_with_mock(self, mock_external, sample_config):
        """Test with mocked external dependency."""
        mock_external.return_value = expected_value
        
        result = function_name(test_input, sample_config)
        
        assert result == expected_result
        mock_external.assert_called_once()
```

### Integration Test Template

```python
@pytest.mark.integration
@pytest.mark.slow
class TestWorkflowName:
    """Test complete workflow_name workflow."""

    def test_workflow_step_by_step(self, sample_config, temp_output_dir):
        """Test workflow step by step."""
        # Step 1: Setup
        setup_data = create_setup_data()
        
        # Step 2: Execute
        with patch('langlab.module.slow_function') as mock_slow:
            mock_slow.return_value = expected_result
            
            result = execute_workflow(setup_data, sample_config)
        
        # Step 3: Verify
        assert result.success
        assert result.metrics.accuracy > 0.8
        mock_slow.assert_called_once()
```

## 🎉 Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Test Structure**: Follow Arrange-Act-Assert pattern
3. **Test Isolation**: Each test should be independent
4. **Mock Appropriately**: Mock external dependencies, not internal logic
5. **Use Fixtures**: Leverage pytest fixtures for common setup
6. **Test Edge Cases**: Include tests for error conditions and edge cases
7. **Keep Tests Fast**: Unit tests should run quickly
8. **Document Tests**: Add docstrings explaining test purpose
9. **Regular Maintenance**: Update tests when code changes
10. **Coverage Monitoring**: Monitor coverage trends over time

This testing infrastructure ensures the Language Emergence Lab project maintains high quality and reliability while supporting rapid development and experimentation.
