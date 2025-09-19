# Python Package Template

A minimal, modern Python package template with development tools pre-configured.

## Features

- **Modern tooling**: Black, Ruff, MyPy, Pytest
- **Pre-commit hooks**: Automated code quality checks
- **Type hints**: Full type checking with MyPy
- **Testing**: Pytest with fixtures and coverage
- **Package structure**: Standard src/ layout

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd your-package-name
   make init
   ```

2. **Development workflow**:
   ```bash
   make fmt    # Format code
   make lint   # Lint code
   make type   # Type check
   make test   # Run tests
   make all    # Run all checks
   ```

## Project Structure

```
├── src/           # Source code
├── tests/         # Test files
├── scripts/       # Utility scripts
├── Makefile       # Development commands
├── pyproject.toml # Package configuration
└── README.md      # This file
```

## Customization

1. **Update package info** in `pyproject.toml`:
   - Change `name`, `description`, `authors`
   - Update repository URLs

2. **Add your code** to `src/`

3. **Write tests** in `tests/`

## License

MIT License - see LICENSE file for details.
