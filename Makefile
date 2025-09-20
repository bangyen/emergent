.PHONY: init fmt lint type test test-unit test-integration test-slow test-all test-coverage clean
init: ## install tooling
	python -m pip install -U pip
	pip install -e ".[dev]"
	pre-commit install

fmt:  ## format code
	black .

lint: ## lint code
	ruff check .

type: ## type-check
	mypy .

test: ## run all tests
	python -m pytest

test-unit: ## run unit tests only
	python -m pytest tests/unit/ -m "not slow"

test-integration: ## run integration tests only
	python -m pytest tests/integration/ -m "not slow"

test-slow: ## run slow tests
	python -m pytest -m "slow"

test-coverage: ## run tests with coverage report
	python -m pytest --cov=src --cov-report=html --cov-report=term-missing

test-all: fmt lint type test ## run all checks and tests

clean: ## clean up generated files
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
