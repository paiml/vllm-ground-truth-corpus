.PHONY: all install lint fmt test coverage clean

all: lint test

install:
	uv sync

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

fmt:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest --cov=vllm_gtc --cov-report=term-missing -v

test-fast:
	uv run pytest -x -v

coverage:
	uv run pytest --cov=vllm_gtc --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
