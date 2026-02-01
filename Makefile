.PHONY: all install lint fmt typecheck test test-fast coverage clean quality

all: quality

install:
	uv sync

# Tier 1: On-save (<1s)
lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

fmt:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

# Tier 2: Pre-commit (<5s)
typecheck:
	uv run ty check src/

test-fast:
	uv run pytest -x -q

# Tier 3: Pre-push (<2min)
test:
	uv run pytest --cov=vllm_gtc --cov-report=term-missing --cov-fail-under=95 -v

coverage:
	uv run pytest --cov=vllm_gtc --cov-report=html --cov-report=term-missing --cov-fail-under=95
	@echo "Coverage report: htmlcov/index.html"

# Full quality gate (PMAT compliant)
quality: lint typecheck test
	@echo "✅ All quality gates passed"

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
