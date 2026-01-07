.PHONY: help install install-dev install-ml lint format test test-cov smoke smoke-ml

help:
	@echo "Targets:"
	@echo "  install      - pip install -e ."
	@echo "  install-dev  - pip install -e '.[dev]'"
	@echo "  install-ml   - pip install -e '.[ml]'"
	@echo "  lint         - ruff check apps src tests"
	@echo "  format       - ruff format apps src tests"
	@echo "  test         - pytest"
	@echo "  test-cov     - pytest with coverage"
	@echo "  smoke        - lightweight import smoke"
	@echo "  smoke-ml     - import smoke with ML extras"

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e '.[dev]'

install-ml:
	python -m pip install -e '.[ml]'

lint:
	ruff check apps src tests

format:
	ruff format apps src tests

test:
	pytest

test-cov:
	pytest --cov=src/news_structurizer --cov-report=term-missing

smoke:
	python scripts/smoke_imports.py

smoke-ml:
	python scripts/smoke_imports_ml.py
