# Repository Guidelines

## Project Structure & Module Organization

- `src/news_structurizer/`: installable Python package (core pipeline, CLI, utils).
- `apps/recorder/`: FastAPI service that creates recording jobs and publishes to RabbitMQ.
- `apps/worker/`: consumer that processes recordings via `news_structurizer`.
- `apps/demo_streamlit/`: Streamlit UI for manual pipeline checks.
- `tests/`: `pytest` tests (`tests/test_*.py`).
- `scripts/`: local smoke checks and runnable examples (e.g. `run_pipeline_text.py`).
- `deploy/`: Docker Compose deployment (`deploy/docker-compose.yml`).
- `models/`: local model folders (mounted as `/models`; do not commit).
- `research/`: experiments/training artifacts (not part of the product runtime).

## Build, Test, and Development Commands

Prefer `make` targets (see `Makefile`):
- `make install` / `make install-dev` / `make install-ml`: editable install with optional extras.
- `make lint`: `ruff` lint for `apps/`, `src/`, `tests/`.
- `make format`: auto-format via `ruff format`.
- `make test` / `make test-cov`: run `pytest` (with coverage in `test-cov`).
- `make smoke` / `make smoke-ml`: lightweight import checks.

Local services:
- `docker compose -f deploy/docker-compose.yml up --build`

## Coding Style & Naming Conventions

- Python `>=3.11` (`pyproject.toml`), `src/`-layout.
- Indentation: 4 spaces; LF; trim trailing whitespace (see `.editorconfig`).
- Formatting/linting: `ruff` (line length 100; lint codes `E`, `F` via `ruff.toml`).
- Install hooks: `pre-commit install` (runs `ruff` + whitespace/yaml checks).

## Testing Guidelines

- Framework: `pytest` (`[tool.pytest.ini_options]` in `pyproject.toml`).
- Naming: put tests in `tests/` as `test_*.py`; keep fixtures/helpers in `tests/utils.py`.
- Match CI locally: `python -m compileall -q apps src tests && make lint && make test`.

## Commit & Pull Request Guidelines

- Prefer Conventional-Commit style used in history: `feat: ...`, `fix: ...`, `refactor: ...`, `docs: ...`, `chore: ...`.
- PRs: include a short description, how to run/verify (commands), and link related issues; for `apps/demo_streamlit/` changes, add screenshots or a short GIF.

## Configuration Tips

- Model and runtime config are env-driven (examples: `NS_*` in `deploy/docker-compose.yml`).
- Avoid network downloads in containers by providing local models (e.g. mount `models/` and set `NS_ASR_MODEL=/models/asr`).
