# Repository Guidelines

## Project Structure & Module Organization
This repository is currently a scaffold with only a `.gitignore`. As you add code, keep a clear, conventional layout. Suggested structure for a Python-first repo:

- `src/`: application or library source code
- `tests/`: unit/integration tests (mirror `src/` packages)
- `scripts/`: runnable utilities and one-off tooling
- `docs/`: architecture notes and user documentation
- `assets/`: static files (fixtures, images, sample data)

## Build, Test, and Development Commands
No build or test commands are committed yet. When you add tooling, document it here and in `README.md`. Common starting points:

- `python -m venv .venv` + `source .venv/bin/activate`: create/activate a virtualenv
- `pip install -r requirements.txt`: install dependencies
- `pytest`: run tests (if you add pytest)

## Coding Style & Naming Conventions
No formatting or linting config is present. If you introduce Python code:

- Indentation: 4 spaces; UTF-8; keep lines <= 100 chars.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Prefer adding `ruff` and `black` configs in `pyproject.toml` once code exists.

## Testing Guidelines
Testing framework is not set yet. If you use pytest:

- Place tests in `tests/` with names like `test_<module>.py`.
- Name test functions `test_<behavior>()`.
- Keep unit tests fast and deterministic; isolate external I/O behind mocks.

## Commit & Pull Request Guidelines
Git history shows only `Initial commit`, so no conventions are established.

- Suggested commit style: `type(scope): summary` (e.g., `feat(api): add health check`).
- PRs should include: purpose, summary of changes, how to test, and any relevant screenshots/logs.
- The `dogit` skill can be invoked by the user to commit and push changes quickly.

## Security & Configuration Tips
- Keep secrets out of the repo; use `.env` for local settings (already ignored by `.gitignore`).
- Add a `README.md` and `LICENSE` once the project scope is defined.

# ExecPlans
 
When writing complex features or significant refactors, use an ExecPlan (as described in .agent/PLANS.md) from design to implementation.
