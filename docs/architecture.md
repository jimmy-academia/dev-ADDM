# Architecture Overview

The framework is organized into five layers:

1. **Data layer**: dataset registry, loaders, and context curation.
2. **Method layer**: prompt builders and per-sample execution.
3. **LLM layer**: provider abstraction with async batching.
4. **Evaluation layer**: validators + aggregate metrics.
5. **Output layer**: console display, result logging, and usage tracking.

## Package Layout

- `src/addm/` holds the importable package. This `src/` layout avoids accidental imports from the repo root and mirrors how the package is installed.
- Add new methods as modules under `src/addm/methods/` and register them in `src/addm/methods/__init__.py`.

## Execution Flow

1. CLI parses arguments and configures services (including OutputManager).
2. Dataset loader builds `Dataset` and `Sample` objects.
3. Method runner executes per-sample calls (async or sequential).
4. Usage tracked per-call (UsageTracker), results logged (ResultLogger).
5. Results and metrics persisted to `results/` as single JSON file.

## Output System

Three independent components handle output:

- **OutputManager** (`src/addm/utils/output.py`): Rich-formatted console display
- **ResultLogger** (`src/addm/utils/logging.py`): Experiment result capture to files
- **UsageTracker** (`src/addm/utils/usage.py`): Token/cost tracking with pricing model

See [Output System Spec](specs/output_system.md) for details.
