# Architecture Overview

The framework is organized into four layers:

1. Data layer: dataset registry, loaders, and context curation.
2. Method layer: prompt builders and per-sample execution.
3. LLM layer: provider abstraction with async batching.
4. Evaluation layer: validators + aggregate metrics.

Package layout:

- `src/addm/` holds the importable package. This `src/` layout avoids accidental imports from the repo root and mirrors how the package is installed.
- Add new methods as modules under `src/addm/methods/` and register them in `src/addm/methods/__init__.py`.

Execution flow:

1. CLI parses arguments and configures services.
2. Dataset loader builds `Dataset` and `Sample` objects.
3. Method runner executes per-sample calls (async or sequential).
4. Results and metrics are persisted in `results/`.
