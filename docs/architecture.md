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

## Project Structure

```
data/
├── raw/{dataset}/          # Raw academic dataset
├── hits/{dataset}/         # Topic analysis results (~1.2GB)
├── selection/{dataset}/    # Restaurant selections (topic_100.json)
├── context/{dataset}/      # Built datasets (K=25/50/100/200)
├── query/{dataset}/        # Task prompts
└── answers/{dataset}/      # Ground truth & caches
    ├── judgement_cache.json        # L0 judgement cache (raw + aggregated)
    ├── judgment_overrides.json     # Human corrections to LLM judgment errors
    ├── *_K*_groundtruth.json       # Ground truth files
    ├── batch_manifest_*.json       # Multi-batch tracking (gitignored)
    └── batch_errors_*.jsonl        # Batch API errors for diagnostics (gitignored)

results/
├── dev/{timestamp}_{name}/     # Dev run outputs (results.json)
├── prod/                       # Production runs
├── cache/                      # Method caches (gitignored)
│   └── rag_embeddings.json     # RAG embedding/retrieval cache
└── logs/                       # Debug & pipeline logs (gitignored)
    ├── extraction/             # Extraction pipeline logs
    └── debug/{run_id}/         # LLM prompt/response capture

docs/
├── sessions/           # Claude session logs (written by /bye)
├── README.md           # Doc index
├── architecture.md     # This file
├── tasks/TAXONOMY.md   # 72 task definitions (6 groups × 12 tasks)
└── specs/              # Detailed specifications

scripts/
├── data/               # Data preparation scripts
│   ├── build_dataset.py
│   ├── build_topic_selection.py
│   ├── select_topic_restaurants.py
│   └── download_amazon.sh
├── utils/              # Utility scripts
├── run_g1_allergy.sh   # G1_allergy extraction pipeline
└── manual_review.txt   # Reference doc

src/addm/
├── methods/            # LLM methods (direct, rlm, rag, amos)
├── tasks/              # Extraction, execution, CLI
├── query/              # Query construction system
│   ├── models/         # PolicyIR, Term, Operator
│   ├── libraries/      # Term & operator YAML files
│   ├── policies/       # Policy definitions (G1-G6, V0-V3)
│   └── generators/     # Prompt generation
├── data/               # Dataset loaders
├── eval/               # Evaluation metrics
└── llm.py              # LLM service

.claude/                # Claude Code configuration
├── CLAUDE.md           # Project essentials (minimal)
└── rules/              # Detailed reference docs
```
