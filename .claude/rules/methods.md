# Methods

Available methods for `--method` flag in `run_experiment.py`:

| Method | Description | Token Cost |
|--------|-------------|------------|
| `direct` | All K reviews in prompt (default) | ~K×200 tokens |
| `rlm` | Recursive LLM - code execution to search | ~50k tokens |
| `rag` | Retrieval-Augmented Generation | ~5k tokens |
| `amos` | Adaptive Multi-Output Sampling (proposed) | ~5k tokens |

## RLM Method

Uses [recursive-llm](https://github.com/ysz/recursive-llm) (forked to `lib/recursive-llm/`)

- Stores reviews as Python variable, LLM writes code to search
- Token budget: `--token-limit 50000` (default)
- **Known issue**: gpt-5-nano produces inconsistent results

```bash
# Basic RLM
.venv/bin/python -m addm.tasks.cli.run_experiment \
    --policy G1_allergy_V2 -n 1 --k 50 --dev --method rlm

# Custom token limit
.venv/bin/python -m addm.tasks.cli.run_experiment \
    --policy G1_allergy_V2 -n 1 --k 50 --dev --method rlm --token-limit 30000
```

## Token Budget Comparison

| Method | Tokens/Restaurant |
|--------|-------------------|
| AMOS (proposed) | ~5k |
| RLM | ~50k (10x, acceptable for comparison) |

---

**Full documentation:** [docs/BASELINES.md](../../docs/BASELINES.md) | **Add new method:** [docs/developer/add-method.md](../../docs/developer/add-method.md)
