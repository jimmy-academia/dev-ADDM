# Data Pipeline

## Context Building

```bash
# Full pipeline (analyze → select → build)
.venv/bin/python scripts/data/build_topic_selection.py --data yelp
.venv/bin/python scripts/data/select_topic_restaurants.py --data yelp
.venv/bin/python scripts/data/build_dataset.py --data yelp

# Rebuild datasets only (if topic_100.json exists)
.venv/bin/python scripts/data/build_dataset.py --data yelp
```

## Selection Algorithm

Layered greedy by cell coverage:
- 18 topics × 2 severities (critical/high) = 36 cells
- Fills level n (all cells ≥n) before level n+1
- Multi-topic restaurants prioritized

## Data Notes

**Hits lists** (`data/selection/yelp/G1_*.json`):
- `critical_list` and `high_list` are regex-based, not 100% accurate
- Use as starting point for finding relevant restaurants
- Critical reviews may not be in K=25 subset (sampling effect)
- Always verify actual review content

**Research goal**: Find complexity that causes baseline LLM errors
- Test across K=25, 50, 100, 200 to observe scaling
- Look for cases where more context → incorrect verdicts

## Query Pipeline

PolicyIR → NL prompts

```bash
.venv/bin/python -m addm.query.cli.generate \
    --policy src/addm/query/policies/G1/allergy/V2.yaml \
    --output data/query/yelp/G1_allergy_V2_prompt.txt
```

---

**Full documentation:** [docs/specs/data_creation.md](../../docs/specs/data_creation.md) | **Query system:** [docs/specs/query_construction.md](../../docs/specs/query_construction.md)
