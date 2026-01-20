# Data Creation Workflow

End-to-end steps to build datasets from raw data.

## Prerequisites

Place raw files in `data/raw/{dataset}/`. For Yelp:

- `data/raw/yelp/yelp_academic_dataset_business.json`
- `data/raw/yelp/yelp_academic_dataset_review.json`
- `data/raw/yelp/yelp_academic_dataset_user.json`

## Pipeline Overview

```
data/raw/ → data/selection/ (hits + restaurant selections) → data/context/
```

### Step 1: Topic Analysis (`scripts/build_topic_selection.py`)

Scan all reviews for topic-relevant patterns across 18 topics (6 groups × 3 topics).
Uses weighted regex patterns to score reviews by severity (critical vs. high).

```bash
# Analyze all 18 topics
.venv/bin/python scripts/build_topic_selection.py --data yelp

# Analyze single topic
.venv/bin/python scripts/build_topic_selection.py --data yelp --topic G1_allergy

# Preview patterns without processing
.venv/bin/python scripts/build_topic_selection.py --dry-run
```

Output: `data/selection/{dataset}/G{1-6}_{topic}.json`

### Step 2: Restaurant Selection (`scripts/select_topic_restaurants.py`)

Select restaurants using layered greedy selection to ensure balanced cell coverage:
- Cells = (topic, severity) where severity is "critical" or "high"
- 18 topics × 2 severities = 36 cells
- Algorithm fills level n (all cells have ≥n restaurants) before level n+1
- Multi-topic restaurants naturally prioritized (fill more cells per pick)

```bash
# Default: 100 restaurants
.venv/bin/python scripts/select_topic_restaurants.py --data yelp

# Select 50 restaurants
.venv/bin/python scripts/select_topic_restaurants.py --data yelp --target 50
```

Output:
- `data/selection/{dataset}/topic_100.json` - Selected restaurants
- `data/selection/{dataset}/topic_all.json` - Full ranking by cell count

### Step 3: Build Datasets (`scripts/build_dataset.py`)

Build datasets with K reviews per restaurant (K=25/50/100/200).

```bash
.venv/bin/python scripts/build_dataset.py --data yelp
```

Output: `data/context/{dataset}/dataset_K{25,50,100,200}.jsonl`

## Quick Start (Full Pipeline)

```bash
# Run all three steps
.venv/bin/python scripts/build_topic_selection.py --data yelp
.venv/bin/python scripts/select_topic_restaurants.py --data yelp
.venv/bin/python scripts/build_dataset.py --data yelp
```

## Directory Structure

```
data/
├── raw/{dataset}/          # Raw academic dataset
│   ├── {dataset}_academic_dataset_business.json
│   ├── {dataset}_academic_dataset_review.json
│   └── {dataset}_academic_dataset_user.json
├── selection/{dataset}/    # Topic hits + restaurant selections
│   ├── G{1-6}_{topic}.json  # Topic analysis results (hits)
│   ├── topic_100.json       # Selected restaurants
│   └── topic_all.json       # Full ranking
├── context/{dataset}/      # Built datasets
│   └── dataset_K{25,50,100,200}.jsonl
└── query/{dataset}/        # Task prompts
```

## Dataset Record Schema

Each line in `dataset_K*.jsonl`:

```json
{
  "business_id": "abc123",
  "business": {
    "business_id": "abc123",
    "name": "Restaurant Name",
    "address": "123 Main St",
    "city": "Philadelphia",
    "state": "PA",
    "postal_code": "19102",
    "latitude": 39.948591,
    "longitude": -75.164706,
    "stars": 4.5,
    "review_count": 150,
    "is_open": 1,
    "attributes": {...},
    "categories": "Italian, Pizza",
    "hours": {...}
  },
  "reviews": [
    {
      "review_id": "...",
      "user_id": "...",
      "business_id": "...",
      "stars": 5,
      "useful": 0,
      "funny": 0,
      "cool": 0,
      "text": "Full review text (not truncated)",
      "date": "2023-01-15 12:30:00",
      "user": {
        "name": "...",
        "review_count": 45,
        "yelping_since": "2010-01-15 10:00:00",
        "elite": "",
        "average_stars": 4.2,
        "fans": 5
      }
    }
  ],
  "review_count_actual": 50
}
```

Note: Review text is preserved in full (not truncated). The `user` object is embedded from the Yelp user dataset.

## Custom Datasets

For non-Yelp datasets, use path overrides:

```bash
# Custom selection path
.venv/bin/python scripts/build_dataset.py --data custom \
    --selection data/selection/custom/my_selection.json
```

## CLI Reference

### build_topic_selection.py

| Flag | Description |
|------|-------------|
| `--data` | Dataset name (default: yelp) |
| `--topic` | Single topic to process (e.g., G1_allergy) |
| `--dry-run` | Show patterns only, don't process |
| `--parallel` | Number of parallel processes (default: auto) |

### select_topic_restaurants.py

| Flag | Description |
|------|-------------|
| `--data` | Dataset name (required) |
| `--target` | Number to select (default: 100) |
| `--hits-dir` | Override input directory |
| `--output-dir` | Override output directory |
| `--quiet` | Suppress progress output |

### build_dataset.py

| Flag | Description |
|------|-------------|
| `--data` | Dataset name (required) |
| `--scales` | K values (default: 25,50,100,200) |
| `--selection` | Override selection file |
| `--output-dir` | Override output directory |
| `--no-user` | Skip user metadata enrichment |
