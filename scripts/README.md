# Scripts

Utility scripts for data acquisition, processing, and pipeline execution.

## Directory Structure

```
scripts/
├── data/                        # Data preparation scripts
│   ├── build_dataset.py
│   ├── build_topic_selection.py
│   ├── select_topic_restaurants.py
│   ├── create_restaurant_ids.py
│   └── download_amazon.sh
└── select_diverse_samples.py    # Sample selection utility
```

## Data Scripts (`data/`)

| Script | Description |
|--------|-------------|
| `build_topic_selection.py` | Analyze reviews for topic patterns (18 topics) |
| `select_topic_restaurants.py` | Select restaurants by cell coverage (layered greedy) |
| `build_dataset.py` | Build K=25/50/100/200 review datasets |
| `create_restaurant_ids.py` | Generate restaurant ID mappings |
| `download_amazon.sh` | Download Amazon Reviews 2023 dataset |

## Quick Start

```bash
# Full data pipeline
.venv/bin/python scripts/data/build_topic_selection.py --data yelp
.venv/bin/python scripts/data/select_topic_restaurants.py --data yelp
.venv/bin/python scripts/data/build_dataset.py --data yelp
```

## Usage Examples

### Topic Analysis

```bash
# Analyze all 18 topics
.venv/bin/python scripts/data/build_topic_selection.py --data yelp

# Analyze single topic
.venv/bin/python scripts/data/build_topic_selection.py --data yelp --topic G1_allergy

# Preview patterns without processing
.venv/bin/python scripts/data/build_topic_selection.py --dry-run
```

### Select restaurants

```bash
# Default: 100 restaurants (layered greedy by cell coverage)
.venv/bin/python scripts/data/select_topic_restaurants.py --data yelp

# Select 50 restaurants
.venv/bin/python scripts/data/select_topic_restaurants.py --data yelp --target 50
```

### Build datasets

```bash
# Build all K values (25, 50, 100, 200)
.venv/bin/python scripts/data/build_dataset.py --data yelp

# Build specific K values
.venv/bin/python scripts/data/build_dataset.py --data yelp --scales 50,100
```

For detailed usage, see [docs/specs/data_creation.md](../docs/specs/data_creation.md).
