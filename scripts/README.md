# Scripts

Utility scripts for data acquisition and processing.

## Data Pipeline Scripts

| Script | Description |
|--------|-------------|
| `search_restaurants.py` | Search reviews for topic keywords (G1-G6) |
| `select_topic_restaurants.py` | Select restaurants by topic coverage |
| `build_dataset.py` | Build K=25/50/100/200 review datasets |

## Verification Scripts

| Script | Description |
|--------|-------------|
| `verify_formulas.py` | Verify all 72 formula modules |

## Data Scripts

| Script | Description |
|--------|-------------|
| `download_amazon.sh` | Download Amazon Reviews 2023 dataset |

## Quick Start

```bash
# Full topic-coverage pipeline
.venv/bin/python scripts/search_restaurants.py --data yelp --all
.venv/bin/python scripts/select_topic_restaurants.py --data yelp
.venv/bin/python scripts/build_dataset.py --data yelp
```

## Usage Examples

### Search for keyword hits

```bash
# Search single topic
.venv/bin/python scripts/search_restaurants.py --data yelp --topic allergy

# Search all G1 topics (allergy, dietary, hygiene)
.venv/bin/python scripts/search_restaurants.py --data yelp --group G1

# Search all 18 topics
.venv/bin/python scripts/search_restaurants.py --data yelp --all
```

### Select restaurants

```bash
# Default: 100 restaurants, balanced across quadrants
.venv/bin/python scripts/select_topic_restaurants.py --data yelp

# Select 150 restaurants
.venv/bin/python scripts/select_topic_restaurants.py --data yelp --target 150
```

### Build datasets

```bash
# Build all K values (25, 50, 100, 200)
.venv/bin/python scripts/build_dataset.py --data yelp

# Build specific K values
.venv/bin/python scripts/build_dataset.py --data yelp --scales 50,100
```

For detailed usage, see [docs/specs/data_creation.md](../docs/specs/data_creation.md).
