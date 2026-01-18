# Data Creation Workflow

End-to-end steps to build datasets from raw data.

## Prerequisites

Place raw files in `data/raw/{dataset}/`. For Yelp:

- `data/raw/yelp/yelp_academic_dataset_business.json`
- `data/raw/yelp/yelp_academic_dataset_review.json`
- `data/raw/yelp/yelp_academic_dataset_user.json`

## Topic-Coverage Workflow

### Step 1: Search for keyword hits

Scan reviews for topic-relevant keywords across all 18 topics (6 groups × 3 topics):

```bash
# Search all topics
.venv/bin/python scripts/search_restaurants.py --data yelp --all

# Or search by group
.venv/bin/python scripts/search_restaurants.py --data yelp --group G1
```

Output: `data/hits/{dataset}/G{1-6}_{topic}.json`

### Step 2: Select restaurants by topic coverage

Select a balanced set of restaurants with good coverage across topics:

```bash
.venv/bin/python scripts/select_topic_restaurants.py --data yelp
```

Output:
- `data/selection/{dataset}/topic_100.json` - Selected restaurants
- `data/selection/{dataset}/topic_ranked_all.json` - Full ranking

### Step 3: Build review datasets

Build datasets with K reviews per restaurant (K=25/50/100/200):

```bash
.venv/bin/python scripts/build_dataset.py --data yelp
```

Output: `data/context/{dataset}/dataset_K{25,50,100,200}.jsonl`

## Quick Start (Full Pipeline)

```bash
# Run all three steps
.venv/bin/python scripts/search_restaurants.py --data yelp --all
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
├── hits/{dataset}/         # Keyword search results
│   ├── G{1-6}_{topic}.json
│   └── topic/              # Detailed topic analysis (large)
├── selection/{dataset}/    # Restaurant selections
│   ├── topic_100.json
│   └── topic_ranked_all.json
├── context/{dataset}/      # Built datasets
│   └── dataset_K{25,50,100,200}.jsonl
└── query/{dataset}/        # Task prompts
```

## Dataset Record Schema

Each line in `dataset_K*.jsonl`:

```json
{
  "business_id": "abc123",
  "stratification": "HighVol_HighStars",
  "business": {
    "business_id": "abc123",
    "name": "Restaurant Name",
    "address": "...",
    "city": "Philadelphia",
    "stars": 4.5,
    "review_count": 150,
    "categories": "Italian, Pizza",
    "hours": {...},
    "attributes": {...}
  },
  "reviews": [
    {
      "review_id": "...",
      "user_id": "...",
      "stars": 5,
      "date": "2023-01-15",
      "text": "...",
      "user": {
        "name": "...",
        "review_count": 45,
        "average_stars": 4.2
      }
    }
  ],
  "review_count_actual": 50
}
```

## Custom Datasets

For non-Yelp datasets, use path overrides:

```bash
# Custom raw data paths
.venv/bin/python scripts/search_restaurants.py --data custom \
    --review-file data/raw/custom/reviews.json \
    --business-file data/raw/custom/business.json

# Custom selection path
.venv/bin/python scripts/build_dataset.py --data custom \
    --selection data/selection/custom/my_selection.json
```

## CLI Reference

### search_restaurants.py

| Flag | Description |
|------|-------------|
| `--data` | Dataset name (required) |
| `--topic` | Single topic to search |
| `--group` | Task group (G1-G6) |
| `--all` | Search all 18 topics |
| `--min-hits` | Minimum keyword hits (default: 10) |
| `--top-n` | Top N restaurants per topic (default: 100) |
| `--review-file` | Override review file path |
| `--business-file` | Override business file path |

### select_topic_restaurants.py

| Flag | Description |
|------|-------------|
| `--data` | Dataset name (required) |
| `--target` | Number to select (default: 100) |
| `--no-balance` | Skip quadrant balancing |
| `--keyword-hits-dir` | Override input directory |
| `--output-dir` | Override output directory |

### build_dataset.py

| Flag | Description |
|------|-------------|
| `--data` | Dataset name (required) |
| `--scales` | K values (default: 25,50,100,200) |
| `--selection` | Override selection file |
| `--output-dir` | Override output directory |
| `--no-user` | Skip user metadata enrichment |
