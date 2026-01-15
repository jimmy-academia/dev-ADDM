# Context Dataset Specification

Context datasets store per-item context blocks that methods consume to answer queries.

## Schema (JSONL)

Each line is a JSON object with:

- `id`: unique item identifier (e.g., Yelp `business_id`)
- `context`: plain-text context block
- `metadata`: structured fields for downstream use

Example:

```json
{"id": "abc", "context": "Business: ...", "metadata": {"business": {...}, "reviews": [...]}}
```

## Yelp Context Builder

The Yelp pipeline aggregates reviews for each business into a single context block.

Input directory:
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_review.json`

Output:
- JSONL at a user-defined path (recommended: `data/processed/yelp_context.jsonl`)

Config:
- `--city`: filter by city (optional)
- `--category`: filter by category (repeatable)
- `--max-businesses`: limit number of businesses
- `--max-reviews`: cap reviews per business
- `--min-reviews`: filter businesses with too few reviews

Run:

```bash
python scripts/build_yelp_context.py \
  --data yelp \
  --output data/processed/yelp_context.jsonl \
  --city "Philadelphia" \
  --category "Cafes" \
  --max-reviews 50
```
