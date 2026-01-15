# Data Creation Workflow

End-to-end steps to build Yelp datasets.

1) Place raw data
- `data/raw/yelp/yelp_academic_dataset_business.json`
- `data/raw/yelp/yelp_academic_dataset_review.json`
- `data/raw/yelp/yelp_academic_dataset_user.json`

2) Select core businesses (default 100)

```bash
.venv/bin/python scripts/select_contexts.py --data yelp
```

Output:
- `data/processed/yelp/core_100.json`

3) Build review datasets (K=25/50/100/200)

```bash
.venv/bin/python scripts/build_dataset.py --data yelp
```

Outputs:
- `data/processed/yelp/dataset_K25.jsonl`
- `data/processed/yelp/dataset_K50.jsonl`
- `data/processed/yelp/dataset_K100.jsonl`
- `data/processed/yelp/dataset_K200.jsonl`
