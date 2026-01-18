# Yelp Dataset Build

This step builds datasets with K reviews per selected business.

Inputs:
- Selection file (default: `data/selection/yelp/topic_100.json`)
- Raw Yelp reviews and users under `data/raw/yelp/`

Outputs:
- `data/context/yelp/dataset_K25.jsonl`
- `data/context/yelp/dataset_K50.jsonl`
- `data/context/yelp/dataset_K100.jsonl`
- `data/context/yelp/dataset_K200.jsonl`

Run:

```bash
python scripts/build_dataset.py --data yelp
```

Options:
- `--scales 25,50,100,200`
- `--selection path/to/selection.json`
- `--no-user` to skip user enrichment
