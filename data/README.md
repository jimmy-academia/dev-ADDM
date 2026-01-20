# Data Directory

Quick reference for data subdirectories. See [docs/specs/](../docs/specs/) for detailed specifications.

```
data/
├── raw/        → Raw academic datasets (Yelp JSON) [gitignored]
├── selection/  → Topic hits (G*.json) + restaurant selections (topic_*.json) [large files gitignored]
├── context/    → Built datasets at K=25/50/100/200 review sizes
├── query/      → Generated task prompts from PolicyIR
└── answers/    → Ground truth files & judgment caches
```

## Documentation

| Topic | Spec |
|-------|------|
| Data pipeline | [data_creation.md](../docs/specs/data_creation.md) |
| Raw data sources | [raw_data.md](../docs/specs/raw_data.md) |
| Query construction | [query_construction.md](../docs/specs/query_construction.md) |
