# Data Directory

Quick reference for data subdirectories. See [docs/specs/](../docs/specs/) for detailed specifications.

```
data/
├── raw/        → Raw academic datasets (Yelp JSON) [gitignored]
├── hits/       → Topic analysis from regex scanning [gitignored]
├── selection/  → Restaurant selections for benchmark (topic_100.json)
├── context/    → Built datasets at K=25/50/100/200 review sizes
├── query/      → Generated task prompts from PolicyIR
└── answers/    → Ground truth files & judgment caches
```

## Documentation

| Topic | Spec |
|-------|------|
| End-to-end workflow | [data_creation.md](../docs/specs/data_creation.md) |
| Raw data sources | [raw_data.md](../docs/specs/raw_data.md) |
| Build process | [dataset_build.md](../docs/specs/dataset_build.md) |
| Dataset schema | [datasets.md](../docs/specs/datasets.md) |
