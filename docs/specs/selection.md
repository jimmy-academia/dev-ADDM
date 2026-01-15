# Core Context Selection

This pipeline selects a stratified set of 100 businesses to reduce confounding variables.

Strategy (ported from `../anot/explore/doc/DATA_STRATEGY.md`):
- Filter city (default: Philadelphia) and minimum review count.
- Restrict to 5 anchor categories (Italian, Coffee & Tea, Pizza, Steakhouses, Chinese).
- Stratify into 4 quadrants per category (high/low reviews × high/low stars).
- Sample 20 per category (5 per quadrant), then fill if needed.

Run:

```bash
python scripts/select_contexts.py --data yelp
```

Output:
- JSON list of business entries with `stratification_tag`.
- Default path: `data/processed/yelp/core_100.json`
