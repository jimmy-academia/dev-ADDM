# ADDM Documentation Index

## Quick Start

- [Architecture Overview](architecture.md) - Pipeline overview and module responsibilities
- [CLI Reference](specs/cli.md) - CLI arguments and defaults

## Data Pipeline

- [Data Creation Workflow](specs/data_creation.md) - End-to-end data creation workflow
- [Raw Data Sources](specs/raw_data.md) - Raw data sources and storage
- [Dataset Build](specs/dataset_build.md) - Build datasets with reviews
- [Dataset Schema](specs/datasets.md) - Dataset structure and fields

## Benchmark Tasks

**72 Tasks** = 6 Groups x 3 Topics x 4 Variants

- [Task Taxonomy](tasks/TAXONOMY.md) - Full task list and grouping
- [G1 Health & Safety](tasks/groups/G1_health_safety.md)
- [G2 Social Context](tasks/groups/G2_social_context.md)
- [G3 Economic Value](tasks/groups/G3_economic_value.md)
- [G4 Talent & Performance](tasks/groups/G4_talent_performance.md)
- [G5 Operational Efficiency](tasks/groups/G5_operational.md)
- [G6 Competitive Strategy](tasks/groups/G6_competitive.md)

| Group | Perspective | Topics |
|-------|-------------|--------|
| G1 | Customer Health | Allergy, Dietary, Hygiene |
| G2 | Customer Social | Romance, Business, Group |
| G3 | Customer Economic | Price-Worth, Hidden Costs, Time-Value |
| G4 | Owner Talent | Server, Kitchen, Environment |
| G5 | Owner Operations | Capacity, Execution, Consistency |
| G6 | Owner Strategy | Uniqueness, Comparison, Loyalty |

**Variants:**
- **a** = Simple formula
- **b** = Simple + L1.5 grouping (L1.5 bonus affects score)
- **c** = Complex formula (credibility weighting)
- **d** = Complex + L1.5 (both features)

Task prompts: `data/tasks/yelp/G{1-6}{a-l}_prompt.txt`

## Method Specifications

- [Context Selection](specs/selection.md) - Core context selection strategy

## Output & Evaluation

- [Results Schema](specs/outputs.md) - Results and metrics schema

## Project Structure

```
data/
├── raw/yelp/           # Raw Yelp dataset
├── processed/yelp/     # Built datasets (K=25/50/100/200)
└── tasks/yelp/         # Prompts, cache, ground truth

src/addm/
├── methods/            # LLM methods
├── tasks/formulas/     # Formula modules (G1a.py, etc.)
├── tasks/              # Extraction, execution
├── data/               # Dataset loaders
└── eval/               # Metrics
```
