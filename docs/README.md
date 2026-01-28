# ADDM Documentation Index

## Quick Start

- **[Quickstart Guide (quickstart.md)](quickstart.md)** - Run your first evaluation in 5 steps
- [Architecture Overview (architecture.md)](architecture.md) - Pipeline overview and module responsibilities
- [CLI Reference (specs/cli.md)](specs/cli.md) - CLI arguments and defaults
- [Project Roadmap (ROADMAP.md)](ROADMAP.md) - Milestones, progress, and technical debt
- [Troubleshooting (troubleshooting.md)](troubleshooting.md) - Common issues and solutions

## Data Pipeline

- [Data Creation Workflow (specs/data_creation.md)](specs/data_creation.md) - End-to-end: raw data → topics → selection → datasets
- [Raw Data Sources (specs/raw_data.md)](specs/raw_data.md) - Yelp dataset format and storage

## Benchmark Tasks

**72 Tasks** = 6 Groups x 3 Topics x 4 Variants

- [Task Taxonomy (tasks/TAXONOMY.md)](tasks/TAXONOMY.md) - Full task list and grouping
- [G1 Health & Safety (tasks/groups/G1_health_safety.md)](tasks/groups/G1_health_safety.md)
- [G2 Social Context (tasks/groups/G2_social_context.md)](tasks/groups/G2_social_context.md)
- [G3 Economic Value (tasks/groups/G3_economic_value.md)](tasks/groups/G3_economic_value.md)
- [G4 Talent & Performance (tasks/groups/G4_talent_performance.md)](tasks/groups/G4_talent_performance.md)
- [G5 Operational Efficiency (tasks/groups/G5_operational.md)](tasks/groups/G5_operational.md)
- [G6 Competitive Strategy (tasks/groups/G6_competitive.md)](tasks/groups/G6_competitive.md)

| Group | Perspective | Topics |
|-------|-------------|--------|
| G1 | Customer Health | Allergy, Dietary, Hygiene |
| G2 | Customer Social | Romance, Business, Group |
| G3 | Customer Economic | Price-Worth, Hidden Costs, Time-Value |
| G4 | Owner Talent | Server, Kitchen, Environment |
| G5 | Owner Operations | Capacity, Execution, Consistency |
| G6 | Owner Strategy | Uniqueness, Comparison, Loyalty |

**Variants:**

*Legacy (a/b/c/d)* - Formula complexity × L1.5:
- **a** = Simple formula
- **b** = Simple + L1.5 grouping
- **c** = Complex formula (credibility weighting)
- **d** = Complex + L1.5

*New (V0-V3)* - Policy evolution:
- **V0** = Base (aggregation, multiple incidents)
- **V1** = +Override (single-instance triggers)
- **V2** = +Scoring (point system)
- **V3** = +Recency (time decay)

Task prompts: `data/query/yelp/G{1-6}{a-l}_prompt.txt` or `G{n}_{topic}_V{0-3}_prompt.txt`

## Query Construction

- [Query Construction System (specs/query_construction.md)](specs/query_construction.md) - PolicyIR → NL prompt generation

## Baselines & Methods

- [Baselines (BASELINES.md)](BASELINES.md) - Implemented baseline methods with citations
- [LLM Modes (specs/llm_modes.md)](specs/llm_modes.md) - Ondemand vs batch execution
- [AMOS Phase 1 Agenda Spec (specs/phase1_agenda_spec_generation.md)](specs/phase1_agenda_spec_generation.md) - Agenda → agenda_spec extraction pipeline

## Developer Guides

- [Adding a New Method (developer/add-method.md)](developer/add-method.md) - Step-by-step guide to implement a baseline method

## Output & Evaluation

- [Output System Architecture (specs/output_system.md)](specs/output_system.md) - Console output, logging, result schema, and usage tracking
- [Usage Tracking (specs/usage_tracking.md)](specs/usage_tracking.md) - Token/cost tracking implementation
- [Ground Truth System (specs/ground_truth.md)](specs/ground_truth.md) - GT generation, aggregation, human overrides
- [Evaluation Metrics (specs/evaluation.md)](specs/evaluation.md) - AUPRC, process score, consistency

## Data Analysis

- [Keyword Search Analysis (data_analysis/keyword_search_analysis.md)](data_analysis/keyword_search_analysis.md)
- [Selection Bias Analysis (data_analysis/selection_bias_analysis.md)](data_analysis/selection_bias_analysis.md)

## Project Structure

```
data/
├── raw/yelp/           # Raw Yelp dataset
├── hits/yelp/          # Keyword hits + topic analysis
├── selection/yelp/     # Restaurant selections (topic_100.json)
├── context/yelp/       # Built datasets (K=25/50/100/200)
└── query/yelp/         # Task prompts

src/addm/
├── methods/            # LLM methods (direct, rlm, rag, amos)
├── tasks/              # Extraction, execution, CLI
├── query/              # Query construction (PolicyIR → prompts)
│   ├── models/         # PolicyIR, Term, Operator
│   ├── libraries/      # Term & operator YAML files
│   └── policies/       # Policy definitions (G1-G6, V0-V3)
├── data/               # Dataset loaders
└── eval/               # Metrics
```
