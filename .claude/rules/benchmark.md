# Benchmark: 72 Tasks

## Structure

**6 Groups × 3 Topics × 4 Variants = 72 Tasks**

| Group | Perspective | Topics |
|-------|-------------|--------|
| G1 | Customer Safety | Allergy, Dietary, Hygiene |
| G2 | Customer Experience | Romance, Business, Group |
| G3 | Customer Value | Price-Worth, Hidden Costs, Time-Value |
| G4 | Owner Operations | Server, Kitchen, Environment |
| G5 | Owner Performance | Capacity, Execution, Consistency |
| G6 | Owner Strategy | Uniqueness, Comparison, Loyalty |

## Policy Variants (V1-V4)

| Variant | Description |
|---------|-------------|
| V1 | Base - single condition per verdict |
| V2 | Extended - multiple conditions per verdict (ANY triggers) |
| V3 | Scoring - point system with thresholds |
| V4 | Scoring + recency weighting |

## Task ID Format

- **New format**: `G1_allergy_V3` (group_topic_variant)
- **Legacy format**: `G1a`, `G1b`, etc. (still supported)

## File Locations

- Policy definitions: `src/addm/query/policies/{G1-G6}/{topic}/V{1-4}.yaml`
- Term libraries: `src/addm/query/libraries/terms/{topic}.yaml`
- Generated prompts: `data/query/yelp/{task}_prompt.txt`
- Full taxonomy: `docs/tasks/TAXONOMY.md`
