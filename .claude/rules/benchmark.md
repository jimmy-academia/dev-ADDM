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

## Policy Variants (V0-V3)

| Variant | Description |
|---------|-------------|
| V0 | Base - aggregation, multiple incidents required |
| V1 | +Override - single-instance triggers |
| V2 | +Scoring - point system, thresholds |
| V3 | +Recency - time decay, exceptions |

## Task ID Format

- **New format**: `G1_allergy_V2` (group_topic_variant)
- **Legacy format**: `G1a`, `G1b`, etc. (still supported)

## File Locations

- Policy definitions: `src/addm/query/policies/{G1-G6}/{topic}/V{0-3}.yaml`
- Term libraries: `src/addm/query/libraries/terms/{topic}.yaml`
- Generated prompts: `data/query/yelp/{task}_prompt.txt`
- Full taxonomy: `docs/tasks/TAXONOMY.md`
