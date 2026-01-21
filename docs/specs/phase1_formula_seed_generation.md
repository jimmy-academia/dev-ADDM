# Phase 1: Formula Seed Generation

AMOS Phase 1 converts a natural language policy agenda into a structured Formula Seed JSON specification. This document describes how the PLAN_AND_ACT approach works after the Level 1+2 generalization.

## Overview

**What Phase 1 Does**: Converts a natural language agenda into an executable Formula Seed JSON specification that Phase 2 can execute against restaurant reviews.

**Key Design Insight**: The prompts are domain-agnostic. All content (field names, values, thresholds, keywords) is extracted from the agenda text, not hardcoded by domain.

**Input**: A `.txt` agenda file (e.g., `data/query/yelp/G1_allergy_V2_prompt.txt`)

**Output**: Formula Seed JSON for Phase 2 execution

## Architecture

```
Natural Language Agenda (.txt)
           ↓
     ┌──────────┐
     │ OBSERVE  │ → Extract policy type, categories, values, thresholds
     └────┬─────┘
          ↓
     ┌──────────┐
     │   PLAN   │ → Design keywords, fields, compute strategy
     └────┬─────┘
          ↓
     ┌──────────┐
     │   ACT    │ → Generate Formula Seed JSON
     └────┬─────┘
          ↓
     ┌──────────┐
     │ VALIDATE │ → Check schema, fix errors if needed (up to 3 attempts)
     └────┬─────┘
          ↓
    Formula Seed JSON
```

**Files**:
- Entry point: `src/addm/methods/amos/phase1.py`
- PLAN_AND_ACT prompts: `src/addm/methods/amos/phase1_plan_and_act.py`
- Configuration: `src/addm/methods/amos/config.py`

## The Three Steps

### Step 1: OBSERVE

Analyzes the agenda and extracts structured observations.

**Input**: Raw agenda text (from `.txt` file)

**Output**: Observations JSON with extracted structure

**What It Extracts**:

| Field | Description |
|-------|-------------|
| `policy_type` | "scoring" or "rule_based" |
| `core_concepts` | Primary topic and explicit terms from agenda |
| `categories` | Severity/outcome field with its values (e.g., "mild", "moderate", "severe") |
| `scoring_system` | Point values and modifiers (for scoring policies) |
| `verdict_rules` | Thresholds and conditions for verdict determination |
| `account_handling` | Account types with field_name and possible values |
| `description_field` | Incident description field name and purpose |
| `extraction_fields` | Domain-specific modifier fields |

**Policy Type Detection**:
- **Scoring**: Agenda specifies point values (e.g., "Severe: 15 points"). Verdict uses score thresholds.
- **Rule-based**: Agenda specifies count thresholds (e.g., "Critical if 2+ severe incidents"). No point values.

### Step 2: PLAN

Creates a strategy for generating the Formula Seed based on observations.

**Input**: Observations from OBSERVE step

**Output**: Strategy JSON

**What It Plans**:

| Strategy | Description |
|----------|-------------|
| `keyword_strategy` | Terms derived from observations (core concepts, category indicators, account type phrases) |
| `field_strategy` | Required and additional extraction fields |
| `compute_strategy` | Operations based on policy type (scoring vs rule-based) |
| `search_strategy` | Early stopping and priority logic |

**Keyword Derivation**:
- Primary terms from `core_concepts.primary_topic`
- Category indicators from category definitions
- Account type indicators (firsthand/secondhand/hypothetical phrases)
- Priority terms for early stopping

### Step 3: ACT

Generates the complete Formula Seed following the plan.

**Input**: Observations + Plan

**Output**: Complete Formula Seed JSON

**Branching by Policy Type**:

For **scoring** policies:
```
BASE_POINTS (from severity) + MODIFIER_POINTS (from modifiers) = SCORE → VERDICT
```

For **rule_based** policies:
```
N_SEVERE, N_MODERATE, etc. (counts) → VERDICT (via count thresholds)
```

## Formula Seed JSON Schema

The Formula Seed has this structure:

```json
{
  "task_name": "string",

  "filter": {
    "keywords": ["string", ...]
  },

  "extract": {
    "fields": [
      {
        "name": "FIELD_NAME",
        "type": "enum | string | int | float | bool",
        "values": {"value1": "description1", ...},
        "description": "for string fields"
      }
    ]
  },

  "compute": [
    {"name": "...", "op": "count | sum | expr | case", ...}
  ],

  "output": ["VERDICT", "SCORE", "N_INCIDENTS"],

  "search_strategy": {
    "priority_keywords": ["..."],
    "priority_expr": "Python expression",
    "stopping_condition": "Python expression",
    "early_verdict_expr": "Python expression",
    "use_embeddings_when": "Python expression"
  },

  "expansion_hints": {
    "domain": "string",
    "expand_on": ["field names to expand"]
  }
}
```

### Compute Operation Types

| Operation | Format | Purpose |
|-----------|--------|---------|
| `count` | `{"name": "N_X", "op": "count", "where": {...}}` | Count extractions matching conditions |
| `sum` | `{"name": "X_POINTS", "op": "sum", "expr": "CASE WHEN...", "where": {...}}` | Sum values using SQL-style CASE WHEN |
| `expr` | `{"name": "TOTAL", "op": "expr", "expr": "A + B"}` | Combine already-computed values |
| `case` | `{"name": "VERDICT", "op": "case", "source": "SCORE", "rules": [...]}` | Threshold-based classification |

### Field Value Format

**Important**: Enum field values must be a dict, not an array:

```json
{
  "name": "INCIDENT_SEVERITY",
  "type": "enum",
  "values": {
    "No reaction": "No allergic reaction described",
    "Mild incident": "Minor symptoms, not requiring urgent care",
    "Moderate incident": "Visible symptoms requiring medication",
    "Severe incident": "Life-threatening reaction"
  }
}
```

## Hardcoded vs Extracted

| Element | Status | Notes |
|---------|--------|-------|
| Pipeline structure | Hardcoded | OBSERVE→PLAN→ACT fixed sequence |
| Policy types | Hardcoded | "scoring" or "rule_based" only |
| Compute operation types | Hardcoded | count, sum, expr, case |
| JSON schema structure | Hardcoded | filter, extract, compute, output sections |
| Account type field name | **Extracted** | From agenda (e.g., ACCOUNT_TYPE, EVIDENCE_SOURCE) |
| Account type values | **Extracted** | From agenda (e.g., firsthand, secondhand, hypothetical) |
| Description field name | **Extracted** | From agenda (e.g., SPECIFIC_INCIDENT) |
| Category field name | **Extracted** | From agenda (e.g., INCIDENT_SEVERITY, OUTCOME) |
| Category values | **Extracted** | From agenda (e.g., mild, moderate, severe) |
| All point values | **Extracted** | From agenda scoring system |
| All thresholds | **Extracted** | From agenda verdict rules |
| All keywords | **Extracted** | Derived from agenda concepts |
| Verdict labels | **Extracted** | From agenda (e.g., "Critical Risk", "High Risk") |
| Modifier fields | **Extracted** | From agenda (e.g., STAFF_RESPONSE, ASSURANCE_OF_SAFETY) |

**Ratio**: ~15-20% hardcoded structure, ~80-85% extracted content

## Validation & Fix Loop

After generation, the Formula Seed undergoes validation (`phase1.py:_validate_formula_seed`):

1. **Required keys check**: filter, extract, compute, output must exist
2. **Type validation**: keywords is a list, fields is a list, etc.
3. **Expression syntax validation**: Python expressions must compile
4. **Field reference validation**: Compute operations can only reference defined extraction fields

If validation fails, the LLM gets up to 3 attempts to fix the errors via `FIX_PROMPT`.

**Common fixes**:
- Add missing extraction fields referenced in compute operations
- Fix quote matching in expressions
- Correct malformed JSON

## Example Walkthrough: G1_allergy_V2

### Input Agenda (excerpt)

```markdown
# Allergy Safety Risk Assessment

## Scoring System
For each firsthand incident, assign points based on severity:
- Mild incident: 2 points
- Moderate incident: 5 points
- Severe incident: 15 points

Additional modifiers:
- False assurance: +5 points
- Dismissive staff: +3 points

## Verdict Rules
- Critical Risk if total score >= 8
- High Risk if total score >= 4
- Low Risk otherwise
```

### OBSERVE Output

```json
{
  "policy_type": "scoring",
  "core_concepts": {
    "primary_topic": "allergy safety",
    "explicit_terms": ["allergic reaction", "cross-contact", "accommodation"]
  },
  "categories": {
    "field_name": "INCIDENT_SEVERITY",
    "values": [
      {"name": "No reaction", "description": "No allergic reaction"},
      {"name": "Mild incident", "description": "Minor symptoms"},
      {"name": "Moderate incident", "description": "Visible symptoms"},
      {"name": "Severe incident", "description": "Life-threatening"}
    ]
  },
  "scoring_system": {
    "has_scoring": true,
    "base_point_categories": [
      {"category": "Mild incident", "points": 2},
      {"category": "Moderate incident", "points": 5},
      {"category": "Severe incident", "points": 15}
    ],
    "modifiers": [
      {"name": "False assurance", "points": 5},
      {"name": "Dismissive staff", "points": 3}
    ]
  },
  "verdict_rules": {
    "type": "scoring",
    "rules": [
      {"verdict": "Critical Risk", "condition": ">= 8"},
      {"verdict": "High Risk", "condition": ">= 4"},
      {"verdict": "Low Risk", "condition": "else"}
    ]
  },
  "account_handling": {
    "field_name": "ACCOUNT_TYPE",
    "types": [
      {"type": "Firsthand", "counts_for_verdict": true},
      {"type": "Secondhand", "counts_for_verdict": false},
      {"type": "Hypothetical", "counts_for_verdict": false}
    ]
  }
}
```

### PLAN Output

```json
{
  "policy_type": "scoring",
  "keyword_strategy": {
    "from_core_concepts": {
      "primary_terms": ["allergy", "allergic", "allergen"],
      "related_terms": ["reaction", "cross-contact", "accommodation"]
    },
    "from_category_definitions": {
      "category_indicators": {
        "Mild incident": ["discomfort", "minor symptoms"],
        "Moderate incident": ["hives", "swelling", "antihistamine"],
        "Severe incident": ["anaphylaxis", "EpiPen", "emergency", "hospitalization"]
      }
    },
    "priority_terms": ["anaphylaxis", "EpiPen", "severe", "emergency"]
  },
  "compute_strategy": {
    "reasoning": "Scoring policy - need BASE_POINTS, MODIFIER_POINTS, SCORE",
    "for_scoring_type": {
      "base_scoring": "CASE WHEN severity = 'Severe incident' THEN 15 WHEN ... ELSE 0 END",
      "modifier_scoring": "CASE WHEN assurance = 'given' THEN 5 ELSE 0 END + ..."
    }
  }
}
```

### Final Formula Seed

See `results/dev/formula_seeds_snapshot/G1_allergy.json` for the complete output:

```json
{
  "task_name": "Allergy Safety Risk Assessment",
  "filter": {
    "keywords": ["allergy", "allergic", "anaphylaxis", "EpiPen", ...]
  },
  "extract": {
    "fields": [
      {"name": "ACCOUNT_TYPE", "type": "enum", "values": {...}},
      {"name": "INCIDENT_SEVERITY", "type": "enum", "values": {...}},
      {"name": "SPECIFIC_INCIDENT", "type": "string", ...},
      {"name": "ASSURANCE_OF_SAFETY", "type": "enum", ...},
      {"name": "STAFF_RESPONSE", "type": "enum", ...}
    ]
  },
  "compute": [
    {"name": "N_INCIDENTS", "op": "count", "where": {"ACCOUNT_TYPE": "Firsthand"}},
    {"name": "BASE_POINTS", "op": "sum", "expr": "CASE WHEN ... THEN 15 ... END", "where": {...}},
    {"name": "MODIFIER_POINTS", "op": "sum", "expr": "CASE WHEN ... THEN 5 ... END", "where": {...}},
    {"name": "SCORE", "op": "expr", "expr": "BASE_POINTS + MODIFIER_POINTS"},
    {"name": "VERDICT", "op": "case", "source": "SCORE", "rules": [
      {"when": ">= 8", "then": "'Critical Risk'"},
      {"when": ">= 4", "then": "'High Risk'"},
      {"else": "'Low Risk'"}
    ]}
  ],
  "output": ["VERDICT", "SCORE", "N_INCIDENTS"]
}
```

## Alternative Approaches

Phase 1 supports three generation approaches, configured via `Phase1Approach`:

| Approach | LLM Calls | Description |
|----------|-----------|-------------|
| `PLAN_AND_ACT` | 3 | Fixed pipeline: OBSERVE → PLAN → ACT. Default. |
| `REACT` | 5-10 | Iterative loop with self-correction |
| `REFLEXION` | 7-15 | Initial generation + quality analysis + revision |

Configure in code:
```python
from addm.methods.amos.config import AMOSConfig, Phase1Approach

config = AMOSConfig(phase1_approach=Phase1Approach.PLAN_AND_ACT)
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/addm/methods/amos/phase1.py` | Entry point, validation, fix loop |
| `src/addm/methods/amos/phase1_plan_and_act.py` | OBSERVE/PLAN/ACT prompts |
| `src/addm/methods/amos/phase1_react.py` | ReAct approach implementation |
| `src/addm/methods/amos/phase1_reflexion.py` | Reflexion approach implementation |
| `src/addm/methods/amos/config.py` | AMOSConfig and Phase1Approach |
| `data/query/yelp/{policy}_prompt.txt` | Input agenda files |
| `results/dev/formula_seeds_snapshot/` | Example Formula Seed outputs |

## Related Documentation

- [AMOS Generalization](../AMOS_GENERALIZATION.md) - Fixes, limitations, and task coverage assessment
- [Methods Overview](../../.claude/rules/methods.md) - Comparison with other methods (direct, rlm, rag)
