# Query Construction System

This document specifies the programmatic system for generating natural-language agenda prompts from structured policy definitions.

## Overview

The Query Construction System replaces manual prompt authoring with a structured approach:
- **PolicyIR** (Policy Intermediate Representation): YAML-based structured format for evaluation policies
- **Term Libraries**: Reusable definitions shared across topics
- **Prompt Generator**: Jinja2 templates that render PolicyIR to NL prompts

## V0-V3 Variant Progression

The old a/b/c/d variant system (simple/complex × with/without L1.5) is replaced with V0-V3, representing policy evolution from qualitative to quantitative:

| Variant | Name | Description |
|---------|------|-------------|
| **V0** | Base | Aggregation-based. Requires multiple incidents (e.g., "2+ severe"). No single-instance overrides. |
| **V1** | +Override | Single-instance triggers. One severe incident → Critical. One moderate → High. |
| **V2** | +Scoring | Point-based system with numeric thresholds. Forces explicit calculation. |
| **V3** | +Recency | Recency weighting with time decay. Policy exceptions for old incidents. |

### Prompt Structure by Variant

**V0-V1** (Qualitative):
```
# [Title]
[Purpose paragraph]

## Definitions of Terms
[Auto-generated from term library]

## Verdict Rules
[Precedence ladder: Critical → High → Low]
```

**V2-V3** (Quantitative):
```
# [Title]
[Purpose paragraph]

## Definitions of Terms
[Auto-generated from term library]

## Scoring System
[Point values, modifiers, thresholds]

## Verdict Rules
[Score-based thresholds]
```

## Directory Structure

```
src/addm/query/
├── models/                     # Core data models
│   ├── term.py                 # Term, TermLibrary
│   ├── logic.py                # Operator, LogicLibrary
│   └── policy.py               # PolicyIR, NormativeCore, ScoringSystem
│
├── libraries/                  # Shared definitions (YAML)
│   ├── terms/
│   │   ├── _shared.yaml        # Cross-topic (Account Type, Staff Response)
│   │   ├── allergy.yaml        # Topic-specific terms
│   │   └── ...                 # Other topic files
│   └── operators/
│       └── operators.yaml      # ANY, ALL, COUNT, EXISTS, etc.
│
├── policies/                   # Policy IR definitions
│   └── G1/allergy/
│       ├── V0.yaml             # Base variant
│       ├── V1.yaml             # + Overrides
│       ├── V2.yaml             # + Scoring
│       └── V3.yaml             # + Recency
│
├── generators/                 # Generation pipeline
│   ├── prompt_generator.py     # PolicyIR → NL prompt
│   └── templates/              # Jinja2 templates
│       ├── overview.jinja2
│       ├── definitions.jinja2
│       ├── scoring.jinja2
│       └── verdict_rules.jinja2
│
└── cli/
    └── generate.py             # CLI for prompt generation
```

## PolicyIR Format

```yaml
policy_id: "G1_allergy_V2"
extends: "G1_allergy_V1"  # Optional inheritance

# === OVERVIEW ===
overview:
  title: "Allergy Safety Risk Assessment"
  purpose: |
    Help assess the allergy safety risk for this restaurant...
  incident_definition: |
    Only firsthand accounts are treated as confirmed incident evidence...

# === NORMATIVE ===
normative:
  # Terms (references to term library)
  terms:
    - ref: "shared:ACCOUNT_TYPE"
    - ref: "shared:STAFF_RESPONSE"
    - ref: "allergy:INCIDENT_SEVERITY"
    - ref: "allergy:ASSURANCE_CLAIM"

  # Scoring system (V2+ only)
  scoring:
    description: "Calculate a total risk score..."
    severity_points:
      - label: "Mild incident"
        points: 2
      - label: "Moderate incident"
        points: 5
      - label: "Severe incident"
        points: 15
    modifiers:
      - label: "False assurance"
        description: "incident after explicit safety guarantee"
        points: 5
      - label: "Dismissive staff"
        description: "staff dismissed allergy concern during incident"
        points: 3
      - label: "High-risk cuisine"
        description: "restaurant is Thai, Vietnamese, or Chinese cuisine"
        points: 2
    recency_rules:  # V3 only
      - age: "Within 2 years"
        weight: "full point value"
      - age: "2-3 years old"
        weight: "half point value (multiply by 0.5)"
      - age: "Over 3 years old"
        weight: "quarter point value (multiply by 0.25)"
    thresholds:
      - verdict: "Critical Risk"
        min_score: 8
      - verdict: "High Risk"
        min_score: 4

  # Decision rules
  decision:
    verdicts: ["Low Risk", "High Risk", "Critical Risk"]
    ordered: true
    rules:
      - verdict: "Critical Risk"
        logic: ANY
        conditions:
          - "Total risk score is 8 or higher"
      - verdict: "High Risk"
        precondition: "Critical Risk does not apply"
        logic: ANY
        conditions:
          - "Total risk score is 4 or higher"
      - verdict: "Low Risk"
        default: true
        especially_when:
          - "No firsthand incidents are reported"
          - "Total risk score is below 4"
```

## Term Library Format

```yaml
# libraries/terms/_shared.yaml
ACCOUNT_TYPE:
  name: "Account Type"
  description: "How the reviewer relates to the described event"
  type: enum
  values:
    - id: firsthand
      label: "Firsthand"
      description: "The reviewer or their dining party experienced the event directly"
      examples: ["I had...", "my child..."]
    - id: secondhand
      label: "Secondhand"
      description: "The reviewer reports someone else's experience"
      examples: ["my friend...", "I heard..."]
    - id: hypothetical
      label: "Hypothetical"
      description: "Concern or preference without describing an actual incident"
      examples: ["I'm allergic so I worry..."]

STAFF_RESPONSE:
  name: "Staff Response"
  description: "How staff responded when the concern was raised"
  type: enum
  values:
    - id: accommodated
      label: "Accommodated"
      description: "Staff took the concern seriously and successfully accommodated"
    - id: refused
      label: "Refused"
      description: "Staff stated they could not or would not accommodate"
    - id: dismissive
      label: "Dismissive"
      description: "Staff minimized the concern or did not take it seriously"
    - id: not_described
      label: "Not described"
      description: "No staff interaction about the concern is described"
    - id: unknown
      label: "Unknown"
      description: "Unclear from the review"
```

## Scoring System (V2+)

### Point Values
- **Severity**: Mild=2, Moderate=5, Severe=15
- **Modifiers**: False assurance=+5, Dismissive staff=+3, High-risk cuisine=+2

### Thresholds
- **Critical Risk**: Score ≥ 8
- **High Risk**: Score ≥ 4
- **Low Risk**: Score < 4

### Recency Weighting (V3 only)
- Within 2 years: 1.0× (full value)
- 2-3 years old: 0.5× (half value)
- Over 3 years old: 0.25× (quarter value, or discard if 3+ recent reviews show improvement)

## Generation Commands

```bash
# Generate prompt from PolicyIR
.venv/bin/python -m addm.query.cli.generate \
    --policy src/addm/query/policies/G1/allergy/V2.yaml \
    --output data/query/yelp/G1_allergy_V2_prompt.txt

# Generate all variants for a topic
.venv/bin/python -m addm.query.cli.generate \
    --topic G1/allergy \
    --all-variants
```

## Design Decisions

### NL-Only Prompts
Prompts contain only natural language descriptions. No L0/L1/L2 pseudo-code or programmatic notation. The scoring system is described in words, not formulas.

### Policy-Based Ground Truth
Ground truth computation uses policy definitions (`src/addm/query/policies/`) with deterministic scoring rules in `src/addm/tasks/policy_gt.py`. This allows:
- Prompts to be human-readable (natural language)
- Ground truth computation to be programmatically precise (structured rules)
- Independent evolution of policy variants (V0-V3)

### No Bypass Rules in V2/V3
V2 and V3 use score-based thresholds exclusively, with no single-instance bypasses (e.g., "severe incident → Critical"). This forces the model to perform full scoring calculation.

### High-Risk Cuisine Modifier
Thai, Vietnamese, and Chinese cuisines receive +2 points due to common use of peanuts/tree nuts. Applied once per restaurant, not per incident.

## Current Status

- **Policy definitions**: ✅ All 72 complete (G1-G6, all topics, V0-V3)
- **Term libraries**: ✅ All 18 topics complete
- **Prompt generation**: ✅ CLI tool implemented (`addm.query.cli.generate`)
- **System integration**: ✅ Experiment code updated (`--policy` flag in run_experiment.py)
- **Legacy support**: ✅ Task IDs (G1a-G6l) still supported for backward compatibility
