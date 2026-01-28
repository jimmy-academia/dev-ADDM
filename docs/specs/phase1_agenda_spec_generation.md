# Phase 1: Agenda Spec Generation

Phase 1 converts a policy agenda into a structured `agenda_spec` JSON used by
Phase 2. It focuses on extracting *terms* (fields + enum values) and *verdict
rules* (labels + rule logic). It does **not** produce scoring logic or any
Phase 2 execution outputs.

## Overview

**Input**: Agenda text (natural language policy prompt)

**Output**: `agenda_spec` JSON (stored as `agenda_spec.json` in the run dir)

**Entry point**: `src/addm/methods/amos/phase1.py`

**Prompts**: `src/addm/methods/amos/phase1_prompts.py`

## Pipeline Steps

```
Agenda text
  ->
Step 0: Locate blocks (anchors only)
  ->
Step 1: Extract verdict rules skeleton
  ->
Step 2: Select term blocks (anchors only)
  ->
Step 3: Extract term enums (per term)
  ->
Step 4: Compile clauses (per clause)
  ->
Step 5: Assemble + prune + validate
  ->
agenda_spec JSON
```

### Step 0: Locate blocks (anchors only)
- **Goal**: Identify definition blocks and verdict rule blocks via unique
  `start_quote`/`end_quote` anchors.
- **Validation**: Anchors must appear exactly once in the agenda.
- **Fallback**: If slicing fails, use the full agenda text.

### Step 1: Extract verdict rules skeleton
- **Goal**: Extract labels, default label, rule order, and clause quotes.
- **Structure**: Each non-default rule includes connective info and the list of
  raw clause quotes. The default rule records its default quote and optional
  hints.
- **Validation**: All quotes must be exact substrings of the verdict block.

### Step 2: Select term blocks
- **Goal**: Identify only the term definitions required by *non-default* rules.
- **Output**: A list of term blocks, each with `term_title`, `start_quote`,
  `end_quote` anchors.

### Step 3: Extract term enums (per term)
- **Goal**: For each term block, extract enum options and descriptions.
- **Normalization**:
  - `field_id` = normalized `term_title` (UPPERCASE_WITH_UNDERSCORES)
  - `value_id` = normalized enum value (lower_snake_case)
- **Validation**: Each enum option must have a `source_value` and a
  `value_quote` that appears in the term block.

### Step 4: Compile clauses (per clause)
- **Goal**: Convert each clause quote into structured conditions.
- **Output**: `min_count`, `logic`, and a list of `{field_id, values}`
  conditions.
- **Validation**: `field_id` and `values` must be drawn from the extracted
  terms.

### Step 5: Assemble + prune + validate
- **Pruning**: Remove terms not referenced by any non-default clause.
- **Warnings** (non-fatal):
  - Must have exactly one default rule
  - Rules should cover all verdict labels
  - Rule order should end with `default_label`

## agenda_spec Schema

```json
{
  "terms": [
    {
      "term_title": "Account Type",
      "field_id": "ACCOUNT_TYPE",
      "type": "enum",
      "values": [
        {
          "value_id": "firsthand",
          "source_value": "firsthand",
          "description": "The reviewer experienced the event",
          "value_quote": "- **firsthand**: The reviewer..."
        }
      ]
    }
  ],
  "verdict_rules": {
    "labels": ["Critical Risk", "High Risk", "Low Risk"],
    "default_label": "Low Risk",
    "order": ["Critical Risk", "High Risk", "Low Risk"],
    "rules": [
      {
        "label": "Critical Risk",
        "default": false,
        "connective": "ANY",
        "connective_quote": "any of the following is true",
        "clauses": [
          {
            "clause_quote": "- 2 or more severe incidents",
            "min_count": 2,
            "logic": "ALL",
            "conditions": [
              {"field_id": "INCIDENT_SEVERITY", "values": ["severe"]}
            ]
          }
        ]
      },
      {
        "label": "Low Risk",
        "default": true,
        "default_quote": "Low Risk otherwise",
        "hints": []
      }
    ]
  }
}
```

## Notes

- Phase 1 is *agenda-first*: all fields, values, labels, and clause logic are
  extracted directly from the agenda text.
- Validation errors trigger automatic retries (up to the configured limit). If
  retries fail, Phase 1 proceeds with warnings and logs details to
  `debug/phase1.jsonl` in the run directory.
