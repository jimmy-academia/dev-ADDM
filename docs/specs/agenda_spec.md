# Agenda Spec: Natural-Language Standards With Programmatic Rules

This spec defines a structured "agenda" format that can be rendered into natural-language
standards and compiled into evaluation logic. The goal is to author rules once and generate:
1) a readable standard, and 2) executable scoring/metrics.

## Goals
- Keep standards readable and real-world (not programmatic checklists).
- Ensure every term in the standard is backed by a controlled vocabulary ID.
- Ensure every rule in the standard has a machine-readable form for evaluation.

## Non-goals
- This spec does not prescribe UI or storage format.
- This spec does not define task-specific content; it defines structure.

## Required Output Format
All task responses must follow:
- `verdict` (required): one of task-specific labels.
- `evidence` (required): short evidence summary.
- `reasoning` (optional): if included, concise and non-redundant.

## Agenda Schema (Top-Level)
Represent the agenda as structured data with stable IDs.

```yaml
agenda_id: "G1a"
topic: "Allergy Safety"
labels: ["Low", "High", "Critical"]
sections:
  - id: "overview"
    title: "Purpose / overall goal"
    text: "..."
  - id: "evidence_scope"
    title: "Evidence scope"
    text: "..."
  - id: "definitions"
    title: "Definitions of terms"
    terms:
      - id: "account_type"
        title: "Account type"
        entries: [ ... ]
      - id: "incident_severity"
        title: "Incident severity"
        entries: [ ... ]
  - id: "final_rules"
    title: "Final verdict rules"
    ruleset_ref: "ruleset:G1a"
  - id: "confidence"
    title: "Confidence note"
    text: "..."
output_format:
  required_fields: ["verdict", "evidence"]
  optional_fields: ["reasoning"]
```

## Controlled Vocabulary
All definition entries MUST reference canonical terms with IDs and synonyms. Use these IDs
in rules and evaluation logic.

```yaml
vocab:
  account_type:
    firsthand:
      label: "Firsthand"
      synonyms: ["I", "my child", "we"]
    secondhand:
      label: "Secondhand"
      synonyms: ["my friend", "someone else"]
    hypothetical:
      label: "Hypothetical / preference"
      synonyms: ["I avoid", "I am allergic so I won't"]
  incident_severity:
    mild:
      label: "Mild incident"
      cues: ["discomfort", "minor symptoms"]
    moderate:
      label: "Moderate incident"
      cues: ["hives", "swelling", "antihistamine"]
    severe:
      label: "Severe incident"
      cues: ["anaphylaxis", "epipen", "ER", "hospitalization"]
```

## Rule Representation
Rules are authored once in a structured format and used for:
- Natural-language rendering.
- Evaluation logic compilation.

```yaml
rulesets:
  - id: "ruleset:G1a"
    verdicts:
      - label: "Critical"
        any_of:
          - id: "crit_severe_incident"
            when: { incident: "severe", account_type: "firsthand" }
          - id: "crit_assurance_breakdown"
            when:
              assurance: true
              account_type: "firsthand"
              incident: ["moderate", "severe"]
          - id: "crit_pattern"
            when:
              pattern:
                min_incidents: 2
                account_type: "firsthand"
                incident: ["moderate", "severe"]
      - label: "High"
        any_of:
          - id: "high_moderate_incident"
            when: { incident: "moderate", account_type: "firsthand" }
          - id: "high_assurance_mild"
            when: { assurance: true, account_type: "firsthand", incident: "mild" }
          - id: "high_staff_dismissive"
            when: { staff_response: ["refused", "dismissive"] }
          - id: "high_baseline_risk"
            when:
              cuisine_risk: "high"
              allergy_handling: "inconsistent"
      - label: "Low"
        default: true
```

### Rule Fields
- `when`: declarative predicate using vocab IDs.
- `pattern`: aggregate constraints across multiple reviews.
- `default`: fallback when no higher-risk rule matches.
- `id`: stable rule identifier for metrics and debugging.

## Recency and Weighting
Use a simple policy descriptor to keep it interpretable.

```yaml
weights:
  recency:
    enabled: true
    decay: "linear"
    half_life_days: 365
```

## Confidence Note
Confidence is not a risk label. It is derived from evidence volume.

```yaml
confidence:
  thresholds:
    limited: { max_reviews: 2 }
    substantial: { min_reviews: 5 }
```

## Rendering Rules
- Use section titles from the schema.
- Definition entries are expanded from vocab labels + clarifying text.
- Final rules section is a natural-language rendering of the ruleset.
- Always include the output format constraint (verdict + evidence + optional reasoning).

## Evaluation Compilation
The compiler should:
- Map extracted signals to vocab IDs.
- Evaluate rule predicates in priority order: Critical -> High -> Low.
- Record which rule IDs fired for auditability.
- Emit confidence based on evidence volume.

## Minimal Example (G1a)
This is a minimal payload for authoring:

```yaml
agenda_id: "G1a"
topic: "Allergy Safety"
labels: ["Low", "High", "Critical"]
vocab_ref: "vocab:allergy_safety"
ruleset_ref: "ruleset:G1a"
output_format:
  required_fields: ["verdict", "evidence"]
  optional_fields: ["reasoning"]
```
