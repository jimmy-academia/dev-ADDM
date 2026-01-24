"""Phase 1 Prompts: All prompts for Formula Seed generation.

Contains prompts for:
- TEXT2YAML (Hybrid approach)
- EXTRACT_TERMS, EXTRACT_SCORING, EXTRACT_VERDICTS (Parts approach - recommended)

Legacy Plan-and-Act prompts are in backup/phase1_prompts.py.

Separated from logic for maintainability and easier prompt iteration.
"""

# =============================================================================
# TEXT2YAML Prompt for Hybrid Approach
# =============================================================================

TEXT2YAML_PROMPT = '''You are a policy specification expert. Analyze this query and generate a PolicyYAML specification.

## QUERY/AGENDA

{agenda}

## YOUR TASK

Read the query carefully and produce a PolicyYAML that captures:
1. What fields/signals to extract from reviews
2. What values each field can have
3. What verdict rules determine the outcome

## OUTPUT FORMAT (YAML)

```yaml
policy_type: count_rule_based  # use "count_rule_based" for counting occurrences, "scoring" for point systems

task_name: "<brief description of what this policy evaluates>"

terms:
  - name: <FIELD_NAME_IN_CAPS>
    values: [<value1>, <value2>, ...]  # all possible values for this field
    descriptions:
      <value1>: "<what this value means>"
      <value2>: "<what this value means>"

verdicts: [<verdict1>, <verdict2>, <verdict3>]  # exactly the verdicts from the query

rules:
  # Non-default rules: checked in order
  - verdict: <verdict_label>
    logic: ANY  # ANY = at least one condition; ALL = all conditions must be true
    conditions:
      - field: <FIELD_NAME>
        values: [<value1>, <value2>]  # which values count toward this verdict
        min_count: <N>  # how many needed (extract from query, e.g., "2 or more" → 2)

  # Default rule: applied when no other rules match
  - verdict: <default_verdict>
    default: true
```

## CRITICAL RULES

1. **FIELD NAMES**: Use UPPERCASE_WITH_UNDERSCORES (e.g., PRICE_PERCEPTION, INCIDENT_SEVERITY)

2. **VALUES**: List ALL possible values for each field. These become enum options for extraction.

3. **VERDICTS**: Use EXACT verdict labels from the query. Copy them character-for-character.

4. **THRESHOLDS**: Extract exact numbers from the query:
   - "2 or more" → min_count: 2
   - "at least 10" → min_count: 10
   - "multiple" → min_count: 2 (standard interpretation)

5. **RULE ORDER**: More specific rules first, default rule last.

6. **DEFAULT RULE**: Exactly ONE rule must have `default: true`

7. **CONDITION VALUES**: The `values` in a condition specify WHICH enum values to count.

## EXAMPLE

For query: "Recommend restaurants with good value. Recommend if 10+ reviews say it's a bargain or good value. Not Recommended if 5+ say it's overpriced or a ripoff. Otherwise Neutral."

```yaml
policy_type: count_rule_based

task_name: "Evaluate restaurant value for money"

terms:
  - name: PRICE_PERCEPTION
    values: [bargain, good_value, fair, overpriced, ripoff]
    descriptions:
      bargain: "Exceptional value, prices much lower than expected for quality"
      good_value: "Good value for money, reasonable prices"
      fair: "Average value, neither good nor bad deal"
      overpriced: "Prices higher than expected for quality"
      ripoff: "Extremely overpriced, poor value"

verdicts: [Recommend, Not Recommended, Neutral]

rules:
  - verdict: Recommend
    logic: ANY
    conditions:
      - field: PRICE_PERCEPTION
        values: [bargain, good_value]
        min_count: 10

  - verdict: Not Recommended
    logic: ANY
    conditions:
      - field: PRICE_PERCEPTION
        values: [overpriced, ripoff]
        min_count: 5

  - verdict: Neutral
    default: true
```

## NOW GENERATE

Analyze the query above and output ONLY the PolicyYAML (no explanation before or after):

```yaml
'''


# =============================================================================
# Part-by-Part Prompts for Structured Extraction (Recommended)
# =============================================================================

EXTRACT_TERMS_PROMPT = '''You are extracting term definitions from a policy query.

## QUERY SECTION

{section}

## YOUR TASK

Extract ALL terms/fields defined in this section. Each term has:
- A name (convert to UPPERCASE_WITH_UNDERSCORES)
- Possible values (the options/categories)
- Descriptions for each value

## OUTPUT FORMAT (YAML)

```yaml
terms:
  - name: FIELD_NAME
    values: [value1, value2, value3]
    descriptions:
      value1: "description of what this value means"
      value2: "description of what this value means"
```

## RULES

1. Use UPPERCASE_WITH_UNDERSCORES for field names (e.g., PRICE_PERCEPTION, INCIDENT_SEVERITY)
2. Use lowercase_with_underscores for values (e.g., good_value, very_poor)
3. Include ALL values mentioned, even neutral ones
4. Copy descriptions verbatim when possible

Output ONLY the YAML, no explanation:

```yaml
'''

EXTRACT_SCORING_PROMPT = '''You are extracting a scoring system from a policy query.

## QUERY SECTION

{section}

## AVAILABLE TERMS AND VALUES (USE ONLY THESE)

{terms_summary}

## YOUR TASK

Extract the scoring rules using ONLY the field names and values listed above.

1. Base points: What points are assigned for each value of the outcome field
2. Modifiers: Additional points for other field values

CRITICAL: The outcome_field MUST be one of the Available Terms above.
CRITICAL: base_points keys MUST use EXACT values from that term's value list.
CRITICAL: modifier field/value pairs MUST use EXACT terms and values from above.

## OUTPUT FORMAT (YAML)

```yaml
policy_type: scoring

scoring:
  outcome_field: FIELD_NAME  # MUST be from Available Terms
  base_points:  # Keys MUST be exact values from that field
    value1: 10
    value2: 5
    value3: 0
    value4: -5
  modifiers:
    - field: OTHER_FIELD  # MUST be from Available Terms
      value: some_value   # MUST be from that field's values
      points: 3
```

## RULES

1. Extract EXACT point values from the query (e.g., "+10 points", "-5 points")
2. outcome_field: Pick the MAIN outcome field from Available Terms (usually has severity-like values)
3. base_points: Map the outcome field's VALUES to points
4. modifiers: Use OTHER field+value pairs from Available Terms
5. If no scoring system exists, output: `policy_type: count_rule_based`

Output ONLY the YAML, no explanation:

```yaml
'''

EXTRACT_VERDICTS_PROMPT = '''You are extracting verdict rules from a policy query.

## QUERY SECTION

{section}

## AVAILABLE TERMS (use ONLY these field names)

{context}

## YOUR TASK

Extract the verdict rules that determine the final outcome.

CRITICAL: You may ONLY use field names listed in "Available Terms" above.
Do NOT invent new field names.

## OUTPUT FORMAT (YAML)

For SCORING policies (point-based):
```yaml
verdicts: [Verdict1, Verdict2, Verdict3]

rules:
  - verdict: Verdict1
    condition: score >= 44
  - verdict: Verdict2
    condition: score <= -13
  - verdict: Verdict3
    default: true
```

For COUNT-BASED policies:
```yaml
verdicts: [Verdict1, Verdict2, Verdict3]

rules:
  - verdict: Verdict1
    logic: ANY
    conditions:
      - field: FIELD_NAME
        values: [value1, value2]
        min_count: 10
  - verdict: Verdict2
    logic: ALL
    conditions:
      - field: FIELD_NAME
        values: [value3]
        min_count: 5
  - verdict: Verdict3
    default: true
```

## RULES

1. Use EXACT verdict labels from the query (copy character-for-character)
2. Extract EXACT threshold numbers (e.g., "44 or higher" → >= 44)
3. For count-based: min_count is how many reviews needed
4. Exactly ONE rule must have `default: true`
5. Default rule should be the "neutral" or "middle" verdict
6. ONLY use field names from "Available Terms" - NEVER invent new names

Output ONLY the YAML, no explanation:

```yaml
'''
