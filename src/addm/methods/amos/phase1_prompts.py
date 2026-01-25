"""Phase 1 Prompts: Prompts for Formula Seed generation.

Contains prompts for part-by-part extraction:
- EXTRACT_TERMS: Extract field/term definitions
- EXTRACT_SCORING: Extract scoring system (V2/V3 policies)
- EXTRACT_VERDICTS: Extract verdict rules

Deprecated TEXT2YAML prompt moved to: backup/phase1_hybrid.py
"""

# =============================================================================
# Part-by-Part Prompts for Structured Extraction
# =============================================================================

EXTRACT_TERMS_PROMPT = '''You are extracting term definitions from a policy query.

## QUERY SECTION

{section}

## YOUR TASK

Extract ALL terms/fields defined in this section. Each term has:
- A name (convert to UPPERCASE_WITH_UNDERSCORES)
- Possible values (the options/categories)
- Descriptions for each value

IMPORTANT: Skip sections that are guidance or context (e.g., "What to Consider", "Incident", "Overview").
These describe what to look for but are NOT fields to extract. Only extract sections that have:
- A clear term name (e.g., "Account Type", "Date Outcome")
- Bulleted values with labels and descriptions (e.g., "- **Positive**: Restaurant contributed...")

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
2. For values: Use the EXACT bolded term from each bullet point. The bolded word IS the value ID.
   Example: "- **severe**: Serious health consequences..." → use value "severe"
   Example: "- **none**: No dietary incident..." → use value "none"
3. Do NOT convert values to snake_case or add suffixes. Use them exactly as bolded.
4. Include ALL values mentioned, even neutral ones
5. Copy descriptions verbatim when possible

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
3. Recency rules: If the query mentions recency weighting, extract the time thresholds and weights

CRITICAL: The outcome_field MUST be one of the Available Terms above.
CRITICAL: base_points keys MUST use EXACT values from that term's value list.
CRITICAL: modifier field/value pairs MUST use EXACT terms and values from above.

## IMPORTANT: VALUE MAPPINGS

Many policies use a TWO-STEP mapping:
1. The policy defines term VALUES (e.g., favorable, unfavorable, neutral)
2. Then maps those values to LABELS (e.g., favorable → "Better than competitor")
3. Finally assigns POINTS to each LABEL

When extracting points, follow this chain:
- Find the severity/points table in the query
- Match each term VALUE to its corresponding LABEL
- Use the points for THAT LABEL

Example: If the query says:
  "Comparison Outcome: favorable, unfavorable, neutral"
  "Better than competitor: +5 points"
  "Worse than competitor: -5 points"

Then base_points should be:
  favorable: 5    # "favorable" maps to "Better than competitor" → 5 points
  unfavorable: -5 # "unfavorable" maps to "Worse than competitor" → -5 points
  neutral: 0

DO NOT use extreme values (like 10/-10) unless those are EXPLICITLY the points
for the standard/middle-tier outcomes. Check value_mappings carefully.

## RECENCY WEIGHTING

If the query has a "Recency weighting" section, extract it as structured rules.
Convert the natural language time periods to numeric values:
- "Within 6 months" → max_age_years: 0.5
- "Within 1 year" → max_age_years: 1.0
- "6 months to 1 year" → max_age_years: 1.0 (use upper bound)
- "1-2 years old" → max_age_years: 2.0
- "2-3 years old" → max_age_years: 3.0
- "Over X years" → max_age_years: 999 (catch-all)

Convert weight descriptions to numeric multipliers:
- "full point value" → weight: 1.0
- "three-quarter point value" or "multiply by 0.75" → weight: 0.75
- "half point value" or "multiply by 0.5" → weight: 0.5
- "quarter point value" or "multiply by 0.25" → weight: 0.25

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
  recency_rules:  # ONLY include if recency weighting is mentioned in the query
    reference_date: "2022-01-01"  # Extract from query or use default
    rules:
      - max_age_years: 0.5
        weight: 1.0
      - max_age_years: 1.0
        weight: 0.75
      - max_age_years: 2.0
        weight: 0.5
      - max_age_years: 999
        weight: 0.25
```

## RULES

1. Extract EXACT point values from the query - follow value_mappings if present
2. outcome_field: Pick the MAIN outcome field from Available Terms (usually has severity-like values)
3. base_points: Map the outcome field's VALUES to their corresponding points
4. modifiers: Use OTHER field+value pairs from Available Terms (field MUST be a term name you extracted)
5. recency_rules: ONLY include if the query has recency weighting. Extract the reference_date and all time/weight rules.
6. If no scoring system exists, output: `policy_type: count_rule_based`

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

CRITICAL INSTRUCTIONS:

1. Parse each condition to identify the field and values being checked.
   Example: "1 or more severe dietary incidents"
   → field: INCIDENT_SEVERITY, values: [severe], min_count: 1

2. Match field names to "Available Terms" above (convert to UPPERCASE_WITH_UNDERSCORES).
   Example: "dietary incidents" relates to INCIDENT_SEVERITY
   Example: "staff knowledge" relates to STAFF_KNOWLEDGE

3. Values should match the bolded IDs from the term definitions.
   Example: If condition says "severe", use value "severe" (not "severe_incident")
   Example: If condition says "poor accommodation", use value "poor"

4. Only use field names from "Available Terms" - do not invent new field names.

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
5. Default rule is indicated by words like "otherwise", "else", "by default", or "if none of the above apply". Look for these keywords to identify which verdict is the default.
6. Phrases like "especially when" or "particularly if" are HINTS, not hard conditions. The default rule should have NO conditions - just `default: true`.
7. ONLY use field names from "Available Terms" - NEVER invent new names

Output ONLY the YAML, no explanation:

```yaml
'''
