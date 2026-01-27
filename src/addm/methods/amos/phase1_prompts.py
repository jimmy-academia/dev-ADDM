"""Phase 1 Prompts: Prompts for Formula Seed generation.

Contains prompts for part-by-part extraction:
- OBSERVE: Format-agnostic semantic analysis (Step 0)
- EXTRACT_TERMS: Extract field/term definitions
- EXTRACT_VERDICTS: Extract verdict rules
"""

# =============================================================================
# OBSERVE Prompt: Format-Agnostic Semantic Analysis (Step 0)
# =============================================================================

OBSERVE_PROMPT = '''You are analyzing a policy agenda to understand its structure and content.
The agenda may be in any format: markdown, XML, prose, or other structured text.

## POLICY AGENDA

{agenda}

## YOUR TASK

Analyze this policy agenda and extract its semantic structure, regardless of format.
Output a structured analysis that downstream processing can use.

## OUTPUT FORMAT (JSON)

```json
{{
  "policy_type": "count_rule_based",
  "core_topic": "allergy safety",
  "verdicts": ["Low Risk", "High Risk", "Critical Risk"],
  "verdict_order": "ascending",
  "extraction_fields": [
    {{
      "name": "INCIDENT_SEVERITY",
      "description": "Severity of incident",
      "values": ["none", "mild", "moderate", "severe"]
    }},
    {{
      "name": "ACCOUNT_TYPE",
      "description": "How reviewer relates to event",
      "values": ["firsthand", "secondhand", "hypothetical"]
    }}
  ],
  "verdict_rules": [
    {{
      "verdict": "Critical Risk",
      "logic": "ANY",
      "conditions": [
        "1+ severe firsthand incidents",
        "2+ moderate firsthand incidents"
      ]
    }},
    {{
      "verdict": "High Risk",
      "precondition": "Critical Risk does not apply",
      "logic": "ANY",
      "conditions": [
        "1+ mild or moderate firsthand incidents"
      ]
    }},
    {{
      "verdict": "Low Risk",
      "default": true
    }}
  ],
  "key_concepts": ["allergy", "incident", "firsthand account", "severity"]
}}
```

## ANALYSIS GUIDELINES

1. **policy_type**: Always "count_rule_based" - rules use counts (e.g., "1 or more", "2+ reviews")

2. **verdicts**: List ALL verdict labels exactly as written (case-sensitive)

3. **verdict_order**: "ascending" if higher is better/worse in sequence, "descending" otherwise

4. **extraction_fields**: ALL fields that need to be extracted from reviews
   - Include both explicitly defined fields AND fields implied by rules
   - Use UPPERCASE_WITH_UNDERSCORES for names
   - List all possible values for each field

5. **verdict_rules**: Capture the decision logic
   - Include conditions in natural language (will be parsed later)
   - Note preconditions like "if X does not apply"
   - Mark exactly one rule as "default": true

6. **key_concepts**: Main themes and keywords relevant to this policy

## IMPORTANT

- Be format-agnostic: extract the same information whether input is markdown, XML, or prose
- If the format uses XML tags, interpret their semantic meaning
- If the format is prose, identify structure from language patterns
- Extract EXACT verdict names and field values (case-sensitive for verdicts)

Output ONLY the JSON, no explanation:

```json
'''

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

EXTRACT_VERDICTS_PROMPT = '''You are extracting verdict rules from a policy query.

## QUERY SECTION

{section}

## AVAILABLE TERMS (use ONLY these field names)

{context}

## YOUR TASK

Extract the verdict rules that determine the final outcome.

CRITICAL INSTRUCTIONS:

1. Parse each condition to identify the field, values, and COUNT from the text.
   Examples of count extraction:
   - "1 or more severe incidents" → min_count: 1
   - "2 or more moderate incidents" → min_count: 2
   - "3+ negative reviews" → min_count: 3
   - "at least 5 complaints" → min_count: 5

2. Match field names to "Available Terms" above (convert to UPPERCASE_WITH_UNDERSCORES).
   Example: "dietary incidents" relates to INCIDENT_SEVERITY
   Example: "staff knowledge" relates to STAFF_KNOWLEDGE

3. Values should match the bolded IDs from the term definitions.
   Example: If condition says "severe", use value "severe" (not "severe_incident")
   Example: If condition says "poor accommodation", use value "poor"

4. Only use field names from "Available Terms" - do not invent new field names.

## OUTPUT FORMAT (YAML)

```yaml
verdicts: [Verdict1, Verdict2, Verdict3]

rules:
  # Example with ANY logic (OR conditions - at least ONE must be true)
  - verdict: Verdict1
    logic: ANY
    conditions:
      - field: INCIDENT_SEVERITY
        values: [severe]
        min_count: 1
      - field: INCIDENT_SEVERITY
        values: [moderate]
        min_count: 2

  # Example with ALL logic (AND conditions - ALL must be true simultaneously)
  - verdict: Verdict2
    logic: ALL
    conditions:
      - field: INCIDENT_SEVERITY
        values: [severe]
        min_count: 1
      - field: INCIDENT_SEVERITY
        values: [moderate]
        min_count: 1

  - verdict: Verdict3
    default: true
```

## RULES

1. Use EXACT verdict labels from the query (copy character-for-character)
2. Extract EXACT count numbers from the text - do NOT default to 1
   - "1 or more" → min_count: 1
   - "2 or more" → min_count: 2
   - Read the actual number from the condition text
3. Exactly ONE rule must have `default: true`
4. Default rule is indicated by words like "otherwise", "else", "by default", or "if none of the above apply". Look for these keywords to identify which verdict is the default.
5. Phrases like "especially when" or "particularly if" are HINTS, not hard conditions. The default rule should have NO conditions - just `default: true`.
6. ONLY use field names from "Available Terms" - NEVER invent new names
7. LOGIC DETECTION:
   - If text says "if **all** of the following are true", "AND", or "both...and" → use `logic: ALL`
   - If text says "if **any** of the following is true", "OR", or "either...or" → use `logic: ANY`
   - Default to `logic: ANY` if not specified

Output ONLY the YAML, no explanation:

```yaml
'''
