"""Phase 1 Prompts: Agenda -> Terms + Verdict Rules.

Phase 1 does NOT do scoring. It only extracts:
1) Verdict labels + count-based rules
2) Term definitions (fields + enum values) referenced by rules
"""

# =============================================================================
# Step 0: OBSERVE (locate relevant sections)
# =============================================================================

OBSERVE_PROMPT = """You are extracting structure from a policy agenda.

## AGENDA
{agenda}

## TASK
1) Locate the section(s) that define terms/fields (definitions of terms).
2) Locate the section that defines verdict rules.
3) Extract the verdict labels as they appear in the agenda.

## OUTPUT (JSON ONLY)
{{
  "terms_block": "<COPY the exact text that defines terms/fields. Preserve bullets.>",
  "verdict_rules_block": "<COPY the exact text that defines verdict rules. Preserve bullets.>",
  "verdict_labels": ["<Verdict 1>", "<Verdict 2>", "<Verdict 3>"]
}}

## RULES
- terms_block must include only the term definitions (NOT guidance or prose context).
- verdict_rules_block must include only the verdict rules.
- verdict_labels must match the exact labels in the agenda (case-sensitive).
- If the agenda is XML, convert tags to readable text in the blocks.

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 1: Extract Verdict Rules
# =============================================================================

EXTRACT_VERDICTS_PROMPT = """You are extracting verdict rules from a policy agenda.

## VERDICT RULES BLOCK
{verdict_rules_block}

## TASK
Extract verdict labels and count-based rules.

Each rule uses one of these formats:
- Non-default rule:
  {{
    "verdict": "<label>",
    "logic": "ANY" | "ALL",
    "conditions": [
      {{"field": "FIELD_NAME", "values": ["value1", "value2"], "min_count": 2}}
    ]
  }}

- Default rule:
  {{"verdict": "<label>", "default": true}}

## OUTPUT (JSON ONLY)
{{
  "verdicts": ["Low Risk", "High Risk", "Critical Risk"],
  "rules": [
    {{
      "verdict": "Critical Risk",
      "logic": "ANY",
      "conditions": [
        {{"field": "INCIDENT_SEVERITY", "values": ["severe"], "min_count": 1}},
        {{"field": "INCIDENT_SEVERITY", "values": ["moderate"], "min_count": 2}}
      ]
    }},
    {{"verdict": "Low Risk", "default": true}}
  ]
}}

## RULES
1) Use EXACT verdict labels from the agenda (case-sensitive).
2) Extract field names and values exactly as written in the verdict rules section.
3) Normalize field names to UPPERCASE_WITH_UNDERSCORES.
4) Extract exact min_count values (e.g., "2 or more" -> 2).
5) If logic is not explicit, default to "ANY".
6) Exactly ONE rule must have "default": true.

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 2: Ground / Repair Verdict Rules (with definitions)
# =============================================================================

REFINE_VERDICTS_PROMPT = """You are verifying and correcting extracted verdict rules.

## DEFINITIONS BLOCK (for mapping field names and values)
{terms_block}

## VERDICT RULES BLOCK (ground truth text)
{verdict_rules_block}

## EXTRACTED RULES (JSON)
{verdict_json}

## TASK
Fix any missing or incorrect conditions so the rules match the verdict rules text.

Key guidance:
- If a rule clause includes qualifiers (e.g., "firsthand", "secondhand"), map them to the
  correct field names and values from DEFINITIONS BLOCK.
- If a clause combines multiple attributes (e.g., "severe firsthand incidents"), encode
  them in the SAME rule using logic: "ALL" and include both conditions with the same
  min_count.
- If a verdict has multiple bullet clauses, you may output multiple rules for the same
  verdict label (one rule per clause).
- Use EXACT verdict labels from the agenda (case-sensitive).
- Use ONLY field names and values that appear in the definitions.
- Exactly ONE rule must have "default": true.
- Phrases like "especially when" are hints, not hard conditions. Do NOT add them as rules.

## OUTPUT (JSON ONLY)
{{
  "verdicts": ["Low Risk", "High Risk", "Critical Risk"],
  "rules": [
    {{
      "verdict": "Critical Risk",
      "logic": "ALL",
      "conditions": [
        {{"field": "ACCOUNT_TYPE", "values": ["firsthand"], "min_count": 1}},
        {{"field": "INCIDENT_SEVERITY", "values": ["severe"], "min_count": 1}}
      ]
    }},
    {{
      "verdict": "Critical Risk",
      "logic": "ALL",
      "conditions": [
        {{"field": "ACCOUNT_TYPE", "values": ["firsthand"], "min_count": 2}},
        {{"field": "INCIDENT_SEVERITY", "values": ["moderate"], "min_count": 2}}
      ]
    }},
    {{"verdict": "Low Risk", "default": true}}
  ]
}}

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 3: Extract Terms
# =============================================================================

EXTRACT_TERMS_PROMPT = """You are extracting term definitions from a policy agenda.

## DEFINITIONS BLOCK
{terms_block}

## REQUIRED TERMS
{required_fields}

## TASK
Extract ONLY the term definitions listed in REQUIRED TERMS. Each term has:
- A canonical name (UPPERCASE_WITH_UNDERSCORES)
- A list of enum values (EXACT IDs as written in the agenda)
- Optional descriptions for each value

## OUTPUT (JSON ONLY)
{{
  "terms": [
    {{
      "name": "INCIDENT_SEVERITY",
      "type": "enum",
      "values": ["none", "mild", "moderate", "severe"],
      "descriptions": {{
        "none": "No allergic reaction described",
        "mild": "Minor symptoms",
        "moderate": "Visible symptoms or required medication",
        "severe": "Life-threatening reaction"
      }}
    }}
  ]
}}

## RULES
1) Use UPPERCASE_WITH_UNDERSCORES for term names.
2) For values, use the EXACT value IDs from the agenda (do not paraphrase).
3) Do NOT invent values that are not listed.
4) Skip guidance sections that are not definitions (e.g., "What to Consider").
5) Only output the terms listed in REQUIRED TERMS (ignore all other terms).

Output ONLY the JSON object, no explanation.
"""
