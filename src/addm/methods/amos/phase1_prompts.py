"""Phase 1 Prompts: Agenda -> Agenda Spec (terms + verdict rules)."""

# =============================================================================
# Step 0: Segment agenda into blocks
# =============================================================================

SEGMENT_AGENDA_PROMPT = """You are splitting an agenda into term definitions and verdict rules.

## AGENDA
{agenda}

## OUTPUT (JSON ONLY)
{{
  "definitions_blocks": ["<text block>", "<text block>"] ,
  "verdict_rules_blocks": ["<text block>", "<text block>"]
}}

## RULES
- definitions_blocks: include ONLY parts that define terms/fields and their possible values.
- verdict_rules_blocks: include ONLY parts that define verdict/classification rules.
- If verdict rules appear in multiple sections (e.g., "Verdict Rules" and "Verdict Rules (Continued)"), include all blocks in order.
- If the agenda is XML, extract readable text for <definitions> and <verdict_rules> contents (preserve option/value IDs exactly).
- Do NOT include guidance text unless it defines enum values.

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 1: Extract verdict labels and default
# =============================================================================

EXTRACT_VERDICT_LABELS_PROMPT = """Extract verdict labels and identify the default verdict.

## VERDICT RULES TEXT
{verdict_text}

## OUTPUT (JSON ONLY)
{{
  "verdicts": ["<label>", "<label>", "<label>"],
  "default_verdict": "<label>"
}}

## RULES
- verdict labels must match the exact labels in the text (case-sensitive).
- default_verdict is the label used for "otherwise", "if none apply", or "default".

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 2: Extract verdict rule outline (per verdict)
# =============================================================================

EXTRACT_VERDICT_OUTLINE_PROMPT = """Extract the rule outline for ONE verdict label.

## VERDICT RULES TEXT
{verdict_text}

## TARGET VERDICT
{target_verdict}

## DEFAULT VERDICT
{default_verdict}

## OUTPUT (JSON ONLY)
If target verdict is the default:
{{
  "verdict": "<label>",
  "default": true,
  "hint_texts": ["<optional hints>"]
}}

If target verdict is NOT the default:
{{
  "verdict": "<label>",
  "connective": "ANY" | "ALL",
  "clause_texts": [
    "<one clause per bullet/condition>",
    "<one clause per bullet/condition>"
  ]
}}

## RULES
- connective is the BETWEEN-CLAUSES logic derived from "any/all of the following" for this verdict.
- clause_texts must be literal bullet/condition texts from the verdict rules (no paraphrase).
- Do NOT include "especially when" hints as clauses; only include them in hint_texts for default verdict.
- Use the exact verdict label (case-sensitive).

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 3: Select required term titles (per non-default verdict)
# =============================================================================

SELECT_REQUIRED_TERMS_PROMPT = """Select which term definitions are required to represent these clauses.

## DEFINITIONS
{definitions_text}

## VERDICT RULE OUTLINE (JSON)
{outline_json}

## OUTPUT (JSON ONLY)
{{
  "required_term_titles": ["<exact term title from definitions>", "<exact term title>"]
}}

## RULES
- required_term_titles MUST be copied from definition headings/titles (case-sensitive).
- Include a term if any clause implies it (e.g., "firsthand" => "Account Type").
- Do not invent terms that are not present in the definitions.

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 4: Extract a term definition (per term)
# =============================================================================

EXTRACT_TERM_DEF_PROMPT = """Extract ONE term definition as an enum.

## DEFINITIONS
{definitions_text}

## TARGET TERM TITLE
{term_title}

## OUTPUT (JSON ONLY)
{{
  "term_title": "<exact title>",
  "type": "enum",
  "values": ["<EXACT value IDs as written>", "..."],
  "descriptions": {{
    "<value>": "<description>",
    "<value>": "<description>"
  }}
}}

## RULES
- values must be the exact value IDs from the definition (do not paraphrase).
- descriptions should be copied or lightly condensed from the definition lines.
- Only output the target term.

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 5: Compile a clause (per clause)
# =============================================================================

COMPILE_CLAUSE_PROMPT = """Compile ONE clause into structured conditions using ONLY the allowed enums.

## CLAUSE TEXT
{clause_text}

## ALLOWED TERMS (JSON)
{terms_json}

## OUTPUT (JSON ONLY)
{{
  "min_count": 1,
  "conditions": [
    {{"field": "FIELD_NAME", "values": ["enum_id"]}}
  ]
}}

## RULES
- min_count must reflect the clause count (e.g., "2 or more" -> 2). If no count, use 1.
- Each condition.field must be one of the provided term fields.
- Each condition.values[] entry MUST be chosen from the allowed enum IDs for that field.
- If the clause includes multiple attributes (e.g., "moderate firsthand"), include multiple conditions.
- Do not invent new fields or values.

Output ONLY the JSON object, no explanation.
"""
