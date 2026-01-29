"""Phase 1 Prompts: Agenda -> Agenda Spec (terms + verdict rules)."""

# =============================================================================
# Step 0: Locate blocks (anchors only)
# =============================================================================

LOCATE_BLOCKS_PROMPT = """Identify blocks in the agenda that contain:
1) term definitions (fields + allowed values)
2) verdict labels and verdict rules (classification logic, including any label list)

Return JSON ONLY.

AGENDA:
{agenda}

OUTPUT JSON:
{{
  "definitions_blocks": [{{"start_quote":"...","end_quote":"..."}}],
  "verdict_blocks": [{{"start_quote":"...","end_quote":"..."}}]
}}

RULES:
- start_quote and end_quote MUST be exact substrings copied from the agenda (verbatim).
- Choose anchors that appear exactly once in the agenda.
- Each block is the substring from start_quote to end_quote (inclusive).
- definitions_blocks: include only blocks needed to define fields + allowed values.
- verdict_blocks: include only blocks needed to define labels + classification rules.
- Do NOT paraphrase any agenda text.

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 1: Extract verdict rules skeleton
# =============================================================================

EXTRACT_VERDICT_RULES_PROMPT = """Extract verdict labels and verdict rules from the text.

VERDICT TEXT:
{verdict_text}

Return JSON ONLY.

OUTPUT JSON SCHEMA:
{{
  "labels": ["..."],
  "default_label": "...",
  "order": ["..."],
  "rules": [
    {{
      "label": "...",
      "default": false,
      "connective": "ANY" | "ALL",
      "connective_quote": "...",
      "clause_quotes": ["..."]
    }},
    {{
      "label": "...",
      "default": true,
      "default_quote": "...",
      "hints": [{{"hint_quote":"..."}}]
    }}
  ]
}}

RULES:
- All quotes MUST be exact substrings copied from VERDICT TEXT (verbatim).
- labels must include every verdict label used by the rules (case-sensitive).
- default_label is the “otherwise / if none apply” label.
- order is the evaluation order; default_label must be last.
- For each non-default rule:
  - connective is the between-clauses connective for that verdict (ANY vs ALL).
  - connective_quote must be a substring that justifies connective.
  - clause_quotes are the literal condition/bullet texts for that verdict.
- For the default rule:
  - default_quote must be the exact substring indicating default/otherwise.
  - hints are optional exact substrings; do not treat hints as required clauses.

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 2: Select term blocks (anchors only)
# =============================================================================

SELECT_TERM_BLOCKS_PROMPT = """Select the term definitions needed to interpret the NON-DEFAULT verdict rules.

DEFINITIONS TEXT:
{definitions_text}

VERDICT RULES (JSON):
{verdict_rules_json}

Return JSON ONLY.

OUTPUT JSON:
{{
  "terms": [
    {{"term_title":"...","start_quote":"...","end_quote":"..."}}
  ]
}}

RULES:
- term_title must be the name of a FIELD being defined (not an enum option).
- start_quote and end_quote MUST be exact substrings copied from DEFINITIONS TEXT (verbatim).
- Choose anchors that appear exactly once in the agenda.
- Each (start_quote..end_quote) slice must include the term title and all of its allowed values.
- Only include terms needed by non-default verdict rules.
- Do NOT invent terms that are not defined in DEFINITIONS TEXT.

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 2b: Select overview/guidance blocks (anchors only)
# =============================================================================

SELECT_OVERVIEW_BLOCKS_PROMPT = """Select the overview/guidance blocks that describe
what to consider or what counts as relevant evidence in reviews.

DEFINITIONS TEXT:
{definitions_text}

VERDICT RULES (JSON):
{verdict_rules_json}

Return JSON ONLY.

OUTPUT JSON:
{{
  "overview_blocks": [
    {{"title":"...","start_quote":"...","end_quote":"..."}}
  ]
}}

RULES:
- overview_blocks may be empty if no guidance block exists.
- Choose anchors that appear exactly once in DEFINITIONS TEXT.
- Each block is the substring from start_quote to end_quote (inclusive).
- Focus on guidance like "What to Consider", "Guidance", "Relevant signals".
- EXCLUDE term enum blocks (field/value lists).
- EXCLUDE verdict rules or label thresholds.
- start_quote and end_quote MUST be exact substrings copied from DEFINITIONS TEXT (verbatim).

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 3: Extract term enum (per term)
# =============================================================================

EXTRACT_TERM_ENUM_PROMPT = """Extract ONE term definition as an enum.

TERM TITLE:
{term_title}

TERM BLOCK (verbatim):
{term_block}

Return JSON ONLY.

OUTPUT JSON:
{{
  "term_title": "...",
  "values": [
    {{"source_value":"...","description":"...","value_quote":"..."}}
  ]
}}

RULES:
- source_value MUST be copied exactly from TERM BLOCK (case-sensitive).
- value_quote MUST be an exact substring copied from TERM BLOCK, and must contain source_value.
- Include ALL enum options defined for the term.
- Do NOT invent values or fields.

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 2c: Extract overview/guidance text (summary)
# =============================================================================

EXTRACT_OVERVIEW_PROMPT = """Summarize the guidance/overview text into concise,
review-language signals. Focus on what reviewers actually say.

OVERVIEW TEXT (verbatim):
{overview_text}

TERM SCHEMA (JSON):
{terms_json}

Return JSON ONLY.

OUTPUT JSON:
{{
  "overview_text": "..."
}}

RULES:
- Do NOT include verdict thresholds or label ordering.
- Do NOT use rubric phrasing like "incidents are reported" or "1 or more".
- Use review-like phrases and examples from the term schema where helpful.
- Keep it concise (<= 8 sentences).

Output ONLY the JSON object, no explanation.
"""


# =============================================================================
# Step 4: Compile a clause (per clause)
# =============================================================================

COMPILE_CLAUSE_PROMPT = """Compile ONE clause into structured conditions.

CLAUSE (verbatim):
{clause_quote}

ALLOWED TERMS (JSON):
{terms_json}

Return JSON ONLY.

OUTPUT JSON:
{{
  "min_count": 1,
  "logic": "ANY" | "ALL",
  "conditions": [
    {{"field_id":"FIELD_ID","values":["value_id"]}}
  ]
}}

RULES:
- min_count reflects any explicit count (e.g., "2 or more" -> 2). If no count, use 1.
- field_id MUST be one of the allowed field_id values.
- Each values[] entry MUST be one of the allowed value_id values for that field_id.
- logic describes how to combine multiple conditions inside this clause.
- Do NOT invent fields or values.

Output ONLY the JSON object, no explanation.
"""
