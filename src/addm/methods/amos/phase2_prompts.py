"""Phase 2 Prompts: Extraction prompt builders and response parsers.

Prompt construction and response parsing for the Formula Seed interpreter.
Separated from logic for maintainability and easier prompt iteration.
"""

import json
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Verdict Label Normalization
# =============================================================================

# Standard verdict labels (GT format) vs common LLM abbreviations
# Maps various LLM-generated labels to canonical GT format
VERDICT_LABEL_MAP = {
    # =========================================================================
    # G1 - Customer Safety (Risk-based)
    # Verdicts: Low Risk, High Risk, Critical Risk
    # =========================================================================
    "critical": "Critical Risk",
    "critical risk": "Critical Risk",
    "Critical": "Critical Risk",
    "Critical Risk": "Critical Risk",
    "CRITICAL": "Critical Risk",
    "CRITICAL_RISK": "Critical Risk",

    "high": "High Risk",
    "high risk": "High Risk",
    "High": "High Risk",
    "High Risk": "High Risk",
    "HIGH": "High Risk",
    "HIGH_RISK": "High Risk",

    "low": "Low Risk",
    "low risk": "Low Risk",
    "Low": "Low Risk",
    "Low Risk": "Low Risk",
    "LOW": "Low Risk",
    "LOW_RISK": "Low Risk",

    # =========================================================================
    # G2 - Customer Experience (Recommendation-based)
    # Verdicts: Not Recommended, Recommended
    # =========================================================================
    "recommended": "Recommended",
    "Recommended": "Recommended",
    "RECOMMENDED": "Recommended",

    "not recommended": "Not Recommended",
    "not_recommended": "Not Recommended",
    "Not Recommended": "Not Recommended",
    "NOT_RECOMMENDED": "Not Recommended",
    "NotRecommended": "Not Recommended",

    # =========================================================================
    # G3 - Customer Value (Value-based)
    # Verdicts: Poor Value, Fair Value, Great Value
    # =========================================================================
    "great value": "Great Value",
    "great_value": "Great Value",
    "Great Value": "Great Value",
    "GREAT_VALUE": "Great Value",
    "excellent value": "Great Value",

    "fair value": "Fair Value",
    "fair_value": "Fair Value",
    "Fair Value": "Fair Value",
    "FAIR_VALUE": "Fair Value",
    "moderate value": "Fair Value",

    "poor value": "Poor Value",
    "poor_value": "Poor Value",
    "Poor Value": "Poor Value",
    "POOR_VALUE": "Poor Value",
    "bad value": "Poor Value",

    # =========================================================================
    # G4 - Owner Operations (Performance-based)
    # Verdicts: Needs Improvement, Satisfactory, Excellent
    # =========================================================================
    "excellent": "Excellent",
    "Excellent": "Excellent",
    "EXCELLENT": "Excellent",

    "satisfactory": "Satisfactory",
    "Satisfactory": "Satisfactory",
    "SATISFACTORY": "Satisfactory",
    "adequate": "Satisfactory",

    "needs improvement": "Needs Improvement",
    "needs_improvement": "Needs Improvement",
    "Needs Improvement": "Needs Improvement",
    "NEEDS_IMPROVEMENT": "Needs Improvement",
    "NeedsImprovement": "Needs Improvement",
    "poor": "Needs Improvement",
    "Poor": "Needs Improvement",

    # =========================================================================
    # G5 - Owner Performance (Performance-based - same as G4)
    # Verdicts: Needs Improvement, Satisfactory, Excellent
    # (Already covered above)
    # =========================================================================

    # =========================================================================
    # G6 - Owner Strategy (Strategy-based - varies by topic)
    # =========================================================================

    # G6 Uniqueness: Generic, Differentiated, Highly Unique
    "highly unique": "Highly Unique",
    "highly_unique": "Highly Unique",
    "Highly Unique": "Highly Unique",
    "HIGHLY_UNIQUE": "Highly Unique",
    "unique": "Highly Unique",

    "differentiated": "Differentiated",
    "Differentiated": "Differentiated",
    "DIFFERENTIATED": "Differentiated",

    "generic": "Generic",
    "Generic": "Generic",
    "GENERIC": "Generic",
    "common": "Generic",

    # G6 Comparison: Weaker, Comparable, Stronger
    "stronger": "Stronger",
    "Stronger": "Stronger",
    "STRONGER": "Stronger",
    "better": "Stronger",

    "comparable": "Comparable",
    "Comparable": "Comparable",
    "COMPARABLE": "Comparable",
    "similar": "Comparable",

    "weaker": "Weaker",
    "Weaker": "Weaker",
    "WEAKER": "Weaker",
    "worse": "Weaker",

    # G6 Loyalty: Low Loyalty, Moderate Loyalty, High Loyalty
    "high loyalty": "High Loyalty",
    "high_loyalty": "High Loyalty",
    "High Loyalty": "High Loyalty",
    "HIGH_LOYALTY": "High Loyalty",
    "strong loyalty": "High Loyalty",

    "moderate loyalty": "Moderate Loyalty",
    "moderate_loyalty": "Moderate Loyalty",
    "Moderate Loyalty": "Moderate Loyalty",
    "MODERATE_LOYALTY": "Moderate Loyalty",
    "medium loyalty": "Moderate Loyalty",

    "low loyalty": "Low Loyalty",
    "low_loyalty": "Low Loyalty",
    "Low Loyalty": "Low Loyalty",
    "LOW_LOYALTY": "Low Loyalty",
    "weak loyalty": "Low Loyalty",
}


def normalize_verdict_label(label):
    """Normalize a verdict label to GT format.

    Args:
        label: Raw verdict label from LLM (string, bool, int, or None)

    Returns:
        Normalized verdict label (e.g., "Critical Risk") or original value if not a string
    """
    # Handle non-string types (booleans, numbers, None)
    if not isinstance(label, str):
        return label

    if not label:
        return label

    # Try exact match first
    if label in VERDICT_LABEL_MAP:
        return VERDICT_LABEL_MAP[label]

    # Try case-insensitive match
    label_lower = label.lower().strip()
    for key, value in VERDICT_LABEL_MAP.items():
        if key.lower() == label_lower:
            return value

    # Return as-is if no mapping found
    return label


# =============================================================================
# Enum Value Normalization
# =============================================================================

# Common variations that map to canonical enum values
SEVERITY_NORMALIZATION_MAP = {
    "none": "none", "no": "none", "n/a": "none", "na": "none",
    "not applicable": "none", "no incident": "none", "no reaction": "none",
    "no allergy": "none", "no allergy incident": "none", "no allergic reaction": "none",
    "no relevant incident": "none", "not relevant": "none", "irrelevant": "none",
    "no issue": "none", "no issues": "none", "no problem": "none", "no problems": "none",
    "nothing": "none", "absent": "none", "no symptoms": "none",
    "mild": "mild", "minor": "mild", "slight": "mild", "minimal": "mild", "low": "mild",
    "mild incident": "mild", "mild reaction": "mild", "minor incident": "mild",
    "minor reaction": "mild", "discomfort": "mild",
    "moderate": "moderate", "medium": "moderate", "moderate incident": "moderate",
    "moderate reaction": "moderate", "some symptoms": "moderate", "noticeable": "moderate",
    "severe": "severe", "serious": "severe", "critical": "severe", "high": "severe",
    "extreme": "severe", "severe incident": "severe", "severe reaction": "severe",
    "life-threatening": "severe", "life threatening": "severe", "emergency": "severe",
    "anaphylaxis": "severe", "anaphylactic": "severe", "hospitalization": "severe",
    "hospital": "severe",
}


def normalize_enum_value(value: Any, expected_values: List[str]) -> Any:
    """Normalize an enum value to match expected canonical values.

    Handles common variations in LLM outputs:
    - "no reaction" -> "none"
    - "no allergy incident described" -> "none"
    - Keyword-based fuzzy matching for severity fields

    Args:
        value: The value to normalize (from LLM extraction)
        expected_values: List of valid enum values (e.g., ["none", "mild", "moderate", "severe"])

    Returns:
        Normalized value if a match is found, otherwise original value
    """
    if value is None:
        return value

    value_str = str(value).strip().lower()

    # 1. Exact match
    for expected in expected_values:
        if value_str == expected.lower():
            return expected

    # 2. Check normalization map
    if value_str in SEVERITY_NORMALIZATION_MAP:
        normalized = SEVERITY_NORMALIZATION_MAP[value_str]
        for expected in expected_values:
            if normalized == expected.lower():
                return expected

    # 3. Keyword-based matching for severity enums
    expected_lower = {e.lower() for e in expected_values}
    is_severity_enum = expected_lower & {"none", "mild", "moderate", "severe"}

    if is_severity_enum:
        none_indicators = ["no ", "none", "n/a", "not ", "absent", "nothing", "irrelevant"]
        severity_keywords = ["mild", "moderate", "severe", "serious", "critical", "emergency", "life"]

        has_none_indicator = any(ind in value_str for ind in none_indicators)
        has_severity_keyword = any(kw in value_str for kw in severity_keywords)

        if has_none_indicator and not has_severity_keyword:
            if "none" in expected_lower:
                return "none"

        if "life-threatening" in value_str or "life threatening" in value_str:
            if "severe" in expected_lower:
                return "severe"
        if "anaphyla" in value_str:
            if "severe" in expected_lower:
                return "severe"
        if "emergency" in value_str or "hospital" in value_str:
            if "severe" in expected_lower:
                return "severe"
        if "severe" in value_str or "serious" in value_str or "critical" in value_str:
            if "severe" in expected_lower:
                return "severe"
        if "moderate" in value_str or "medium" in value_str:
            if "moderate" in expected_lower:
                return "moderate"
        if "mild" in value_str or "minor" in value_str or "slight" in value_str:
            if "mild" in expected_lower:
                return "mild"

    return value


def validate_enum_strict(
    value: Any,
    expected_values: List[str],
    field_name: str,
) -> Tuple[bool, str, Any]:
    """Strictly validate an enum value after normalization.

    Unlike normalize_enum_value which tries to recover bad values,
    this function rejects values that don't match expected enums.

    Args:
        value: The value to validate (already normalized)
        expected_values: List of valid enum values
        field_name: Name of the field (for error messages)

    Returns:
        Tuple of (is_valid, error_message, normalized_value)
        - is_valid: True if value matches an expected enum
        - error_message: Empty if valid, otherwise describes the rejection
        - normalized_value: The normalized value (even if invalid)
    """
    if value is None:
        return True, "", None

    # First normalize
    normalized = normalize_enum_value(value, expected_values)

    # Check if normalized value is in expected values (case-insensitive)
    expected_lower = {v.lower(): v for v in expected_values}
    normalized_lower = str(normalized).lower().strip()

    if normalized_lower in expected_lower:
        return True, "", expected_lower[normalized_lower]

    # Value doesn't match any expected enum - reject
    return False, f"invalid_{field_name}:{value}->'{normalized}'_not_in_{expected_values}", normalized


# =============================================================================
# Single Review Extraction Prompt
# =============================================================================

def build_extraction_prompt(
    fields: List[Dict[str, Any]],
    review_text: str,
    review_id: str,
    task_name: str = "relevant information",
    extraction_guidelines: Optional[str] = None,
) -> str:
    """Build extraction prompt from Formula Seed field definitions.

    Args:
        fields: List of field definitions from Formula Seed
        review_text: The review text to analyze
        review_id: Review identifier
        task_name: Human-readable task description (e.g., "allergy safety", "romantic dining")
        extraction_guidelines: Optional task-specific extraction guidelines

    Returns:
        Extraction prompt string
    """
    field_defs = []
    field_names = []
    for field in fields:
        name = field["name"]
        field_names.append(name)
        ftype = field.get("type", "enum")
        values = field.get("values", {})

        if ftype == "enum" and values:
            value_list = ", ".join(values.keys())
            field_defs.append(f"{name.upper()} // MUST be exactly one of: {{{value_list}}}")
            field_defs.append("  IMPORTANT: Use ONLY these exact values. Do NOT paraphrase or use synonyms.")
            for value, description in values.items():
                field_defs.append(f'  - "{value}": {description}')
            field_defs.append("")
        elif ftype == "int":
            field_defs.append(f"{name.upper()} // integer value")
            if values:
                for value, description in values.items():
                    field_defs.append(f"  - {value}: {description}")
            field_defs.append("")
        elif ftype == "float":
            field_defs.append(f"{name.upper()} // numeric value (0.0-1.0 or as specified)")
            field_defs.append("")
        elif ftype == "bool":
            field_defs.append(f"{name.upper()} // true or false")
            field_defs.append("")

    field_definitions = "\n".join(field_defs)

    # Build explicit field list for output constraint
    allowed_fields_list = ", ".join(f'"{name}"' for name in field_names)

    # Build output format with example values from enum definitions
    output_field_examples = []
    for field in fields:
        name = field["name"]
        ftype = field.get("type", "enum")
        values = field.get("values", {})
        if ftype == "enum" and values:
            # Use first enum value as example
            example_value = list(values.keys())[0]
            output_field_examples.append(f'"{name}": "{example_value}"')
        else:
            output_field_examples.append(f'"{name}": <value>')
    output_fields = ", ".join(output_field_examples)

    # Use task-specific guidelines if provided, otherwise generate generic ones
    if extraction_guidelines:
        guidelines_section = f"EXTRACTION GUIDELINES:\n{extraction_guidelines}"
    else:
        guidelines_section = """EXTRACTION GUIDELINES:
- Only extract fields that are explicitly mentioned or strongly implied in the review
- If a field is not discussed or cannot be determined, use the default/none value
- Focus on factual content, not inferences"""

    prompt = f"""Extract the following fields from this review for {task_name}.

{guidelines_section}

STRICT OUTPUT RULES:
1. ONLY output the fields listed below. Do NOT add any additional fields (no "staff_response", "cuisine_risk", "additional_context", etc.).
2. For enum fields, you MUST use EXACTLY one of the listed values. Do NOT paraphrase or use synonyms (e.g., use "none" not "no reaction" or "no incident").
3. The ONLY allowed fields in your output are: "review_id", "is_relevant", "supporting_quote", {allowed_fields_list}

FIELD DEFINITIONS:
{field_definitions}

REVIEW TEXT:
{review_text}

WHEN TO SET is_relevant: false
- If the review does NOT contain any ACTUAL evidence related to {task_name}
- If the review only mentions ingredients, menu items, or general topics without describing an actual incident or relevant experience
- If you cannot find a direct quote that demonstrates evidence for the task
- If the review is just a general quality statement without the specific type of evidence this task requires
- For COMPARISON tasks: the review must EXPLICITLY mention another restaurant/competitor by name or reference (e.g., "better than X", "compared to the place down the street"). Generic negative/positive reviews are NOT comparisons.
- For LOYALTY tasks: the review must mention return visits, regular customer status, or intentions to return. One-time visits are NOT loyalty signals.
- For UNIQUENESS tasks: the review must describe distinctive features. Generic praise/criticism is NOT uniqueness evidence.
- When in doubt, set is_relevant: false

WHEN TO SET is_relevant: true
- ONLY if the review contains ACTUAL evidence (a real incident, experience, or observation directly related to {task_name})
- You must be able to quote specific text that demonstrates the evidence
- The quoted text must DIRECTLY support the task type, not just be related to the general topic

CRITICAL - EVIDENCE REQUIREMENT:
You MUST include a "supporting_quote" field with the EXACT text from the review that supports your extraction.
- Copy the relevant sentence(s) verbatim from the review
- The quote must exist in the review text exactly as written
- If you cannot quote specific evidence, set is_relevant: false

Output JSON only. If the review does not contain relevant evidence for this task, output:
{{"is_relevant": false}}

Otherwise output ONLY these fields (no extra fields):
{{
  "review_id": "{review_id}",
  "is_relevant": true,
  "supporting_quote": "<EXACT text from review>",
  {output_fields}
}}"""

    return prompt


def parse_extraction_response(response: str) -> Dict[str, Any]:
    """Parse LLM extraction response to JSON."""
    response = response.strip()

    # Handle markdown code blocks
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end > start:
            response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            response = response[start:end].strip()

    # Find JSON object boundaries
    brace_start = response.find("{")
    brace_end = response.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        response = response[brace_start : brace_end + 1]

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        # Return non-relevant for parse failures
        return {"is_relevant": False, "_parse_error": str(e)}


# =============================================================================
# Batched Extraction Prompt (Process multiple reviews per LLM call)
# =============================================================================

def build_batch_extraction_prompt(
    fields: List[Dict[str, Any]],
    reviews: List[Dict[str, Any]],
    task_name: str = "relevant information",
    extraction_guidelines: Optional[str] = None,
) -> str:
    """Build extraction prompt for multiple reviews in a single call.

    Args:
        fields: List of field definitions from Formula Seed
        reviews: List of review dicts with 'text' and 'review_id'
        task_name: Human-readable task description
        extraction_guidelines: Optional task-specific guidelines

    Returns:
        Batch extraction prompt string
    """
    field_defs = []
    field_names = []
    for field in fields:
        name = field["name"]
        field_names.append(name)
        ftype = field.get("type", "enum")
        values = field.get("values", {})

        if ftype == "enum" and values:
            value_list = ", ".join(values.keys())
            field_defs.append(f"{name.upper()} / MUST be exactly one of: {{{value_list}}}")
            for value, description in values.items():
                field_defs.append(f'  - "{value}": {description}')
            field_defs.append("")
        elif ftype == "string":
            desc = field.get("description", "text description")
            field_defs.append(f"{name.upper()} // {desc}")
            field_defs.append("")

    field_definitions = "\n".join(field_defs)
    allowed_fields_list = ", ".join(f'"{name}"' for name in field_names)

    # Build reviews section
    reviews_section = []
    for r in reviews:
        reviews_section.append(f"--- REVIEW {r['review_id']} ---")
        reviews_section.append(r.get("text", ""))
        reviews_section.append("")
    reviews_text = "\n".join(reviews_section)

    # Guidelines
    if extraction_guidelines:
        guidelines_section = f"EXTRACTION GUIDELINES:\n{extraction_guidelines}"
    else:
        guidelines_section = """EXTRACTION GUIDELINES:
- Only extract fields that are explicitly mentioned or strongly implied
- If a field cannot be determined, use the default/none value
- Focus on factual content, not inferences"""

    prompt = f"""Extract fields from MULTIPLE reviews for {task_name}.

{guidelines_section}

FIELD DEFINITIONS:
{field_definitions}

STRICT RULES:
1. Output a JSON array with one object per review
2. For enum fields, use EXACTLY one of the listed values
3. Include "supporting_quote" with EXACT text from the review
4. If review has no relevant evidence, output {{"review_id": "...", "is_relevant": false}}

REVIEWS TO PROCESS:
{reviews_text}

Output a JSON array with exactly {len(reviews)} objects, one per review:
[
  {{"review_id": "<id1>", "is_relevant": true/false, "supporting_quote": "...", {allowed_fields_list}}},
  ...
]

Output ONLY the JSON array:"""

    return prompt


def parse_batch_extraction_response(response: str, review_ids: List[str]) -> List[Dict[str, Any]]:
    """Parse batch extraction response to list of extractions.

    Validates review_ids against expected input and marks invalid ones.

    Args:
        response: LLM response (should be JSON array)
        review_ids: Expected review IDs for validation

    Returns:
        List of extraction dicts with _validation_errors for invalid entries
    """
    response = response.strip()
    valid_review_ids = set(review_ids)

    # Handle markdown code blocks
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end > start:
            response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            response = response[start:end].strip()

    # Find JSON array boundaries
    bracket_start = response.find("[")
    bracket_end = response.rfind("]")
    if bracket_start >= 0 and bracket_end > bracket_start:
        response = response[bracket_start : bracket_end + 1]

    try:
        results = json.loads(response)
        if isinstance(results, list):
            # Validate review_ids against expected input
            validated = []
            for r in results:
                rid = r.get("review_id", "")
                if rid and rid not in valid_review_ids:
                    # Mark as rejected with invalid review_id
                    r["_validation_errors"] = r.get("_validation_errors", [])
                    r["_validation_errors"].append(f"invalid_review_id:{rid}")
                    r["is_relevant"] = False  # Force non-relevant for invalid IDs
                validated.append(r)
            return validated
    except json.JSONDecodeError:
        pass

    # Fallback: return non-relevant for all reviews
    return [{"review_id": rid, "is_relevant": False, "_parse_error": "batch_parse_failed"} for rid in review_ids]
