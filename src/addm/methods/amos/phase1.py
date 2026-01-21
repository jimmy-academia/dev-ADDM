"""Phase 1: Formula Seed Generation with Multi-Step Decomposition.

LLM reads agenda/query and produces a Formula Seed (executable JSON specification).
This is done once per policy and cached to disk.

Multi-Step Decomposition (3 steps):
1. Task Understanding: Extract structured concepts from policy text
2. Seed Keyword Generation: Generate initial keyword list for the domain
3. Seed Assembly: Generate formula seed with fixed schema and compute format

Includes iterative validation and fix approach after generation.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from addm.llm import LLMService

logger = logging.getLogger(__name__)

# Maximum attempts to fix expression errors
MAX_FIX_ATTEMPTS = 3


# =============================================================================
# Step 1: Task Understanding
# =============================================================================

STEP1_PROMPT = '''Analyze this task agenda and extract its structured requirements.

## TASK AGENDA

{agenda}

## YOUR JOB

Read the agenda carefully and extract the key components needed to evaluate restaurants.
Output a JSON object with the following structure:

```json
{{
  "task_type": "<brief description, e.g., 'allergy safety assessment'>",

  "incident_definition": {{
    "what_counts": "<what constitutes a relevant incident>",
    "what_does_not_count": "<what should NOT be counted as incidents>"
  }},

  "severity_levels": [
    {{"level": "none", "description": "<when to use>", "points": 0}},
    {{"level": "mild", "description": "<when to use>", "points": <number>}},
    {{"level": "moderate", "description": "<when to use>", "points": <number>}},
    {{"level": "severe", "description": "<when to use>", "points": <number>}}
  ],

  "account_types": [
    {{"type": "firsthand", "description": "Reviewer personally experienced it", "weight": 1.0}},
    {{"type": "secondhand", "description": "Reviewer heard about it from others", "weight": <0-1>}},
    {{"type": "general", "description": "General statement without specific incident", "weight": <0-1>}}
  ],

  "modifiers": [
    {{"name": "<modifier_name>", "condition": "<when it applies>", "points": <number>}}
  ],

  "thresholds": {{
    "critical": <score for critical verdict>,
    "high": <score for high verdict>,
    "low": <score for low verdict, or 0 if none>
  }},

  "verdict_labels": {{
    "critical": "<exact verdict text for critical>",
    "high": "<exact verdict text for high>",
    "low": "<exact verdict text for low>"
  }},

  "approach_hints": {{
    "early_stop_score": <score at which verdict is definitely critical>,
    "priority_indicators": ["<words that indicate high-priority reviews>"]
  }},

  "temporal_requirements": {{
    "has_recency_weighting": <true/false>,
    "decay_description": "<how recency affects scoring, if applicable>"
  }}
}}
```

Extract EXACTLY what the agenda specifies - don't add assumptions.
If the agenda doesn't specify something (like modifiers), use an empty list.

Output ONLY the JSON:

```json
'''


# =============================================================================
# Step 2: Seed Keyword Generation
# =============================================================================

STEP2_PROMPT = '''Generate an initial keyword list for filtering reviews based on this task understanding.

## TASK UNDERSTANDING

{task_understanding}

## YOUR JOB

Generate a seed keyword list that will help identify potentially relevant reviews.
This is an INITIAL list - Phase 2 will expand it based on actual review content.

Guidelines:
1. Include the core domain terms (e.g., "allergy", "allergic" for allergy tasks)
2. Include key examples mentioned in the task (e.g., specific allergens if mentioned)
3. Include severity indicators (e.g., "sick", "reaction", "hospital")
4. Include account-type indicators (e.g., "my friend said", "I heard")
5. Use word boundaries - prefer specific phrases over single common words
6. For food/allergen tasks: include common allergen types (peanut, gluten, dairy, shellfish, etc.)
7. For service tasks: include staff-related terms (waiter, server, manager, etc.)
8. For experience tasks: include emotion words (romantic, intimate, loud, cramped, etc.)

DO NOT try to be exhaustive - Phase 2 handles expansion.

Output a JSON object:

```json
{{
  "core_terms": ["<primary domain terms>"],
  "examples_from_task": ["<specific examples mentioned in task>"],
  "severity_indicators": ["<words indicating severity>"],
  "account_indicators": ["<phrases for firsthand vs secondhand>"],
  "domain_vocabulary": ["<common domain-specific terms>"],
  "priority_terms": ["<terms that indicate high-priority reviews>"]
}}
```

Output ONLY the JSON:

```json
'''


# =============================================================================
# Step 3: Seed Assembly
# =============================================================================

# Fixed extraction field names for consistency
REQUIRED_EXTRACTION_FIELDS = """
You MUST use these EXACT field names in extraction schema:
- ACCOUNT_TYPE: "firsthand" | "secondhand" | "general"
- INCIDENT_SEVERITY: "none" | "mild" | "moderate" | "severe"
- SPECIFIC_INCIDENT: string describing what happened (or null if none)

Additional fields may be added based on task requirements (e.g., STAFF_RESPONSE, RECURRENCE).
"""

COMPUTE_TEMPLATE = """
Use this EXACT compute structure (adjust values based on task):

1. Count incidents:
   {{"name": "N_INCIDENTS", "op": "count", "where": {{"ACCOUNT_TYPE": "firsthand", "INCIDENT_SEVERITY": ["mild", "moderate", "severe"]}}}}

2. Sum base points (use CASE WHEN for severity scoring):
   {{"name": "BASE_POINTS", "op": "sum", "expr": "CASE WHEN INCIDENT_SEVERITY = 'severe' THEN <severe_pts> WHEN INCIDENT_SEVERITY = 'moderate' THEN <mod_pts> WHEN INCIDENT_SEVERITY = 'mild' THEN <mild_pts> ELSE 0 END", "where": {{"ACCOUNT_TYPE": "firsthand"}}}}

3. Sum modifier points (if task has modifiers):
   {{"name": "MODIFIER_POINTS", "op": "sum", "expr": "CASE WHEN MODIFIER_FIELD = 'trigger_value' THEN <mod_points> ELSE 0 END", "where": {{"ACCOUNT_TYPE": "firsthand"}}}}

4. Total score:
   {{"name": "SCORE", "op": "expr", "expr": "BASE_POINTS + MODIFIER_POINTS"}}

5. Verdict (use threshold values from task):
   {{"name": "VERDICT", "op": "case", "source": "SCORE", "rules": [
     {{"when": ">= <critical_threshold>", "then": "<critical_label>"}},
     {{"when": ">= <high_threshold>", "then": "<high_label>"}},
     {{"else": "<low_label>"}}
   ]}}
"""

STEP3_PROMPT = '''Generate the complete Formula Seed specification.

## TASK UNDERSTANDING

{task_understanding}

## SEED KEYWORDS

{seed_keywords}

## REQUIREMENTS

{field_requirements}

{compute_requirements}

## YOUR JOB

Generate a complete Formula Seed JSON that:
1. Uses ALL keywords from the seed (combine all categories into one list)
2. Uses the EXACT extraction field names specified above
3. Uses the scoring/threshold values from task understanding
4. Includes a search strategy for adaptive processing

Output the complete Formula Seed:

```json
{{
  "task_name": "<policy identifier from task>",

  "filter": {{
    "keywords": ["<all keywords from seed, flattened into one list>"]
  }},

  "extract": {{
    "fields": [
      {{
        "name": "ACCOUNT_TYPE",
        "type": "enum",
        "values": {{
          "firsthand": "Reviewer personally experienced the incident",
          "secondhand": "Reviewer heard about it from others",
          "general": "General statement without specific incident"
        }}
      }},
      {{
        "name": "INCIDENT_SEVERITY",
        "type": "enum",
        "values": {{
          "none": "No relevant incident",
          "mild": "<description from task>",
          "moderate": "<description from task>",
          "severe": "<description from task>"
        }}
      }},
      {{
        "name": "SPECIFIC_INCIDENT",
        "type": "string",
        "description": "Brief description of what happened"
      }}
    ]
  }},

  "compute": [
    <compute operations using exact structure from requirements>
  ],

  "output": ["VERDICT", "SCORE", "N_INCIDENTS"],

  "search_strategy": {{
    "priority_keywords": ["<high-priority terms from seed>"],
    "priority_expr": "len(keyword_hits) * 2 + (1.0 if is_recent else 0.5)",
    "stopping_condition": "SCORE >= <early_stop_score> or remaining == 0",
    "early_verdict_expr": "'<critical_label>' if SCORE >= <critical_threshold> else ('<high_label>' if SCORE >= <high_threshold> else None)",
    "use_embeddings_when": "len(keyword_matched) < 5"
  }},

  "expansion_hints": {{
    "domain": "<task domain for Phase 2 vocabulary expansion>",
    "expand_on": ["<categories to expand: allergens, symptoms, etc.>"]
  }}
}}
```

Output ONLY the JSON:

```json
'''


# =============================================================================
# Legacy single-shot prompt (kept for reference/fallback)
# =============================================================================

PHASE1_PROMPT_LEGACY = '''You are translating a task agenda into an executable specification.

## TASK AGENDA

{agenda}

## YOUR JOB

Read the agenda carefully. Produce a JSON specification that captures EVERYTHING needed to evaluate restaurants for this task.

### 1. FILTERING
Identify keywords that indicate a review is relevant to this task.
- Include variations (singular/plural, verb forms)
- Include related terms the task mentions
- Include common misspellings if relevant
- For allergy/safety tasks: include ALL common allergen types (peanut, tree nut, gluten/celiac, dairy, shellfish, soy, egg) plus reaction terms (hives, swelling, anaphylaxis, EpiPen, Benadryl, "got sick", "allergic reaction")
- Include contextual phrases like "my allergy", "food allergy", "gluten-free", "nut-free", "dairy-free"
- Avoid overly generic terms that match menu items (e.g., use "shellfish allergy" not just "shrimp")

### 2. EXTRACTION
Define what semantic signals must be extracted from each relevant review.
- Each field should have a name, type, and possible values with descriptions
- Types: "enum" (categorical), "int" (count), "float" (score), "bool" (yes/no)
- For enum types, include "none" as the first value for reviews with no relevant content

TEMPORAL INFORMATION:
- If the task mentions recency, time-based scoring, or temporal decay, extract review dates
- Use field name "REVIEW_DATE" with type "string" (format: "YYYY-MM-DD")
- Add an "AGE_YEARS" field with type "float" to represent years since review
- The computation phase can apply time-based weighting (e.g., decay factors, recency multipliers)

### 3. COMPUTATION
Define how to aggregate extractions into final results:
- "count": Count extractions matching conditions
- "sum": Sum values from extractions using SQL-style CASE WHEN expressions
  - Use for point-based scoring: {{"op": "sum", "expr": "CASE WHEN FIELD = 'value' THEN points ELSE 0 END", "where": {{...}}}}
  - The expression is evaluated PER EXTRACTION, then summed
- "expr": Mathematical expression using ONLY other computed values (N_INCIDENTS, BASE_POINTS, etc.)
  - Use for combining already-computed aggregates: "BASE_POINTS + MODIFIER_POINTS"
  - Do NOT use extraction fields directly in expr - use sum with where conditions instead
- "lookup": Map restaurant attributes to values
- "case": Threshold-based classification rules for final verdict

IMPORTANT for scoring:
- To compute point totals, use "sum" operations with CASE WHEN expressions for per-extraction scoring
- Example: {{"name": "BASE_POINTS", "op": "sum", "expr": "CASE WHEN SEVERITY = 'mild' THEN 2 WHEN SEVERITY = 'moderate' THEN 5 ELSE 0 END", "where": {{"ACCOUNT_TYPE": "firsthand"}}}}
- Then combine computed values with "expr": {{"name": "TOTAL_SCORE", "op": "expr", "expr": "BASE_POINTS + MODIFIER_POINTS"}}

TEMPORAL WEIGHTING:
- If the task requires recency weighting, use "sum" with conditional multipliers
- Example: {{"op": "sum", "expr": "5 if AGE_YEARS < 2 else 2.5 if AGE_YEARS < 3 else 1.25", "where": {{...}}}}
- Common patterns: full weight (recent), half weight (moderate age), quarter weight (old)

### 4. SEARCH STRATEGY
Define an intelligent search strategy to optimize processing:

1. **priority_keywords**: List of keywords that indicate high-value reviews (most likely to affect the final verdict). These should be the strongest signals.

2. **priority_expr**: Python expression to compute review priority score.
   - Available variables: `keyword_hits` (list of matched keywords), `is_recent` (bool), `embedding_sim` (float 0-1)
   - Higher score = process first
   - Example: `"len(keyword_hits) * 2 + (1.0 if is_recent else 0.5)"`

3. **stopping_condition**: Python expression that returns True when verdict is determinable (can stop early).
   - Available variables: `extractions` (list), `score` (float), `remaining` (int), `namespace` (dict of computed values)
   - Example: `"score >= 10 or remaining == 0"`
   - Example with counts: `"namespace.get('N_SEVERE', 0) >= 2"`

4. **early_verdict_expr**: Python expression to compute verdict when stopping early (return None if undetermined).
   - Available variables: `extractions` (list), `score` (float), `namespace` (dict)
   - Example: `"'Critical Risk' if score >= 8 else ('High Risk' if score >= 4 else None)"`

5. **use_embeddings_when**: Python expression for when to enable embedding retrieval (for finding reviews missed by keywords).
   - Available variables: `keyword_matched` (list), `total_reviews` (int)
   - Example: `"len(keyword_matched) < 5"`

Think step by step about what the task requires, then output the JSON specification.

OUTPUT FORMAT:
```json
{{
  "task_name": "<policy identifier>",

  "filter": {{
    "keywords": ["keyword1", "keyword2", ...]
  }},

  "extract": {{
    "fields": [
      {{
        "name": "<field_name>",
        "type": "enum",
        "values": {{
          "none": "No relevant content",
          "value1": "Description of value1",
          "value2": "Description of value2"
        }}
      }}
    ]
  }},

  "compute": [
    {{"name": "N_INCIDENTS", "op": "count", "where": {{"ACCOUNT_TYPE": "firsthand", "SEVERITY": ["mild", "moderate", "severe"]}}}},
    {{"name": "BASE_POINTS", "op": "sum", "expr": "CASE WHEN SEVERITY = 'mild' THEN 2 WHEN SEVERITY = 'moderate' THEN 5 WHEN SEVERITY = 'severe' THEN 15 ELSE 0 END", "where": {{"ACCOUNT_TYPE": "firsthand"}}}},
    {{"name": "MOD_POINTS", "op": "sum", "expr": "CASE WHEN STAFF_RESPONSE = 'dismissive' THEN 3 ELSE 0 END", "where": {{"ACCOUNT_TYPE": "firsthand"}}}},
    {{"name": "TOTAL_SCORE", "op": "expr", "expr": "BASE_POINTS + MOD_POINTS"}},
    {{"name": "VERDICT", "op": "case", "source": "TOTAL_SCORE", "rules": [
      {{"when": "< 4", "then": "Low Risk"}},
      {{"when": "< 8", "then": "High Risk"}},
      {{"else": "Critical Risk"}}
    ]}}
  ],

  "output": ["VERDICT", "SCORE", "N_INCIDENTS"],

  "search_strategy": {{
    "priority_keywords": ["severe", "critical", "anaphylaxis"],
    "priority_expr": "len(keyword_hits) * 2 + (1.0 if is_recent else 0.5)",
    "stopping_condition": "score >= 10 or remaining == 0",
    "early_verdict_expr": "'Critical Risk' if score >= 8 else ('High Risk' if score >= 4 else None)",
    "use_embeddings_when": "len(keyword_matched) < 5"
  }}
}}
```

Now produce the specification:

```json
'''


def _extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks and cleanup."""
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

    # Try to parse as-is first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Common fixes for LLM JSON issues:
    # 1. Remove trailing commas before closing brackets
    fixed = re.sub(r',\s*([}\]])', r'\1', response)
    # 2. Fix duplicate commas
    fixed = re.sub(r',\s*,', ',', fixed)
    # 3. Remove comments (// style)
    fixed = re.sub(r'//.*$', '', fixed, flags=re.MULTILINE)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Last resort: try to find balanced braces
    depth = 0
    start_idx = -1
    for i, char in enumerate(response):
        if char == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start_idx >= 0:
                try:
                    candidate = response[start_idx:i+1]
                    candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # If all else fails, raise with original response
    return json.loads(response)


def _compute_agenda_hash(agenda: str) -> str:
    """Compute hash of agenda for cache invalidation."""
    return hashlib.sha256(agenda.encode()).hexdigest()[:16]


def _validate_expression(expr: str, field_name: str) -> Optional[str]:
    """Validate a Python expression is syntactically correct and semantically meaningful.

    Args:
        expr: Python expression string
        field_name: Name of the field (for error messages)

    Returns:
        Error message if invalid, None if valid
    """
    if not expr or not isinstance(expr, str):
        return None  # Empty is ok

    # Check for syntax errors
    try:
        compile(expr, f"<{field_name}>", "eval")
    except SyntaxError as e:
        return f"{field_name}: SyntaxError - {e.msg}"

    # Semantic check: detect expressions that are just string literals containing Python code
    # This catches cases like: "'Critical Risk' if score >= 8 else None" wrapped in extra quotes
    # which would evaluate to the literal string instead of the conditional result
    stripped = expr.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        inner = stripped[1:-1]
        # Check if the inner content looks like Python code (contains 'if', 'else', etc.)
        if any(keyword in inner for keyword in [' if ', ' else ', ' and ', ' or ']):
            return (
                f"{field_name}: Expression appears to be a string literal containing Python code. "
                f"Remove the outer quotes. Got: {expr[:50]}..."
            )

    # Same check for single quotes
    if stripped.startswith("'") and stripped.endswith("'"):
        inner = stripped[1:-1]
        if any(keyword in inner for keyword in [' if ', ' else ', ' and ', ' or ']):
            return (
                f"{field_name}: Expression appears to be a string literal containing Python code. "
                f"Remove the outer quotes. Got: {expr[:50]}..."
            )

    return None


def _validate_formula_seed(seed: Dict[str, Any]) -> List[str]:
    """Validate Formula Seed structure, expressions, and field references.

    Args:
        seed: Formula Seed dict

    Returns:
        List of validation errors (empty list if valid)
    """
    errors = []

    required_keys = ["filter", "extract", "compute", "output"]
    for key in required_keys:
        if key not in seed:
            errors.append(f"Missing required key: {key}")

    if errors:
        return errors  # Can't continue without required keys

    # Validate filter
    if "keywords" not in seed["filter"]:
        errors.append("filter missing 'keywords'")
    elif not isinstance(seed["filter"]["keywords"], list):
        errors.append("filter.keywords must be a list")

    # Validate extract
    if "fields" not in seed["extract"]:
        errors.append("extract missing 'fields'")
    elif not isinstance(seed["extract"]["fields"], list):
        errors.append("extract.fields must be a list")

    # Validate compute
    if not isinstance(seed["compute"], list):
        errors.append("compute must be a list")

    # Validate output
    if not isinstance(seed["output"], list):
        errors.append("output must be a list")

    # Validate search_strategy expressions
    if "search_strategy" in seed:
        strategy = seed["search_strategy"]
        if not isinstance(strategy, dict):
            errors.append("search_strategy must be a dict")
        else:
            # Type checks for specific fields
            if "priority_keywords" in strategy:
                if not isinstance(strategy["priority_keywords"], list):
                    errors.append("search_strategy.priority_keywords must be a list")

            # Validate expression syntax
            for expr_field in ["priority_expr", "stopping_condition", "early_verdict_expr", "use_embeddings_when"]:
                if expr_field in strategy:
                    expr = strategy[expr_field]
                    if not isinstance(expr, str):
                        errors.append(f"search_strategy.{expr_field} must be a string expression")
                    else:
                        expr_error = _validate_expression(expr, f"search_strategy.{expr_field}")
                        if expr_error:
                            errors.append(expr_error)

    # Validate field references in compute operations
    field_errors = _validate_field_references(seed)
    errors.extend(field_errors)

    return errors


def _validate_field_references(seed: Dict[str, Any]) -> List[str]:
    """Validate that compute operations reference valid extraction fields.

    Args:
        seed: Formula Seed dict

    Returns:
        List of field reference errors
    """
    errors = []

    # Get extraction field names
    extraction_fields = set()
    for field in seed.get("extract", {}).get("fields", []):
        field_name = field.get("name", "")
        if field_name:
            extraction_fields.add(field_name.upper())
            extraction_fields.add(field_name)  # Keep original case too

    # Get computed value names (for expr operations that reference other computes)
    computed_names = set()
    for op in seed.get("compute", []):
        name = op.get("name", "")
        if name:
            computed_names.add(name.upper())
            computed_names.add(name)

    # Pattern to find field references in CASE WHEN expressions
    case_when_pattern = re.compile(r"WHEN\s+(\w+)\s*=", re.IGNORECASE)

    # Pattern to find field references in where conditions
    for op in seed.get("compute", []):
        op_name = op.get("name", "unknown")
        op_type = op.get("op", "")

        # Check CASE WHEN expressions in sum operations
        if op_type == "sum":
            expr = op.get("expr", "")
            if expr.upper().startswith("CASE"):
                matches = case_when_pattern.findall(expr)
                for field_ref in matches:
                    if field_ref.upper() not in extraction_fields:
                        errors.append(
                            f"compute.{op_name}: CASE expression references unknown field '{field_ref}'. "
                            f"Available extraction fields: {sorted(extraction_fields)}"
                        )

        # Check where conditions
        where = op.get("where", {})
        if isinstance(where, dict):
            for field_ref in where.keys():
                # Skip special keys
                if field_ref in ("and", "or", "field", "equals", "not_equals"):
                    continue
                if field_ref.upper() not in extraction_fields:
                    errors.append(
                        f"compute.{op_name}: where condition references unknown field '{field_ref}'. "
                        f"Available extraction fields: {sorted(extraction_fields)}"
                    )

        # Check case operations that reference extraction fields
        if op_type == "case":
            source = op.get("source", "")
            # Source can be either an extraction field or a computed value
            if source and source.upper() not in extraction_fields and source.upper() not in computed_names:
                # Check if it's a computed value that comes later
                later_computed = set()
                found = False
                for later_op in seed.get("compute", []):
                    if later_op.get("name") == op_name:
                        found = True
                    if found:
                        later_name = later_op.get("name", "")
                        if later_name:
                            later_computed.add(later_name.upper())

                if source.upper() not in later_computed:
                    errors.append(
                        f"compute.{op_name}: case source '{source}' not found in extraction fields or computed values"
                    )

    return errors


def _accumulate_usage(usages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Accumulate usage metrics from multiple LLM calls.

    Args:
        usages: List of usage dicts from LLM calls

    Returns:
        Combined usage dict
    """
    total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }
    for u in usages:
        total["prompt_tokens"] += u.get("prompt_tokens", 0)
        total["completion_tokens"] += u.get("completion_tokens", 0)
        total["cost_usd"] += u.get("cost_usd", 0.0)
    return total


# =============================================================================
# Multi-Step Decomposition Functions
# =============================================================================


async def _step1_task_understanding(
    agenda: str,
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Step 1: Extract structured concepts from policy text.

    Args:
        agenda: The task agenda/query prompt
        llm: LLM service for API calls
        policy_id: Policy identifier for context

    Returns:
        Tuple of (task_understanding, usage)
    """
    prompt = STEP1_PROMPT.format(agenda=agenda)
    messages = [{"role": "user", "content": prompt}]

    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_step1", "policy_id": policy_id},
    )

    try:
        task_understanding = _extract_json_from_response(response)
    except json.JSONDecodeError as e:
        logger.warning(f"Step 1 JSON parse failed: {e}, using fallback")
        # Fallback to minimal structure
        task_understanding = {
            "task_type": "unknown",
            "incident_definition": {"what_counts": "unknown", "what_does_not_count": "unknown"},
            "severity_levels": [
                {"level": "none", "points": 0},
                {"level": "mild", "points": 3},
                {"level": "moderate", "points": 8},
                {"level": "severe", "points": 15},
            ],
            "thresholds": {"critical": 8, "high": 4, "low": 0},
            "verdict_labels": {"critical": "Critical Risk", "high": "High Risk", "low": "Low Risk"},
        }

    return task_understanding, usage


async def _step2_keyword_generation(
    task_understanding: Dict[str, Any],
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Step 2: Generate initial keyword list for the domain.

    Args:
        task_understanding: Output from Step 1
        llm: LLM service for API calls
        policy_id: Policy identifier for context

    Returns:
        Tuple of (seed_keywords, usage)
    """
    prompt = STEP2_PROMPT.format(task_understanding=json.dumps(task_understanding, indent=2))
    messages = [{"role": "user", "content": prompt}]

    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_step2", "policy_id": policy_id},
    )

    try:
        seed_keywords = _extract_json_from_response(response)
    except json.JSONDecodeError as e:
        logger.warning(f"Step 2 JSON parse failed: {e}, using fallback")
        # Fallback to task_type-based keywords
        task_type = task_understanding.get("task_type", "")
        seed_keywords = {
            "core_terms": [task_type] if task_type else [],
            "examples_from_task": [],
            "severity_indicators": ["severe", "serious", "bad"],
            "account_indicators": ["I", "my", "we"],
            "domain_vocabulary": [],
            "priority_terms": ["worst", "terrible", "awful"],
        }

    return seed_keywords, usage


async def _step3_seed_assembly(
    task_understanding: Dict[str, Any],
    seed_keywords: Dict[str, Any],
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Step 3: Generate complete formula seed with fixed schema.

    Args:
        task_understanding: Output from Step 1
        seed_keywords: Output from Step 2
        llm: LLM service for API calls
        policy_id: Policy identifier for context

    Returns:
        Tuple of (formula_seed, usage)
    """
    prompt = STEP3_PROMPT.format(
        task_understanding=json.dumps(task_understanding, indent=2),
        seed_keywords=json.dumps(seed_keywords, indent=2),
        field_requirements=REQUIRED_EXTRACTION_FIELDS,
        compute_requirements=COMPUTE_TEMPLATE,
    )
    messages = [{"role": "user", "content": prompt}]

    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_step3", "policy_id": policy_id},
    )

    seed = _extract_json_from_response(response)
    return seed, usage


def _flatten_keywords(seed_keywords: Dict[str, Any]) -> List[str]:
    """Flatten keyword categories into a single deduplicated list.

    Args:
        seed_keywords: Keyword dict with categories

    Returns:
        Flattened list of unique keywords
    """
    all_keywords = []
    for category in ["core_terms", "examples_from_task", "severity_indicators",
                     "account_indicators", "domain_vocabulary", "priority_terms"]:
        keywords = seed_keywords.get(category, [])
        if isinstance(keywords, list):
            all_keywords.extend(keywords)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for kw in all_keywords:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            unique.append(kw)

    return unique


FIX_PROMPT = '''The Formula Seed you generated has validation errors:

{errors}

Here is the problematic Formula Seed:
```json
{seed_json}
```

Please fix these errors and output the COMPLETE corrected Formula Seed JSON.

Common fixes for expression errors:
- Use consistent quote style: 'Critical Risk' not "Critical Risk' (mismatched quotes)
- Ensure parentheses are balanced
- Use valid Python operators: >= not =>
- String literals inside expressions should use single quotes: 'value' not "value"

Output ONLY the corrected JSON (no explanation):

```json
'''


async def generate_formula_seed(
    agenda: str,
    policy_id: str,
    llm: LLMService,
    cache_dir: Optional[Path] = None,
    force_regenerate: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Formula Seed from agenda prompt using Multi-Step Decomposition.

    Phase 1 of AMOS: LLM reads agenda and produces executable specification.
    Results are cached to disk (data/formula_seeds/{policy_id}.json).

    Multi-Step Decomposition:
    1. Task Understanding: Extract structured concepts from policy text
    2. Keyword Generation: Generate initial keyword seed list
    3. Seed Assembly: Generate formula seed with fixed schema and compute format

    Includes iterative validation and fix approach after generation.

    Args:
        agenda: The task agenda/query prompt
        policy_id: Policy identifier (e.g., "G1_allergy_V2")
        llm: LLM service for API calls
        cache_dir: Override cache directory (default: data/formula_seeds/)
        force_regenerate: Force regeneration even if cached

    Returns:
        Tuple of (formula_seed, usage_dict)
        - formula_seed: The generated Formula Seed specification
        - usage_dict: Token/cost usage from all LLM calls (empty if cached)
    """
    cache_dir = cache_dir or Path("data/formula_seeds")
    cache_path = cache_dir / f"{policy_id}.json"
    agenda_hash = _compute_agenda_hash(agenda)

    # Check cache (unless force regenerate)
    if not force_regenerate and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        # Verify agenda hash matches
        if cached.get("_metadata", {}).get("agenda_hash") == agenda_hash:
            # Remove metadata before returning
            seed = {k: v for k, v in cached.items() if not k.startswith("_")}
            return seed, {}

    # =========================================================================
    # Multi-Step Decomposition
    # =========================================================================
    all_usages = []

    # Step 1: Task Understanding
    logger.info(f"Phase 1 Step 1: Extracting task understanding for {policy_id}")
    task_understanding, usage1 = await _step1_task_understanding(agenda, llm, policy_id)
    all_usages.append(usage1)
    logger.debug(f"Task understanding: {task_understanding.get('task_type', 'unknown')}")

    # Step 2: Keyword Generation
    logger.info(f"Phase 1 Step 2: Generating seed keywords for {policy_id}")
    seed_keywords, usage2 = await _step2_keyword_generation(task_understanding, llm, policy_id)
    all_usages.append(usage2)
    flattened_keywords = _flatten_keywords(seed_keywords)
    logger.debug(f"Generated {len(flattened_keywords)} keywords")

    # Step 3: Seed Assembly
    logger.info(f"Phase 1 Step 3: Assembling formula seed for {policy_id}")
    seed, usage3 = await _step3_seed_assembly(task_understanding, seed_keywords, llm, policy_id)
    all_usages.append(usage3)

    # Store intermediate results in seed for debugging
    seed["_step1_task_understanding"] = task_understanding
    seed["_step2_seed_keywords"] = seed_keywords

    # =========================================================================
    # Validation and Fix Loop
    # =========================================================================
    errors = _validate_formula_seed(seed)

    fix_attempt = 0
    while errors and fix_attempt < MAX_FIX_ATTEMPTS:
        fix_attempt += 1
        logger.info(f"Formula Seed has {len(errors)} errors, attempting fix {fix_attempt}/{MAX_FIX_ATTEMPTS}")
        logger.debug(f"Errors: {errors}")

        # Build fix prompt with the errors and current seed
        # Remove intermediate results before showing to LLM
        seed_for_fix = {k: v for k, v in seed.items() if not k.startswith("_")}
        fix_prompt = FIX_PROMPT.format(
            errors="\n".join(f"- {e}" for e in errors),
            seed_json=json.dumps(seed_for_fix, indent=2),
        )

        # Ask LLM to fix
        fix_messages = [
            {"role": "user", "content": f"Fix this Formula Seed:\n\n{fix_prompt}"},
        ]

        fix_response, fix_usage = await llm.call_async_with_usage(
            fix_messages,
            context={"phase": "phase1_fix", "policy_id": policy_id, "attempt": fix_attempt},
        )
        all_usages.append(fix_usage)

        # Try to parse fixed seed
        try:
            fixed_seed = _extract_json_from_response(fix_response)
            # Preserve intermediate results
            fixed_seed["_step1_task_understanding"] = task_understanding
            fixed_seed["_step2_seed_keywords"] = seed_keywords
            seed = fixed_seed
            errors = _validate_formula_seed(seed)
        except json.JSONDecodeError as e:
            logger.warning(f"Fix attempt {fix_attempt} failed to parse: {e}")
            # Keep the old seed and errors for next attempt

    # If still errors after max attempts, raise
    if errors:
        raise ValueError(
            f"Formula Seed validation failed after {MAX_FIX_ATTEMPTS} fix attempts. "
            f"Errors: {errors}"
        )

    if fix_attempt > 0:
        logger.info(f"Formula Seed fixed after {fix_attempt} attempt(s)")

    # =========================================================================
    # Save to Cache
    # =========================================================================
    total_usage = _accumulate_usage(all_usages)

    seed_with_meta = {
        **seed,
        "_metadata": {
            "agenda_hash": agenda_hash,
            "policy_id": policy_id,
            "generated_by": llm._config.get("model", "unknown"),
            "fix_attempts": fix_attempt,
            "generation_method": "multi_step_decomposition",
            "steps_completed": 3,
        },
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(seed_with_meta, f, indent=2)

    # Remove intermediate results before returning
    seed_clean = {k: v for k, v in seed.items() if not k.startswith("_")}

    return seed_clean, total_usage
