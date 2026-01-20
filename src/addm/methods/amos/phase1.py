"""Phase 1: Formula Seed Generation.

LLM reads agenda/query and produces a Formula Seed (executable JSON specification).
This is done once per policy and cached to disk.

Includes iterative validation and fix approach:
1. Generate Formula Seed from agenda
2. Validate expressions (syntax check with compile())
3. If errors, prompt LLM to fix with specific error feedback
4. Retry up to MAX_FIX_ATTEMPTS times
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


PHASE1_PROMPT = '''You are translating a task agenda into an executable specification.

## TASK AGENDA

{agenda}

## YOUR JOB

Read the agenda carefully. Produce a JSON specification that captures EVERYTHING needed to evaluate restaurants for this task.

### 1. FILTERING
Identify keywords that indicate a review is relevant to this task.
- Include variations (singular/plural, verb forms)
- Include related terms the task mentions
- Include common misspellings if relevant

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
- "sum": Sum values from extractions (can include conditional weighting)
- "expr": Mathematical expression using other computed values
- "lookup": Map restaurant attributes to values
- "case": Threshold-based classification rules

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
    {{"name": "N_INCIDENTS", "op": "count", "where": {{"field": "value"}}}},
    {{"name": "TOTAL", "op": "expr", "expr": "N_A + N_B"}},
    {{"name": "VERDICT", "op": "case", "source": "SCORE", "rules": [
      {{"when": "< 4.0", "then": "Low Risk"}},
      {{"when": "< 8.0", "then": "High Risk"}},
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
    """Validate Formula Seed structure and expressions.

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

    return errors


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
    """Generate Formula Seed from agenda prompt.

    Phase 1 of AMOS: LLM reads agenda and produces executable specification.
    Results are cached to disk (data/formula_seeds/{policy_id}.json).

    Includes iterative validation and fix approach:
    1. Generate Formula Seed from agenda
    2. Validate expressions (syntax check)
    3. If errors, prompt LLM to fix with specific error feedback
    4. Retry up to MAX_FIX_ATTEMPTS times

    Args:
        agenda: The task agenda/query prompt
        policy_id: Policy identifier (e.g., "G1_allergy_V2")
        llm: LLM service for API calls
        cache_dir: Override cache directory (default: data/formula_seeds/)
        force_regenerate: Force regeneration even if cached

    Returns:
        Tuple of (formula_seed, usage_dict)
        - formula_seed: The generated Formula Seed specification
        - usage_dict: Token/cost usage from the LLM call (empty if cached)
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

    # Generate via LLM
    prompt = PHASE1_PROMPT.format(agenda=agenda)
    messages = [{"role": "user", "content": prompt}]

    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1", "policy_id": policy_id},
    )

    # Parse JSON from response
    try:
        seed = _extract_json_from_response(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Formula Seed JSON: {e}\nResponse: {response}")

    # Validate structure and expressions
    errors = _validate_formula_seed(seed)

    # Iterative fix approach: if there are errors, ask LLM to fix them
    total_usage = dict(usage)
    fix_attempt = 0

    while errors and fix_attempt < MAX_FIX_ATTEMPTS:
        fix_attempt += 1
        logger.info(f"Formula Seed has {len(errors)} errors, attempting fix {fix_attempt}/{MAX_FIX_ATTEMPTS}")
        logger.debug(f"Errors: {errors}")

        # Build fix prompt with the errors and current seed
        fix_prompt = FIX_PROMPT.format(
            errors="\n".join(f"- {e}" for e in errors),
            seed_json=json.dumps(seed, indent=2),
        )

        # Ask LLM to fix
        fix_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"```json\n{json.dumps(seed, indent=2)}\n```"},
            {"role": "user", "content": fix_prompt},
        ]

        fix_response, fix_usage = await llm.call_async_with_usage(
            fix_messages,
            context={"phase": "phase1_fix", "policy_id": policy_id, "attempt": fix_attempt},
        )

        # Accumulate usage
        for key in ["prompt_tokens", "completion_tokens"]:
            total_usage[key] = total_usage.get(key, 0) + fix_usage.get(key, 0)
        total_usage["cost_usd"] = total_usage.get("cost_usd", 0) + fix_usage.get("cost_usd", 0)

        # Try to parse fixed seed
        try:
            fixed_seed = _extract_json_from_response(fix_response)
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

    # Add metadata and save to cache
    seed_with_meta = {
        **seed,
        "_metadata": {
            "agenda_hash": agenda_hash,
            "policy_id": policy_id,
            "generated_by": llm._config.get("model", "unknown"),
            "fix_attempts": fix_attempt,
        },
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(seed_with_meta, f, indent=2)

    return seed, total_usage
