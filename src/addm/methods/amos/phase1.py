"""Phase 1: Formula Seed Generation.

LLM reads agenda/query and produces a Formula Seed (executable JSON specification).
This is done once per policy and cached to disk.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from addm.llm import LLMService


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

### 3. COMPUTATION
Define how to aggregate extractions into final results:
- "count": Count extractions matching conditions
- "sum": Sum values from extractions
- "expr": Mathematical expression using other computed values
- "lookup": Map restaurant attributes to values
- "case": Threshold-based classification rules

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

  "output": ["VERDICT", "SCORE", "N_INCIDENTS"]
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


def _validate_formula_seed(seed: Dict[str, Any]) -> None:
    """Validate Formula Seed structure."""
    required_keys = ["filter", "extract", "compute", "output"]
    for key in required_keys:
        if key not in seed:
            raise ValueError(f"Formula Seed missing required key: {key}")

    # Validate filter
    if "keywords" not in seed["filter"]:
        raise ValueError("Formula Seed filter missing 'keywords'")
    if not isinstance(seed["filter"]["keywords"], list):
        raise ValueError("Formula Seed filter.keywords must be a list")

    # Validate extract
    if "fields" not in seed["extract"]:
        raise ValueError("Formula Seed extract missing 'fields'")
    if not isinstance(seed["extract"]["fields"], list):
        raise ValueError("Formula Seed extract.fields must be a list")

    # Validate compute
    if not isinstance(seed["compute"], list):
        raise ValueError("Formula Seed compute must be a list")

    # Validate output
    if not isinstance(seed["output"], list):
        raise ValueError("Formula Seed output must be a list")


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

    # Validate structure
    _validate_formula_seed(seed)

    # Add metadata and save to cache
    seed_with_meta = {
        **seed,
        "_metadata": {
            "agenda_hash": agenda_hash,
            "policy_id": policy_id,
            "generated_by": llm._config.get("model", "unknown"),
        },
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(seed_with_meta, f, indent=2)

    return seed, usage
