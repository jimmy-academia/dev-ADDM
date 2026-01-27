"""Phase 1 Hybrid Approach (Deprecated).

Single-shot NL → PolicyYAML → deterministic compiler.

This approach was replaced by the PARTS approach which uses 3 focused LLM calls
for more reliable extraction.

Kept for reference only.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

from addm.llm import LLMService
from addm.utils.debug_logger import get_debug_logger
from addm.utils.output import output

from ..seed_compiler import compile_yaml_to_seed, validate_policy_yaml, PolicyYAMLValidationError

logger = logging.getLogger(__name__)

# Type alias for progress callback
ProgressCallback = Callable[[int, str, float, str], None]


# =============================================================================
# TEXT2YAML Prompt (moved from phase1_prompts.py)
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


def _extract_yaml_from_response(response: str) -> str:
    """Extract YAML content from LLM response."""
    response = response.strip()

    if "```yaml" in response:
        start = response.find("```yaml") + 7
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    if "```" in response:
        start = response.find("```") + 3
        first_newline = response.find("\n", start)
        if first_newline > start:
            start = first_newline + 1
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    return response


def _parse_yaml_safely(yaml_str: str) -> Dict[str, Any]:
    """Parse YAML string with error handling."""
    import re
    import yaml as yaml_lib

    try:
        data = yaml_lib.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")
        return data
    except yaml_lib.YAMLError as e:
        raise ValueError(f"YAML parse error: {e}")


def _accumulate_usage(usages: list) -> Dict[str, Any]:
    """Accumulate usage metrics from multiple LLM calls."""
    total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "latency_ms": 0.0,
    }
    for u in usages:
        if not u:
            continue
        total["prompt_tokens"] += u.get("prompt_tokens", 0)
        total["completion_tokens"] += u.get("completion_tokens", 0)
        total["cost_usd"] += u.get("cost_usd", 0.0)
        total["latency_ms"] += u.get("latency_ms", 0.0)
    return total


def _validate_formula_seed(seed: Dict[str, Any]) -> list:
    """Validate Formula Seed structure (simplified version)."""
    errors = []
    required_keys = ["extract", "compute", "output"]
    for key in required_keys:
        if key not in seed:
            errors.append(f"Missing required key: {key}")
    return errors


async def _generate_policy_yaml(
    agenda: str,
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate PolicyYAML from natural language agenda using LLM.

    This is the first step of the hybrid approach.

    Args:
        agenda: The task agenda/query prompt
        llm: LLM service for API calls
        policy_id: Policy identifier for context

    Returns:
        Tuple of (policy_yaml_dict, usage)
    """
    output.status(f"[Phase 1 Hybrid] Step 1/2: Generating PolicyYAML...")

    prompt = TEXT2YAML_PROMPT.format(agenda=agenda)
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_hybrid", "step": "text2yaml", "policy_id": policy_id},
    )
    latency_ms = (time.time() - start_time) * 1000
    usage["latency_ms"] = latency_ms

    # Log to debug file
    if debug_logger := get_debug_logger():
        debug_logger.log_llm_call(
            sample_id=policy_id,
            phase="phase1_text2yaml",
            prompt=prompt,
            response=response,
            model=llm._config.get("model", "unknown"),
            latency_ms=latency_ms,
        )

    # Extract and parse YAML
    yaml_str = _extract_yaml_from_response(response)
    logger.debug(f"Extracted YAML ({len(yaml_str)} chars):\n{yaml_str[:500]}...")

    try:
        yaml_data = _parse_yaml_safely(yaml_str)
    except ValueError as e:
        logger.error(f"Failed to parse YAML response: {e}")
        logger.debug(f"Raw response:\n{response}")
        raise ValueError(f"Failed to parse LLM response as YAML: {e}")

    n_terms = len(yaml_data.get("terms", []))
    n_rules = len(yaml_data.get("rules", []))
    output.status(f"[Phase 1 Hybrid] PolicyYAML generated: {n_terms} terms, {n_rules} rules")

    return yaml_data, usage


async def generate_formula_seed_hybrid(
    agenda: str,
    policy_id: str,
    llm: LLMService,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Formula Seed using hybrid approach: NL → PolicyYAML → compiled seed.

    This approach splits the work:
    - LLM generates simple PolicyYAML (good at understanding, bad at precision)
    - Deterministic compiler transforms to Formula Seed (guaranteed correct)

    Args:
        agenda: The task agenda/query prompt
        policy_id: Policy identifier (e.g., "G1_allergy_V2")
        llm: LLM service for API calls
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (formula_seed, usage_dict)
    """
    start_time = time.perf_counter()
    all_usages = []

    def report(step: str, progress: float, detail: str = "") -> None:
        if progress_callback:
            progress_callback(1, step, progress, detail)

    # Step 1: LLM generates PolicyYAML from agenda
    report("TEXT2YAML", 10, "generating yaml")
    logger.info(f"Phase 1 Hybrid: Step 1/2 TEXT2YAML for {policy_id}")

    yaml_data, usage1 = await _generate_policy_yaml(agenda, llm, policy_id)
    all_usages.append(usage1)
    report("TEXT2YAML", 40, "complete")

    # Step 2: Validate PolicyYAML
    report("VALIDATE_YAML", 45, "validating yaml")
    try:
        validate_policy_yaml(yaml_data)
        output.status(f"[Phase 1 Hybrid] PolicyYAML validation: passed")
    except PolicyYAMLValidationError as e:
        logger.warning(f"PolicyYAML validation failed: {e}")
        output.warn(f"[Phase 1 Hybrid] PolicyYAML validation failed: {e}")
        raise ValueError(f"PolicyYAML validation failed: {e}")
    report("VALIDATE_YAML", 50, "passed")

    # Step 3: Deterministic compilation to Formula Seed
    report("COMPILE", 55, "compiling seed")
    output.status(f"[Phase 1 Hybrid] Step 2/2: Compiling to Formula Seed...")
    logger.info(f"Phase 1 Hybrid: Step 2/2 COMPILE for {policy_id}")

    try:
        seed = compile_yaml_to_seed(yaml_data, validate=False)  # Already validated
    except Exception as e:
        logger.error(f"Seed compilation failed: {e}")
        raise ValueError(f"Failed to compile PolicyYAML to seed: {e}")

    seed["_approach"] = "hybrid"
    seed["_policy_yaml"] = yaml_data
    report("COMPILE", 75, "complete")

    # Step 4: Validate the compiled Formula Seed
    report("VALIDATE_SEED", 80, "validating seed")
    errors = _validate_formula_seed(seed)

    if errors:
        output.warn(f"[Phase 1 Hybrid] Seed validation: {len(errors)} errors")
        logger.warning(f"Compiled seed has validation errors: {errors}")
        raise ValueError(f"Compiled seed validation failed: {errors}")
    else:
        output.status(f"[Phase 1 Hybrid] Seed validation: passed")
        report("VALIDATE_SEED", 95, "passed")

    # Return seed
    total_usage = _accumulate_usage(all_usages)
    wall_clock_ms = (time.perf_counter() - start_time) * 1000
    total_usage["wall_clock_ms"] = wall_clock_ms

    # Keep _policy_yaml for saving, remove other intermediates
    seed_clean = {k: v for k, v in seed.items() if not k.startswith("_") or k == "_policy_yaml"}

    return seed_clean, total_usage
