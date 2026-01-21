"""Phase 1 ReAct Approach: Iterative Formula Seed Generation.

Init → [Thought → Action → Observation]* → Finalize → Validate

This approach starts with a minimal seed and iteratively improves it.
Each iteration: LLM thinks about what's missing, takes an action, observes result.

Actions:
- ADD_KEYWORDS: Add keywords to filter
- ADD_FIELD: Add extraction field
- ADD_COMPUTE: Add compute operation
- REFINE_KEYWORDS: Remove bad keywords, add missing ones
- SET_SEARCH_STRATEGY: Define search strategy
- FINALIZE: Declare complete

No domain-specific hints - all reasoning derived from query analysis.
"""

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from addm.llm import LLMService

logger = logging.getLogger(__name__)


# =============================================================================
# ReAct System Prompt (NO domain hints)
# =============================================================================

REACT_SYSTEM_PROMPT = '''You are building a Formula Seed specification for evaluating restaurants.

## TASK AGENDA

{agenda}

## WHAT IS A FORMULA SEED?

A Formula Seed defines:
1. **filter.keywords**: Words to find relevant reviews (array of strings)
2. **extract.fields**: What to extract from each review (structured fields)
3. **compute**: How to score and determine verdict (operations)
4. **search_strategy**: How to prioritize and stop early (optional optimization)

## YOUR APPROACH

You will iteratively build the Formula Seed by:
1. Analyzing what the agenda explicitly states
2. Taking actions to add components
3. Observing the current state
4. Repeating until complete

## KEY PRINCIPLES

- Derive keywords from the AGENDA TEXT, not domain knowledge
- What words would reviewers use when discussing concepts mentioned in the agenda?
- What terms match the severity definitions given in the agenda?
- Be specific to avoid false positives

## REQUIRED FIELDS

Every Formula Seed must have these extraction fields:
- ACCOUNT_TYPE: enum ("firsthand", "secondhand", "general")
- INCIDENT_SEVERITY: enum ("none", "mild", "moderate", "severe")
- SPECIFIC_INCIDENT: string (what happened)

Additional fields may be added based on agenda requirements (e.g., modifiers).

## REQUIRED COMPUTE OPERATIONS

Every Formula Seed must have a VERDICT operation (named exactly "VERDICT") that:
- Uses the threshold values from the agenda
- Uses the verdict LABELS from the agenda (e.g., "Poor Value", "Critical Risk", etc.)
- Uses this exact structure:
  {{"name": "VERDICT", "op": "case", "source": "SCORE", "rules": [{{"when": ">= X", "then": "Label1"}}, {{"when": ">= Y", "then": "Label2"}}, {{"else": "Label3"}}]}}

## CURRENT STATE

{current_state}

## AVAILABLE ACTIONS

1. ADD_KEYWORDS: Add keywords to the filter
   {{"action": "ADD_KEYWORDS", "keywords": ["word1", "word2"], "reasoning": "why these keywords"}}

2. ADD_FIELD: Add an extraction field
   {{"action": "ADD_FIELD", "field": {{"name": "FIELD_NAME", "type": "enum|string|int|float", "values": {{"...": "..."}}}}, "reasoning": "why needed"}}

3. ADD_COMPUTE: Add a compute operation
   {{"action": "ADD_COMPUTE", "operation": {{"name": "OP_NAME", "op": "count|sum|expr|case"}}, "reasoning": "why needed"}}

4. REFINE_KEYWORDS: Remove bad keywords, add better ones
   {{"action": "REFINE_KEYWORDS", "remove": ["bad1"], "add": ["better1"], "reasoning": "why this change"}}

5. SET_SEARCH_STRATEGY: Define search optimization
   {{"action": "SET_SEARCH_STRATEGY", "strategy": {{"priority_keywords": ["kw1"], "stopping_condition": "score >= 10"}}, "reasoning": "why"}}

6. FINALIZE: Declare the seed complete
   {{"action": "FINALIZE", "reasoning": "why it's complete"}}

## YOUR RESPONSE FORMAT

Think step by step, then output a single action.

```
THOUGHT: <your reasoning about what's needed next>

ACTION:
```json
{{"action": "ACTION_TYPE", "param": "value", "reasoning": "why"}}
```
```

What action should you take?
'''


# =============================================================================
# Completeness Check Prompt
# =============================================================================

COMPLETENESS_CHECK_PROMPT = '''Review this Formula Seed and identify what's missing or problematic.

## TASK AGENDA

{agenda}

## CURRENT FORMULA SEED

{current_state}

## CHECK CRITERIA

1. **Keywords**: Are there enough keywords to find relevant reviews? Are any too generic (would match irrelevant reviews)?
2. **Extraction Fields**: Are all required fields present (ACCOUNT_TYPE, INCIDENT_SEVERITY, SPECIFIC_INCIDENT)? Any additional fields needed for modifiers?
3. **Compute Operations**: Is there scoring logic? Is there a VERDICT operation? Do field references match extraction fields?
4. **Search Strategy**: Is there priority_keywords? stopping_condition? early_verdict_expr?

## RESPONSE FORMAT

```json
{{
  "is_complete": <true if ready, false if needs work>,
  "missing": ["<list what's missing>"],
  "problems": ["<list problems found>"],
  "suggestions": ["<specific actions to take>"]
}}
```

Output ONLY the JSON:

```json
'''


# =============================================================================
# Initial Analysis Prompt
# =============================================================================

INIT_PROMPT = '''Analyze this task agenda to understand what the Formula Seed needs to capture.

## TASK AGENDA

{agenda}

## YOUR JOB

Read the agenda carefully and extract:
1. What is being evaluated? (primary concepts)
2. What severity levels are defined? (with point values)
3. What account types matter? (firsthand, secondhand, general)
4. What modifiers affect scoring?
5. What thresholds determine verdicts?

Then suggest the FIRST action to take (usually ADD_KEYWORDS with core terms from the agenda).

## RESPONSE FORMAT

```
ANALYSIS:
- Primary concept: <what the agenda is about>
- Key terms in agenda: <specific words used in the agenda text>
- Severity levels: <levels and points from agenda>
- Verdict thresholds: <from agenda>

THOUGHT: <what the first step should be>

ACTION:
```json
{{"action": "ADD_KEYWORDS", "keywords": ["term1", "term2"], "reasoning": "why"}}
```
```
'''


def _extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling nested objects."""
    import re

    # Find JSON in code block (use greedy match to get full content)
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', response)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find bare JSON object with balanced braces
        json_str = _find_balanced_json(response)
        if not json_str:
            raise ValueError("No JSON found in response")

    # Clean up common issues
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    json_str = re.sub(r',\s*,', ',', json_str)

    return json.loads(json_str)


def _find_balanced_json(text: str) -> Optional[str]:
    """Find a balanced JSON object in text using brace counting."""
    # Find the first opening brace
    start = text.find("{")
    if start < 0:
        return None

    # Count braces to find matching close
    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _accumulate_usage(usages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Accumulate usage metrics from multiple LLM calls."""
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


def _create_empty_seed(policy_id: str) -> Dict[str, Any]:
    """Create an empty Formula Seed structure.

    Output array is minimal - only VERDICT is truly required.
    Additional outputs (SCORE, N_INCIDENTS, etc.) should be added
    dynamically based on what the LLM generates in compute operations.
    """
    return {
        "task_name": policy_id,
        "filter": {
            "keywords": []
        },
        "extract": {
            "fields": []
        },
        "compute": [],
        "output": ["VERDICT"],  # Only VERDICT is required; others added dynamically
        "search_strategy": {}
    }


def _apply_action(seed: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    """Apply an action to the seed and return the updated seed.

    Args:
        seed: Current Formula Seed
        action: Action dict with action type and parameters

    Returns:
        Updated Formula Seed
    """
    seed = deepcopy(seed)
    action_type = action.get("action", "")

    if action_type == "ADD_KEYWORDS":
        keywords = action.get("keywords", [])
        existing = set(seed["filter"]["keywords"])
        for kw in keywords:
            if kw not in existing:
                seed["filter"]["keywords"].append(kw)
                existing.add(kw)

    elif action_type == "ADD_FIELD":
        field = action.get("field", {})
        if field and field.get("name"):
            # Check if field already exists
            existing_names = {f.get("name") for f in seed["extract"]["fields"]}
            if field["name"] not in existing_names:
                seed["extract"]["fields"].append(field)

    elif action_type == "ADD_COMPUTE":
        operation = action.get("operation", {})
        if operation and operation.get("name"):
            # Check if operation already exists
            existing_names = {op.get("name") for op in seed["compute"]}
            if operation["name"] not in existing_names:
                seed["compute"].append(operation)

    elif action_type == "REFINE_KEYWORDS":
        to_remove = set(action.get("remove", []))
        to_add = action.get("add", [])
        # Remove keywords
        seed["filter"]["keywords"] = [
            kw for kw in seed["filter"]["keywords"]
            if kw not in to_remove
        ]
        # Add new keywords
        existing = set(seed["filter"]["keywords"])
        for kw in to_add:
            if kw not in existing:
                seed["filter"]["keywords"].append(kw)
                existing.add(kw)

    elif action_type == "SET_SEARCH_STRATEGY":
        strategy = action.get("strategy", {})
        seed["search_strategy"] = strategy

    elif action_type == "FINALIZE":
        # No changes needed - just marks completion
        pass

    return seed


def _format_current_state(seed: Dict[str, Any]) -> str:
    """Format current seed state for display in prompt."""
    return json.dumps(seed, indent=2)


async def _react_iteration(
    agenda: str,
    seed: Dict[str, Any],
    llm: LLMService,
    policy_id: str,
    iteration: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], bool, Dict[str, Any]]:
    """Run one iteration of ReAct loop.

    Args:
        agenda: The task agenda
        seed: Current Formula Seed
        llm: LLM service
        policy_id: Policy identifier
        iteration: Current iteration number

    Returns:
        Tuple of (updated_seed, action_taken, is_finalized, usage)
    """
    prompt = REACT_SYSTEM_PROMPT.format(
        agenda=agenda,
        current_state=_format_current_state(seed),
    )

    messages = [{"role": "user", "content": prompt}]

    response, usage = await llm.call_async_with_usage(
        messages,
        context={
            "phase": "phase1_react",
            "step": f"iteration_{iteration}",
            "policy_id": policy_id,
        },
    )

    # Extract action from response
    try:
        action = _extract_json_from_response(response)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"ReAct iteration {iteration} failed to parse action: {e}")
        # Return current seed unchanged, not finalized
        return seed, {"action": "PARSE_ERROR", "error": str(e)}, False, usage

    # Check if finalized
    is_finalized = action.get("action") == "FINALIZE"

    # Apply action
    updated_seed = _apply_action(seed, action)

    return updated_seed, action, is_finalized, usage


async def _check_completeness(
    agenda: str,
    seed: Dict[str, Any],
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Check if the seed is complete and get suggestions.

    Args:
        agenda: The task agenda
        seed: Current Formula Seed
        llm: LLM service
        policy_id: Policy identifier

    Returns:
        Tuple of (check_result, usage)
    """
    prompt = COMPLETENESS_CHECK_PROMPT.format(
        agenda=agenda,
        current_state=_format_current_state(seed),
    )

    messages = [{"role": "user", "content": prompt}]

    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_react", "step": "completeness_check", "policy_id": policy_id},
    )

    try:
        result = _extract_json_from_response(response)
    except (json.JSONDecodeError, ValueError):
        result = {
            "is_complete": False,
            "missing": ["Parse error - continuing"],
            "problems": [],
            "suggestions": [],
        }

    return result, usage


async def _init_analysis(
    agenda: str,
    seed: Dict[str, Any],
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Initial analysis to kick off the ReAct loop.

    Args:
        agenda: The task agenda
        seed: Empty Formula Seed
        llm: LLM service
        policy_id: Policy identifier

    Returns:
        Tuple of (updated_seed, action_taken, usage)
    """
    prompt = INIT_PROMPT.format(agenda=agenda)
    messages = [{"role": "user", "content": prompt}]

    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_react", "step": "init", "policy_id": policy_id},
    )

    try:
        action = _extract_json_from_response(response)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"ReAct init failed to parse action: {e}")
        # Default to adding a placeholder keyword
        action = {"action": "ADD_KEYWORDS", "keywords": ["incident"], "reasoning": "fallback"}

    updated_seed = _apply_action(seed, action)
    return updated_seed, action, usage


def _ensure_required_fields(seed: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required extraction fields are present.

    Only adds minimal STRUCTURAL placeholders if the LLM failed to generate them.
    Does NOT add content-specific values (enum values, descriptions, etc.)
    - those must come from the LLM based on the agenda.

    If the LLM didn't generate proper fields, Phase 2 will fail clearly
    rather than silently using wrong task-specific values.
    """
    seed = deepcopy(seed)
    existing_names = {f.get("name") for f in seed["extract"]["fields"]}

    # ACCOUNT_TYPE is a structural constant - same across all tasks
    # (firsthand/secondhand/general is how we categorize review perspective)
    if "ACCOUNT_TYPE" not in existing_names:
        logger.warning("ReAct: LLM did not generate ACCOUNT_TYPE field - adding structural placeholder")
        seed["extract"]["fields"].append({
            "name": "ACCOUNT_TYPE",
            "type": "enum",
            "values": {"firsthand": "firsthand", "secondhand": "secondhand", "general": "general"}
        })

    # INCIDENT_SEVERITY values are TASK-SPECIFIC - don't hardcode!
    # Let the LLM generate appropriate values based on the agenda.
    # If missing, add empty placeholder so Phase 2 fails clearly.
    if "INCIDENT_SEVERITY" not in existing_names:
        logger.warning("ReAct: LLM did not generate INCIDENT_SEVERITY field - adding empty placeholder")
        seed["extract"]["fields"].append({
            "name": "INCIDENT_SEVERITY",
            "type": "enum",
            "values": {}  # Empty - LLM should have generated task-specific values
        })

    if "SPECIFIC_INCIDENT" not in existing_names:
        logger.warning("ReAct: LLM did not generate SPECIFIC_INCIDENT field - adding structural placeholder")
        seed["extract"]["fields"].append({
            "name": "SPECIFIC_INCIDENT",
            "type": "string",
            "description": "Description of what happened"
        })

    return seed


def _ensure_compute_operations(seed: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure minimal compute operations are present.

    DESIGN PRINCIPLE: Fallbacks should only be STRUCTURAL, not content-specific.
    - OK: "Must have a VERDICT operation" (structure)
    - NOT OK: "VERDICT must use 'Has Issues' label" (content)

    If LLM generates task-specific content, trust it completely.
    Only add minimal structural fallbacks if LLM generates NOTHING,
    and let Phase 2 fail clearly if the structure is wrong rather
    than silently using incorrect task-specific values.
    """
    seed = deepcopy(seed)
    existing_names = {op.get("name") for op in seed["compute"]}

    # If LLM generated any compute operations, trust them fully
    if len(seed["compute"]) > 0:
        # Check if VERDICT exists (exact name)
        if "VERDICT" in existing_names:
            return seed

        # Look for verdict-like operations with flexible detection:
        # - Contains "VERDICT" (e.g., PRICE_WORTH_VERDICT, RISK_VERDICT)
        # - Ends with common classification suffixes
        # - Accept any op type that produces categorical output (case, expr)
        verdict_op = None
        verdict_idx = None
        for idx, op in enumerate(seed["compute"]):
            name = op.get("name", "").upper()
            op_type = op.get("op", "")

            # Flexible detection: VERDICT in name, or classification-like suffixes
            is_verdict_like = (
                "VERDICT" in name or
                name.endswith("_CLASSIFICATION") or
                name.endswith("_JUDGMENT") or
                name.endswith("_RESULT") or
                name.endswith("_DECISION")
            )

            # Accept case or expr operations (both can produce verdicts)
            is_categorical_op = op_type in ("case", "expr")

            if is_verdict_like and is_categorical_op:
                verdict_op = op
                verdict_idx = idx
                break

        if verdict_op:
            # Rename to VERDICT and normalize structure (preserves LLM's labels/thresholds)
            normalized = _normalize_verdict_op(verdict_op)
            seed["compute"][verdict_idx] = normalized
            return seed

        # LLM generated compute ops but no verdict-like op found
        # Log error but DON'T add generic fallback with hardcoded content
        # Let Phase 2 fail clearly if there's no verdict logic
        logger.error(
            "ReAct: LLM generated compute ops but no VERDICT found. "
            f"Operations: {[op.get('name') for op in seed['compute']]}. "
            "Seed will likely fail in Phase 2."
        )
        # Add structural placeholder with empty rules - Phase 2 will fail clearly
        seed["compute"].append({
            "name": "VERDICT",
            "op": "case",
            "source": None,
            "rules": []  # Empty - LLM should have generated task-specific rules
        })
        return seed

    # LLM generated NO compute ops at all - add minimal structural fallback
    # This is a serious LLM failure - log error and add empty structure
    logger.error(
        "ReAct: LLM generated NO compute operations. "
        "Adding empty structural fallback - seed will fail in Phase 2."
    )

    seed["compute"] = [
        {
            "name": "VERDICT",
            "op": "case",
            "source": None,
            "rules": []  # Empty - no hardcoded thresholds or labels
        }
    ]

    return seed


def _normalize_verdict_op(op: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a verdict operation to standard structure.

    Handles different formats LLM might generate:
    - {cases: [{cond, value}], default} -> {rules: [{when, then}, {else}]}
    - Already correct format -> return as-is with name=VERDICT
    """
    normalized = {"name": "VERDICT", "op": "case"}

    # Copy source if present
    if "source" in op:
        normalized["source"] = op["source"]

    # Already has rules format
    if "rules" in op:
        normalized["rules"] = op["rules"]
        return normalized

    # Convert cases/default format to rules format
    if "cases" in op:
        rules = []
        for case in op["cases"]:
            cond = case.get("cond", case.get("condition", ""))
            value = case.get("value", case.get("then", ""))
            # Extract operator and threshold from condition like "SCORE >= 5"
            if ">=" in cond:
                parts = cond.split(">=")
                threshold = parts[-1].strip()
                rules.append({"when": f">= {threshold}", "then": value})
            elif ">" in cond:
                parts = cond.split(">")
                threshold = parts[-1].strip()
                rules.append({"when": f"> {threshold}", "then": value})
            elif "<=" in cond:
                parts = cond.split("<=")
                threshold = parts[-1].strip()
                rules.append({"when": f"<= {threshold}", "then": value})
            elif "<" in cond:
                parts = cond.split("<")
                threshold = parts[-1].strip()
                rules.append({"when": f"< {threshold}", "then": value})
            else:
                # Unknown format, use as-is
                rules.append({"when": cond, "then": value})

        # Add default as else
        if "default" in op:
            rules.append({"else": op["default"]})

        normalized["rules"] = rules
        return normalized

    # Unknown format - return original with name changed
    result = deepcopy(op)
    result["name"] = "VERDICT"
    return result


async def generate_react(
    agenda: str,
    policy_id: str,
    llm: LLMService,
    max_iterations: int = 8,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Generate Formula Seed using ReAct approach.

    Iterative loop: Init → [Thought → Action → Observation]* → Finalize

    Args:
        agenda: The task agenda/query prompt
        policy_id: Policy identifier (e.g., "G1_allergy_V2")
        llm: LLM service for API calls
        max_iterations: Maximum number of iterations (default 8)

    Returns:
        Tuple of (formula_seed, intermediates, usage)
        - formula_seed: The generated Formula Seed specification
        - intermediates: Dict with iteration history for debugging
        - usage: Combined token/cost usage from all LLM calls
    """
    all_usages = []
    action_history = []

    # Initialize empty seed
    seed = _create_empty_seed(policy_id)

    # Initial analysis
    logger.info(f"ReAct: Initializing for {policy_id}")
    seed, init_action, init_usage = await _init_analysis(agenda, seed, llm, policy_id)
    all_usages.append(init_usage)
    action_history.append({"iteration": 0, "action": init_action, "type": "init"})
    logger.debug(f"ReAct init action: {init_action.get('action', 'unknown')}")

    # ReAct loop
    for iteration in range(1, max_iterations + 1):
        logger.info(f"ReAct: Iteration {iteration}/{max_iterations} for {policy_id}")

        # Run iteration
        seed, action, is_finalized, usage = await _react_iteration(
            agenda, seed, llm, policy_id, iteration
        )
        all_usages.append(usage)
        action_history.append({
            "iteration": iteration,
            "action": action,
            "finalized": is_finalized,
        })

        action_type = action.get("action", "unknown")
        logger.debug(f"ReAct iteration {iteration}: {action_type}")

        if is_finalized:
            logger.info(f"ReAct: Finalized at iteration {iteration}")
            break

        # Every 3 iterations, do a completeness check
        if iteration % 3 == 0 and iteration < max_iterations:
            check_result, check_usage = await _check_completeness(
                agenda, seed, llm, policy_id
            )
            all_usages.append(check_usage)

            if check_result.get("is_complete", False):
                logger.info(f"ReAct: Completeness check passed at iteration {iteration}")
                break
            else:
                logger.debug(f"ReAct: Missing items: {check_result.get('missing', [])}")

    # Ensure required fields and compute operations
    seed = _ensure_required_fields(seed)
    seed = _ensure_compute_operations(seed)

    # Final completeness check
    final_check, final_usage = await _check_completeness(agenda, seed, llm, policy_id)
    all_usages.append(final_usage)

    intermediates = {
        "action_history": action_history,
        "final_check": final_check,
        "total_iterations": len(action_history),
    }

    total_usage = _accumulate_usage(all_usages)

    return seed, intermediates, total_usage
