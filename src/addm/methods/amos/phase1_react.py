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

## CURRENT STATE

{current_state}

## AVAILABLE ACTIONS

1. ADD_KEYWORDS: Add keywords to the filter
   {{"action": "ADD_KEYWORDS", "keywords": ["word1", "word2"], "reasoning": "why these keywords"}}

2. ADD_FIELD: Add an extraction field
   {{"action": "ADD_FIELD", "field": {{"name": "FIELD_NAME", "type": "enum|string|int|float", "values": {{...}}}}, "reasoning": "why needed"}}

3. ADD_COMPUTE: Add a compute operation
   {{"action": "ADD_COMPUTE", "operation": {{"name": "...", "op": "count|sum|expr|case", ...}}, "reasoning": "why needed"}}

4. REFINE_KEYWORDS: Remove bad keywords, add better ones
   {{"action": "REFINE_KEYWORDS", "remove": ["bad1"], "add": ["better1"], "reasoning": "why this change"}}

5. SET_SEARCH_STRATEGY: Define search optimization
   {{"action": "SET_SEARCH_STRATEGY", "strategy": {{"priority_keywords": [...], "stopping_condition": "...", ...}}, "reasoning": "why"}}

6. FINALIZE: Declare the seed complete
   {{"action": "FINALIZE", "reasoning": "why it's complete"}}

## YOUR RESPONSE FORMAT

Think step by step, then output a single action.

```
THOUGHT: <your reasoning about what's needed next>

ACTION:
```json
{{"action": "...", ...}}
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
{{"action": "...", ...}}
```
```
'''


def _extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response."""
    import re

    # Find JSON in response (may be in code block)
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find bare JSON object
        brace_start = response.rfind("{")
        brace_end = response.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            json_str = response[brace_start:brace_end + 1]
        else:
            raise ValueError("No JSON found in response")

    # Clean up common issues
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    json_str = re.sub(r',\s*,', ',', json_str)

    return json.loads(json_str)


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
    """Create an empty Formula Seed structure."""
    return {
        "task_name": policy_id,
        "filter": {
            "keywords": []
        },
        "extract": {
            "fields": []
        },
        "compute": [],
        "output": ["VERDICT", "SCORE", "N_INCIDENTS"],
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
    """Ensure required extraction fields are present."""
    seed = deepcopy(seed)

    required = [
        {
            "name": "ACCOUNT_TYPE",
            "type": "enum",
            "values": {
                "firsthand": "Reviewer personally experienced it",
                "secondhand": "Reviewer heard from others",
                "general": "General statement without specific incident"
            }
        },
        {
            "name": "INCIDENT_SEVERITY",
            "type": "enum",
            "values": {
                "none": "No relevant incident",
                "mild": "Minor issue",
                "moderate": "Significant issue",
                "severe": "Serious issue"
            }
        },
        {
            "name": "SPECIFIC_INCIDENT",
            "type": "string",
            "description": "Brief description of what happened"
        }
    ]

    existing_names = {f.get("name") for f in seed["extract"]["fields"]}
    for field in required:
        if field["name"] not in existing_names:
            seed["extract"]["fields"].append(field)

    return seed


def _ensure_compute_operations(seed: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure minimal compute operations are present."""
    seed = deepcopy(seed)

    existing_names = {op.get("name") for op in seed["compute"]}

    # Add N_INCIDENTS if missing
    if "N_INCIDENTS" not in existing_names:
        seed["compute"].append({
            "name": "N_INCIDENTS",
            "op": "count",
            "where": {"ACCOUNT_TYPE": "firsthand", "INCIDENT_SEVERITY": ["mild", "moderate", "severe"]}
        })

    # Add BASE_POINTS if missing
    if "BASE_POINTS" not in existing_names:
        seed["compute"].append({
            "name": "BASE_POINTS",
            "op": "sum",
            "expr": "CASE WHEN INCIDENT_SEVERITY = 'severe' THEN 15 WHEN INCIDENT_SEVERITY = 'moderate' THEN 8 WHEN INCIDENT_SEVERITY = 'mild' THEN 3 ELSE 0 END",
            "where": {"ACCOUNT_TYPE": "firsthand"}
        })

    # Re-check after additions
    existing_names = {op.get("name") for op in seed["compute"]}

    # Add SCORE if missing
    if "SCORE" not in existing_names:
        seed["compute"].append({
            "name": "SCORE",
            "op": "expr",
            "expr": "BASE_POINTS"
        })

    # Add VERDICT if missing
    if "VERDICT" not in existing_names:
        seed["compute"].append({
            "name": "VERDICT",
            "op": "case",
            "source": "SCORE",
            "rules": [
                {"when": ">= 8", "then": "Critical Risk"},
                {"when": ">= 4", "then": "High Risk"},
                {"else": "Low Risk"}
            ]
        })

    return seed


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
