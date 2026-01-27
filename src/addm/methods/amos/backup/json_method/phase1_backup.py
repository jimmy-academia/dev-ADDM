"""Phase 1: Formula Seed Generation.

LLM reads agenda/query and produces a Formula Seed (executable JSON specification).
Uses a fixed 3-step pipeline: OBSERVE → PLAN → ACT.

All approaches include validation and fix loop after generation.
"""

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Type alias for progress callback
ProgressCallback = Callable[[int, str, float, str], None]

from addm.llm import LLMService
from addm.utils.debug_logger import get_debug_logger
from addm.utils.output import output

from .phase1_prompts import (
    OBSERVE_PROMPT,
    PLAN_PROMPT,
    ACT_PROMPT,
    FIX_PROMPT,
)

logger = logging.getLogger(__name__)

# Maximum attempts to fix structural errors
MAX_FIX_ATTEMPTS = 3


# =============================================================================
# JSON Extraction and Usage Accumulation
# =============================================================================

def _extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks and common errors."""
    original = response
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

    # Find JSON object boundaries - handle nested braces properly
    brace_start = response.find("{")
    if brace_start >= 0:
        # Find matching closing brace by counting
        depth = 0
        in_string = False
        escape_next = False
        brace_end = -1
        for i, char in enumerate(response[brace_start:], start=brace_start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
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
                    brace_end = i
                    break
        if brace_end > brace_start:
            response = response[brace_start : brace_end + 1]

    # Try to parse as-is first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Apply common fixes iteratively
    fixed = response

    # Remove single-line comments
    fixed = re.sub(r'//.*$', '', fixed, flags=re.MULTILINE)

    # Remove trailing commas before } or ]
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)

    # Remove double commas
    fixed = re.sub(r',\s*,', ',', fixed)

    # Fix missing commas between elements (e.g., "value1" "value2" -> "value1", "value2")
    fixed = re.sub(r'"\s*\n\s*"', '",\n"', fixed)

    # Fix missing commas after } or ] before next element
    fixed = re.sub(r'([}\]])\s*\n\s*"', r'\1,\n"', fixed)
    fixed = re.sub(r'([}\]])\s*\n\s*\{', r'\1,\n{', fixed)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # More aggressive: try to fix unescaped quotes in string values
    try:
        # Try removing all control characters except standard whitespace
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', fixed)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Log the problematic area for debugging
        logger.warning(f"JSON parse failed at position {e.pos}: {e.msg}")
        logger.debug(f"Context around error: ...{fixed[max(0,e.pos-50):e.pos+50]}...")
        raise


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
        "latency_ms": 0.0,
    }
    for u in usages:
        total["prompt_tokens"] += u.get("prompt_tokens", 0)
        total["completion_tokens"] += u.get("completion_tokens", 0)
        total["cost_usd"] += u.get("cost_usd", 0.0)
        total["latency_ms"] += u.get("latency_ms", 0.0)
    return total


# =============================================================================
# Validation Functions
# =============================================================================

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
    stripped = expr.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        inner = stripped[1:-1]
        if any(keyword in inner for keyword in [' if ', ' else ', ' and ', ' or ']):
            return (
                f"{field_name}: Expression appears to be a string literal containing Python code. "
                f"Remove the outer quotes. Got: {expr[:50]}..."
            )

    if stripped.startswith("'") and stripped.endswith("'"):
        inner = stripped[1:-1]
        if any(keyword in inner for keyword in [' if ', ' else ', ' and ', ' or ']):
            return (
                f"{field_name}: Expression appears to be a string literal containing Python code. "
                f"Remove the outer quotes. Got: {expr[:50]}..."
            )

    return None


def _validate_enum_consistency(seed: Dict[str, Any]) -> List[str]:
    """Validate that compute.where values match extract.fields enum values.

    Catches mismatches like compute using "Moderate" when extract defines "Moderate incident".
    This is a warning-level validation (returns warnings, not blocking errors).

    Args:
        seed: Formula Seed dict

    Returns:
        List of warning messages about enum inconsistencies
    """
    warnings = []

    # Build map of field -> valid enum values (lowercased for comparison)
    enum_values: Dict[str, set] = {}
    for field in seed.get("extract", {}).get("fields", []):
        if field.get("type") == "enum":
            name = field.get("name", "")
            values = field.get("values", {})
            if isinstance(values, dict):
                # Store both original and lowercase for matching
                enum_values[name.upper()] = set(v.lower() for v in values.keys())

    # Check compute.where conditions
    for op in seed.get("compute", []):
        op_name = op.get("name", "unknown")
        where = op.get("where", {})

        if not isinstance(where, dict):
            continue

        for field, expected in where.items():
            # Skip logical operators and field/equals keys
            if field in ("and", "or", "field", "equals", "not_equals"):
                continue

            field_upper = field.upper()
            if field_upper not in enum_values:
                continue  # Not an enum field or unknown field

            valid_values = enum_values[field_upper]

            # Check each expected value
            expected_list = expected if isinstance(expected, list) else [expected]
            for e in expected_list:
                if not isinstance(e, str):
                    continue
                e_lower = e.lower()
                # Check if value exists exactly or is a substring/superstring of valid values
                exact_match = e_lower in valid_values
                partial_match = any(e_lower in v or v in e_lower for v in valid_values)

                if not exact_match and partial_match:
                    # This is a potential mismatch that fuzzy matching will handle,
                    # but we should warn about it
                    warnings.append(
                        f"compute.{op_name}: where value '{e}' is a partial match for {field} "
                        f"enum values: {sorted(valid_values)}. Consider using exact enum keys."
                    )
                elif not exact_match and not partial_match:
                    warnings.append(
                        f"compute.{op_name}: where value '{e}' not found in {field} "
                        f"enum values: {sorted(valid_values)}"
                    )

    return warnings


def _validate_formula_seed(seed: Dict[str, Any]) -> List[str]:
    """Validate Formula Seed structure, expressions, and field references.

    Args:
        seed: Formula Seed dict

    Returns:
        List of validation errors (empty list if valid)
    """
    errors = []

    # Required keys (filter is NO LONGER required - simplified schema)
    required_keys = ["extract", "compute", "output"]
    for key in required_keys:
        if key not in seed:
            errors.append(f"Missing required key: {key}")

    if errors:
        return errors  # Can't continue without required keys

    # Validate extract
    if "fields" not in seed["extract"]:
        errors.append("extract missing 'fields'")
    elif not isinstance(seed["extract"]["fields"], list):
        errors.append("extract.fields must be a list")
    else:
        # Check for duplicate field names
        seen_field_names = set()
        for i, field in enumerate(seed["extract"]["fields"]):
            if not isinstance(field, dict):
                continue
            field_name = field.get("name", "")
            if field_name:
                upper_name = field_name.upper()
                if upper_name in seen_field_names:
                    errors.append(
                        f"extract.fields: Duplicate field name '{field_name}'. "
                        f"Each field must appear exactly once."
                    )
                seen_field_names.add(upper_name)

        # Validate field values format (must be dict, not list)
        for i, field in enumerate(seed["extract"]["fields"]):
            if not isinstance(field, dict):
                continue
            field_name = field.get("name", f"field[{i}]")
            if field.get("type") == "enum":
                values = field.get("values")
                if values is not None and not isinstance(values, dict):
                    errors.append(
                        f"extract.fields[{field_name}].values must be a dict "
                        f"(e.g., {{\"value1\": \"desc1\"}}), not {type(values).__name__}"
                    )

    # Validate outcome_field and none_values (required for generalizable incident detection)
    extract = seed.get("extract", {})
    if "outcome_field" not in extract:
        errors.append(
            "extract.outcome_field is required (e.g., 'INCIDENT_SEVERITY', 'QUALITY_LEVEL')"
        )
    if "none_values" not in extract:
        errors.append(
            "extract.none_values is required (e.g., ['none', 'n/a'])"
        )
    elif not isinstance(extract.get("none_values"), list):
        errors.append("extract.none_values must be a list")

    # Validate compute
    if not isinstance(seed["compute"], list):
        errors.append("compute must be a list")
    elif len(seed["compute"]) == 0:
        errors.append("compute must not be empty - at minimum needs N_INCIDENTS and VERDICT operations")
    else:
        # Validate VERDICT operation exists
        has_verdict = False
        for op in seed["compute"]:
            if isinstance(op, dict) and op.get("name") == "VERDICT":
                has_verdict = True
                if op.get("op") != "case":
                    errors.append(
                        f"VERDICT operation must have op='case', got op='{op.get('op')}'"
                    )
                elif "rules" not in op:
                    errors.append("VERDICT case operation must have 'rules' list")
                break
        if not has_verdict:
            errors.append(
                "compute must include a VERDICT operation with op='case' and 'rules'"
            )

    # Validate output
    if not isinstance(seed["output"], list):
        errors.append("output must be a list")

    # Note: search_strategy validation removed - no longer part of simplified schema

    # Validate field references in compute operations
    field_errors = _validate_field_references(seed)
    errors.extend(field_errors)

    # Validate enum consistency (warnings, not blocking errors)
    enum_warnings = _validate_enum_consistency(seed)
    for warning in enum_warnings:
        logger.warning(f"Enum consistency warning: {warning}")

    # Add strict enum consistency validation that produces errors for complete mismatches
    # This catches cases where compute.where uses values that don't exist in extract.fields enums
    extract_fields = seed.get("extract", {}).get("fields", [])
    field_values = {}
    for f in extract_fields:
        if f.get("type") == "enum" and isinstance(f.get("values"), dict):
            field_values[f["name"]] = set(f["values"].keys())
            # Also store lowercase version for case-insensitive matching
            field_values[f["name"].upper()] = set(v.lower() for v in f["values"].keys())

    for op in seed.get("compute", []):
        where = op.get("where", {})
        if not isinstance(where, dict):
            continue
        for field_name, values in where.items():
            # Skip logical operators and special keys
            if field_name in ("and", "or", "field", "equals", "not_equals"):
                continue

            field_upper = field_name.upper()
            if field_upper not in field_values:
                continue  # Field not defined as enum, skip

            valid_values = field_values[field_upper]

            # Check each expected value
            values_list = values if isinstance(values, list) else [values]
            for v in values_list:
                if not isinstance(v, str):
                    continue
                v_lower = v.lower()
                # Check if value exists (exact match or fuzzy)
                exact_match = v_lower in valid_values
                partial_match = any(v_lower in vv or vv in v_lower for vv in valid_values)

                if not exact_match and not partial_match:
                    errors.append(
                        f"compute.{op.get('name')}: where value '{v}' not found in {field_name} "
                        f"enum values: {sorted(valid_values)}"
                    )

    return errors


def _validate_field_references(seed: Dict[str, Any]) -> List[str]:
    """Validate that compute operations reference valid extraction fields.

    Also validates that where clauses only reference enum fields (not string fields).

    Args:
        seed: Formula Seed dict

    Returns:
        List of field reference errors
    """
    errors = []

    # Get extraction field names and their types
    extraction_fields = set()
    field_types = {}  # field_name.upper() -> "enum" or "string"
    for field in seed.get("extract", {}).get("fields", []):
        field_name = field.get("name", "")
        field_type = field.get("type", "string")
        if field_name:
            extraction_fields.add(field_name.upper())
            extraction_fields.add(field_name)
            field_types[field_name.upper()] = field_type

    # Get computed value names
    computed_names = set()
    for i, op in enumerate(seed.get("compute", [])):
        if not isinstance(op, dict):
            errors.append(
                f"compute[{i}]: Expected dict but got {type(op).__name__}: {str(op)[:100]}"
            )
            continue
        name = op.get("name", "")
        if name:
            computed_names.add(name.upper())
            computed_names.add(name)

    # Pattern to find field references in CASE WHEN expressions
    case_when_pattern = re.compile(r"WHEN\s+(\w+)\s*=", re.IGNORECASE)

    for i, op in enumerate(seed.get("compute", [])):
        if not isinstance(op, dict):
            continue
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
                if field_ref in ("and", "or", "field", "equals", "not_equals"):
                    continue
                field_upper = field_ref.upper()
                if field_upper not in extraction_fields:
                    errors.append(
                        f"compute.{op_name}: where condition references unknown field '{field_ref}'. "
                        f"Available extraction fields: {sorted(extraction_fields)}"
                    )
                elif field_types.get(field_upper) == "string":
                    # Where clauses can only filter by enum fields, not string fields
                    errors.append(
                        f"compute.{op_name}: where condition references string field '{field_ref}'. "
                        f"Count/sum where clauses can only filter by enum fields. "
                        f"To count by this criterion, create an enum field instead."
                    )

        # Check case operations that reference extraction fields
        if op_type == "case":
            source = op.get("source", "")
            if source and source.upper() not in extraction_fields and source.upper() not in computed_names:
                later_computed = set()
                found = False
                for later_op in seed.get("compute", []):
                    if not isinstance(later_op, dict):
                        continue
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


# =============================================================================
# Plan-and-Act Pipeline Steps
# =============================================================================

async def _observe(
    agenda: str,
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Step 1: OBSERVE - Extract concepts from query without domain hints.

    Args:
        agenda: The task agenda/query prompt
        llm: LLM service for API calls
        policy_id: Policy identifier for context

    Returns:
        Tuple of (observations, usage)
    """
    output.status(f"[Phase 1] Step 1/3: OBSERVE - Analyzing query...")

    prompt = OBSERVE_PROMPT.format(agenda=agenda)
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_plan_act", "step": "observe", "policy_id": policy_id},
        response_format={"type": "json_object"},
    )
    latency_ms = (time.time() - start_time) * 1000
    usage["latency_ms"] = latency_ms

    # Log to debug file immediately
    if debug_logger := get_debug_logger():
        debug_logger.log_llm_call(
            sample_id=policy_id,
            phase="phase1_observe",
            prompt=prompt,
            response=response,
            model=llm._config.get("model", "unknown"),
            latency_ms=latency_ms,
        )

    try:
        observations = _extract_json_from_response(response)
    except json.JSONDecodeError as e:
        logger.warning(f"OBSERVE JSON parse failed: {e}")
        raise ValueError(f"Failed to parse OBSERVE response: {e}")

    return observations, usage


async def _plan(
    observations: Dict[str, Any],
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Step 2: PLAN - Create keyword and field strategy based on observations.

    Args:
        observations: Output from OBSERVE step
        llm: LLM service for API calls
        policy_id: Policy identifier for context

    Returns:
        Tuple of (plan, usage)
    """
    output.status(f"[Phase 1] Step 2/3: PLAN - Building search strategy...")

    prompt = PLAN_PROMPT.format(observations=json.dumps(observations, indent=2))
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_plan_act", "step": "plan", "policy_id": policy_id},
        response_format={"type": "json_object"},
    )
    latency_ms = (time.time() - start_time) * 1000
    usage["latency_ms"] = latency_ms

    # Log to debug file immediately
    if debug_logger := get_debug_logger():
        debug_logger.log_llm_call(
            sample_id=policy_id,
            phase="phase1_plan",
            prompt=prompt,
            response=response,
            model=llm._config.get("model", "unknown"),
            latency_ms=latency_ms,
        )

    try:
        plan = _extract_json_from_response(response)
    except json.JSONDecodeError as e:
        logger.warning(f"PLAN JSON parse failed: {e}")
        raise ValueError(f"Failed to parse PLAN response: {e}")

    return plan, usage


async def _act(
    observations: Dict[str, Any],
    plan: Dict[str, Any],
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Step 3: ACT - Generate Formula Seed following the plan.

    Args:
        observations: Output from OBSERVE step
        plan: Output from PLAN step
        llm: LLM service for API calls
        policy_id: Policy identifier for context

    Returns:
        Tuple of (formula_seed, usage)
    """
    output.status(f"[Phase 1] Step 3/3: ACT - Generating Formula Seed...")

    prompt = ACT_PROMPT.format(
        observations=json.dumps(observations, indent=2),
        plan=json.dumps(plan, indent=2),
    )
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_plan_act", "step": "act", "policy_id": policy_id},
        response_format={"type": "json_object"},
    )
    latency_ms = (time.time() - start_time) * 1000
    usage["latency_ms"] = latency_ms

    # Log to debug file immediately
    if debug_logger := get_debug_logger():
        debug_logger.log_llm_call(
            sample_id=policy_id,
            phase="phase1_act",
            prompt=prompt,
            response=response,
            model=llm._config.get("model", "unknown"),
            latency_ms=latency_ms,
        )

    seed = _extract_json_from_response(response)
    return seed, usage


# =============================================================================
# Main Entry Point
# =============================================================================

async def generate_formula_seed(
    agenda: str,
    policy_id: str,
    llm: LLMService,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Formula Seed from agenda prompt using Plan-and-Act approach.

    Phase 1 of AMOS: LLM reads agenda and produces executable specification.
    Seeds are generated fresh each run and saved to run directory (not globally cached).

    Uses a fixed 3-step pipeline: OBSERVE → PLAN → ACT.
    No domain-specific hints - keywords are discovered from the agenda text.

    Args:
        agenda: The task agenda/query prompt
        policy_id: Policy identifier (e.g., "G1_allergy_V2")
        llm: LLM service for API calls
        progress_callback: Optional callback for progress updates.
            Signature: callback(phase: int, step: str, progress: float, detail: str)

    Returns:
        Tuple of (formula_seed, usage_dict)
        - formula_seed: The generated Formula Seed specification
        - usage_dict: Token/cost usage from all LLM calls (includes wall_clock_ms)
    """
    start_time = time.perf_counter()
    all_usages = []

    def report(step: str, progress: float, detail: str = "") -> None:
        if progress_callback:
            progress_callback(1, step, progress, detail)

    # Step 1: OBSERVE
    report("OBSERVE", 10, "analyzing query")
    logger.info(f"Phase 1: Step 1/3 OBSERVE for {policy_id}")
    observations, usage1 = await _observe(agenda, llm, policy_id)
    all_usages.append(usage1)
    logger.debug(f"Observed primary topic: {observations.get('core_concepts', {}).get('primary_topic', 'unknown')}")
    report("OBSERVE", 25, "complete")

    # Step 2: PLAN
    report("PLAN", 30, "building strategy")
    logger.info(f"Phase 1: Step 2/3 PLAN for {policy_id}")
    plan, usage2 = await _plan(observations, llm, policy_id)
    all_usages.append(usage2)
    report("PLAN", 45, "complete")

    # Step 3: ACT
    report("ACT", 50, "generating seed")
    logger.info(f"Phase 1: Step 3/3 ACT for {policy_id}")
    seed, usage3 = await _act(observations, plan, llm, policy_id)
    all_usages.append(usage3)
    report("ACT", 70, "complete")

    # Store intermediates
    intermediates = {
        "observations": observations,
        "plan": plan,
    }
    seed["_approach"] = "plan_and_act"
    seed["_intermediates"] = intermediates

    # =========================================================================
    # Validation and Fix Loop
    # =========================================================================
    report("VALIDATE", 75, "checking seed")
    errors = _validate_formula_seed(seed)

    if errors:
        output.status(f"[Phase 1] Structural validation: {len(errors)} errors")
    else:
        output.status(f"[Phase 1] Structural validation: passed")
        report("VALIDATE", 95, "passed")

    fix_attempt = 0
    while errors and fix_attempt < MAX_FIX_ATTEMPTS:
        fix_attempt += 1
        report("FIX", 80 + fix_attempt * 5, f"attempt {fix_attempt}/{MAX_FIX_ATTEMPTS}")
        output.status(f"[Phase 1] Validation: {len(errors)} errors, fixing (attempt {fix_attempt}/{MAX_FIX_ATTEMPTS})...")
        logger.info(f"Formula Seed has {len(errors)} errors, attempting fix {fix_attempt}/{MAX_FIX_ATTEMPTS}")
        logger.debug(f"Errors: {errors}")

        # Build fix prompt
        seed_for_fix = {k: v for k, v in seed.items() if not k.startswith("_")}
        fix_prompt = FIX_PROMPT.format(
            errors="\n".join(f"- {e}" for e in errors),
            seed_json=json.dumps(seed_for_fix, indent=2),
        )

        fix_messages = [
            {"role": "user", "content": f"Fix this Formula Seed:\n\n{fix_prompt}"},
        ]

        start_time = time.time()
        fix_response, fix_usage = await llm.call_async_with_usage(
            fix_messages,
            context={"phase": "phase1_fix", "policy_id": policy_id, "attempt": fix_attempt},
        )
        fix_usage["latency_ms"] = (time.time() - start_time) * 1000
        all_usages.append(fix_usage)

        # Log fix attempt to debug file
        if debug_logger := get_debug_logger():
            debug_logger.log_llm_call(
                sample_id=policy_id,
                phase="phase1_fix",
                prompt=fix_prompt,
                response=fix_response,
                model=llm._config.get("model", "unknown"),
                latency_ms=fix_usage["latency_ms"],
                metadata={"attempt": fix_attempt, "errors": errors},
            )

        try:
            fixed_seed = _extract_json_from_response(fix_response)
            fixed_seed["_approach"] = "plan_and_act"
            fixed_seed["_intermediates"] = intermediates
            seed = fixed_seed
            errors = _validate_formula_seed(seed)
        except json.JSONDecodeError as e:
            logger.warning(f"Fix attempt {fix_attempt} failed to parse: {e}")

    if errors:
        raise ValueError(
            f"Formula Seed validation failed after {MAX_FIX_ATTEMPTS} fix attempts. "
            f"Errors: {errors}"
        )

    if fix_attempt > 0:
        output.status(f"[Phase 1] Formula Seed fixed after {fix_attempt} attempt(s)")
        logger.info(f"Formula Seed fixed after {fix_attempt} attempt(s)")

    # =========================================================================
    # Return seed
    # =========================================================================
    total_usage = _accumulate_usage(all_usages)

    # Add wall-clock timing
    wall_clock_ms = (time.perf_counter() - start_time) * 1000
    total_usage["wall_clock_ms"] = wall_clock_ms

    # Remove intermediate results before returning
    seed_clean = {k: v for k, v in seed.items() if not k.startswith("_")}

    return seed_clean, total_usage


async def generate_formula_seed_with_config(
    agenda: str,
    policy_id: str,
    llm: LLMService,
    config: "AMOSConfig",
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Formula Seed using settings from AMOSConfig.

    Convenience wrapper that delegates to generate_formula_seed.
    Config is accepted for API compatibility but only PLAN_AND_ACT is supported.

    Args:
        agenda: The task agenda/query prompt
        policy_id: Policy identifier (e.g., "G1_allergy_V2")
        llm: LLM service for API calls
        config: AMOSConfig (phase1_approach setting is ignored, always uses PLAN_AND_ACT)
        progress_callback: Optional callback for progress updates.

    Returns:
        Tuple of (formula_seed, usage_dict)
    """
    return await generate_formula_seed(
        agenda=agenda,
        policy_id=policy_id,
        llm=llm,
        progress_callback=progress_callback,
    )
