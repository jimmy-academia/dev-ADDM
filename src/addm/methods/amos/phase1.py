"""Phase 1: Formula Seed Generation.

LLM reads agenda/query and produces a Formula Seed (executable JSON specification).

Uses PARTS approach: Part-by-part extraction with 3 focused LLM calls:
1. Extract terms/field definitions
2. Extract scoring system (if present)
3. Extract verdict rules

File Structure (top-down):
1. Entry Point: generate_formula_seed_with_config()
2. Main Function: generate_formula_seed_parts() - orchestrates the 3 LLM calls
3. Extraction Functions: _extract_terms/scoring/verdicts_from_section()

Helper functions in phase1_helpers.py:
- extract_yaml_from_response(), parse_yaml_safely()
- validate_formula_seed(), validate_field_references()
- accumulate_usage()

Deprecated HYBRID approach moved to: backup/phase1_hybrid.py
"""

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
    EXTRACT_TERMS_PROMPT,
    EXTRACT_SCORING_PROMPT,
    EXTRACT_VERDICTS_PROMPT,
)
from .phase1_helpers import (
    extract_yaml_from_response,
    parse_yaml_safely,
    accumulate_usage,
    validate_formula_seed,
)
from .seed_compiler import compile_yaml_to_seed, validate_policy_yaml, PolicyYAMLValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# ENTRY POINT
# =============================================================================

async def generate_formula_seed_with_config(
    agenda: str,
    policy_id: str,
    llm: LLMService,
    config: "AMOSConfig",
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Formula Seed from policy agenda.

    This is the main entry point for Phase 1. Uses part-by-part extraction
    with 3 focused LLM calls for reliable seed generation.

    Args:
        agenda: The task agenda/query prompt (natural language policy description)
        policy_id: Policy identifier (e.g., "G1_allergy_V2")
        llm: LLM service for API calls
        config: AMOSConfig (currently unused, kept for API compatibility)
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (formula_seed, usage_dict)

    Raises:
        ValueError: If seed generation or validation fails
    """
    return await generate_formula_seed_parts(
        agenda=agenda,
        policy_id=policy_id,
        llm=llm,
        progress_callback=progress_callback,
    )


# =============================================================================
# Part-by-Part Query Parsing and Extraction
# =============================================================================

def _parse_query_sections(query: str) -> Dict[str, str]:
    """Parse query into sections based on markdown headers.

    Args:
        query: The full query/agenda text

    Returns:
        Dict mapping section names to their content
    """
    sections = {}

    # Split by ## headers
    parts = re.split(r'\n##\s+', query)

    # First part is the header/intro
    if parts:
        sections["header"] = parts[0].strip()

    # Process remaining sections
    for part in parts[1:]:
        lines = part.split('\n', 1)
        if lines:
            section_name = lines[0].strip().lower()
            section_content = lines[1].strip() if len(lines) > 1 else ""

            # Normalize section names
            if "definition" in section_name or "term" in section_name:
                sections["terms"] = section_content
            elif "scoring" in section_name or "point" in section_name:
                sections["scoring"] = section_content
            elif "verdict" in section_name or "rule" in section_name:
                sections["verdicts"] = section_content
            else:
                # Store with original name
                sections[section_name] = section_content

    return sections


async def _extract_terms_from_section(
    section: str,
    llm: LLMService,
    policy_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract terms/field definitions from the Definitions section.

    Args:
        section: The "Definitions of Terms" section content
        llm: LLM service for API calls
        policy_id: Policy identifier

    Returns:
        Tuple of (terms_list, usage)
    """
    prompt = EXTRACT_TERMS_PROMPT.format(section=section)
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_parts", "step": "extract_terms", "policy_id": policy_id},
    )
    usage["latency_ms"] = (time.time() - start_time) * 1000

    # Log to debug file
    if debug_logger := get_debug_logger():
        debug_logger.log_llm_call(
            sample_id=policy_id,
            phase="phase1_extract_terms",
            prompt=prompt,
            response=response,
            model=llm._config.get("model", "unknown"),
            latency_ms=usage["latency_ms"],
        )

    # Parse YAML response
    yaml_str = extract_yaml_from_response(response)
    try:
        data = parse_yaml_safely(yaml_str)
    except ValueError as e:
        logger.error(f"YAML parse failed for terms extraction")
        logger.debug(f"Raw response:\n{response[:1000]}")
        logger.debug(f"Extracted YAML:\n{yaml_str[:1000]}")
        raise ValueError(f"Failed to parse terms YAML: {e}")

    terms = data.get("terms", [])
    if not terms:
        logger.warning(f"No terms found in response. Data: {data}")
        raise ValueError("No terms extracted from section")

    logger.info(f"Extracted {len(terms)} terms from definitions section")
    return terms, usage


async def _extract_scoring_from_section(
    section: str,
    terms: List[Dict[str, Any]],
    llm: LLMService,
    policy_id: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Extract scoring system from the Scoring section.

    Args:
        section: The "Scoring System" section content
        terms: Previously extracted terms (for context)
        llm: LLM service for API calls
        policy_id: Policy identifier

    Returns:
        Tuple of (scoring_dict or None, usage)
    """
    # Build terms summary with explicit values
    terms_parts = []
    for t in terms:
        name = t.get('name', 'UNKNOWN')
        values = t.get('values', [])
        terms_parts.append(f"- {name}:")
        if isinstance(values, dict):
            for v in values.keys():
                terms_parts.append(f"    * {v}")
        elif isinstance(values, list):
            for v in values:
                terms_parts.append(f"    * {v}")
    terms_summary = "\n".join(terms_parts)

    prompt = EXTRACT_SCORING_PROMPT.format(
        section=section,
        terms_summary=terms_summary,
    )
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_parts", "step": "extract_scoring", "policy_id": policy_id},
    )
    usage["latency_ms"] = (time.time() - start_time) * 1000

    # Log to debug file
    if debug_logger := get_debug_logger():
        debug_logger.log_llm_call(
            sample_id=policy_id,
            phase="phase1_extract_scoring",
            prompt=prompt,
            response=response,
            model=llm._config.get("model", "unknown"),
            latency_ms=usage["latency_ms"],
        )

    # Parse YAML response
    yaml_str = extract_yaml_from_response(response)
    data = parse_yaml_safely(yaml_str)

    policy_type = data.get("policy_type", "count_rule_based")

    if policy_type == "scoring":
        scoring = data.get("scoring", {})
        logger.info(f"Extracted scoring system: {len(scoring.get('modifiers', []))} modifiers")
        return {"policy_type": "scoring", "scoring": scoring}, usage
    else:
        logger.info("No scoring system found, using count_rule_based")
        return {"policy_type": "count_rule_based"}, usage


async def _extract_verdicts_from_section(
    section: str,
    terms: List[Dict[str, Any]],
    scoring: Optional[Dict[str, Any]],
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract verdict rules from the Verdict Rules section.

    Args:
        section: The "Verdict Rules" section content
        terms: Previously extracted terms
        scoring: Previously extracted scoring (or None)
        llm: LLM service for API calls
        policy_id: Policy identifier

    Returns:
        Tuple of (verdicts_dict, usage)
    """
    # Build context from terms and scoring with explicit values
    context_parts = ["## Available Terms and Values (USE ONLY THESE):"]
    for t in terms:
        name = t.get('name', 'UNKNOWN')
        values = t.get('values', [])
        context_parts.append(f"- {name}:")
        if isinstance(values, dict):
            for v in values.keys():
                context_parts.append(f"    * {v}")
        elif isinstance(values, list):
            for v in values:
                context_parts.append(f"    * {v}")

    if scoring and scoring.get("policy_type") == "scoring":
        context_parts.append("\n## Scoring:")
        context_parts.append(f"- Policy type: scoring")
        s = scoring.get("scoring", {})
        context_parts.append(f"- Outcome field: {s.get('outcome_field', 'unknown')}")
        context_parts.append(f"- Base points: {s.get('base_points', {})}")
    else:
        context_parts.append("\n## Policy type: count_rule_based")

    context = "\n".join(context_parts)

    prompt = EXTRACT_VERDICTS_PROMPT.format(
        section=section,
        context=context,
    )
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_parts", "step": "extract_verdicts", "policy_id": policy_id},
    )
    usage["latency_ms"] = (time.time() - start_time) * 1000

    # Log to debug file
    if debug_logger := get_debug_logger():
        debug_logger.log_llm_call(
            sample_id=policy_id,
            phase="phase1_extract_verdicts",
            prompt=prompt,
            response=response,
            model=llm._config.get("model", "unknown"),
            latency_ms=usage["latency_ms"],
        )

    # Parse YAML response
    yaml_str = extract_yaml_from_response(response)
    data = parse_yaml_safely(yaml_str)

    verdicts = data.get("verdicts", [])
    rules = data.get("rules", [])

    if not verdicts:
        raise ValueError("No verdicts extracted from section")
    if not rules:
        raise ValueError("No rules extracted from section")

    # Build lookup tables for validation
    valid_fields = {t.get("name", "").upper() for t in terms if t.get("name")}
    valid_fields.add("ACCOUNT_TYPE")  # Always valid

    # Build field -> valid values mapping
    field_values = {"ACCOUNT_TYPE": {"firsthand", "secondhand", "general"}}
    for t in terms:
        name = t.get("name", "").upper()
        values = t.get("values", [])
        if isinstance(values, dict):
            field_values[name] = {str(v).lower() for v in values.keys()}
        elif isinstance(values, list):
            field_values[name] = {str(v).lower() for v in values}

    repaired_rules = []
    for rule in rules:
        if rule.get("default"):
            repaired_rules.append(rule)
            continue

        # For scoring policies, rules may have "condition: score >= X" format
        if "condition" in rule and "conditions" not in rule:
            repaired_rules.append(rule)
            continue

        # Validate conditions: field references and values
        conditions = rule.get("conditions", [])
        valid_conditions = []
        for cond in conditions:
            field = cond.get("field", "").upper()

            # Validate/repair field name
            actual_field = None
            if field in valid_fields:
                actual_field = field
            else:
                for vf in valid_fields:
                    if field in vf or vf in field:
                        actual_field = vf
                        logger.warning(f"Repaired field reference: {field} -> {vf}")
                        break

            if not actual_field:
                logger.warning(f"Dropping condition with unknown field: {field}")
                continue

            # Validate/repair values
            cond_values = cond.get("values", [])
            allowed_values = field_values.get(actual_field, set())
            repaired_values = []
            for v in cond_values:
                v_lower = str(v).lower()
                if v_lower in allowed_values:
                    repaired_values.append(v)
                else:
                    # Try fuzzy match
                    matched = False
                    for av in allowed_values:
                        # Check if v is contained in av or vice versa
                        if v_lower in av or av in v_lower:
                            repaired_values.append(av)
                            logger.warning(f"Repaired value: {v} -> {av}")
                            matched = True
                            break
                    if not matched:
                        logger.warning(f"Dropping value {v} not in {actual_field} values: {allowed_values}")

            if repaired_values:
                new_cond = dict(cond)
                new_cond["field"] = actual_field
                new_cond["values"] = repaired_values
                valid_conditions.append(new_cond)

        if valid_conditions:
            new_rule = {k: v for k, v in rule.items() if k != "conditions"}
            new_rule["conditions"] = valid_conditions
            repaired_rules.append(new_rule)
        else:
            # No valid conditions - convert to default if it's the last verdict
            if rule.get("verdict") == verdicts[-1]:
                repaired_rules.append({"verdict": rule.get("verdict"), "default": True})
                logger.warning(f"Converted rule to default: {rule.get('verdict')}")

    # Ensure at least one default rule exists
    has_default = any(r.get("default") for r in repaired_rules)
    if not has_default and verdicts:
        repaired_rules.append({"verdict": verdicts[-1], "default": True})
        logger.warning(f"Added missing default rule for: {verdicts[-1]}")

    logger.info(f"Extracted {len(verdicts)} verdicts and {len(repaired_rules)} rules (after repair)")
    return {"verdicts": verdicts, "rules": repaired_rules}, usage


def _combine_parts_to_yaml(
    terms: List[Dict[str, Any]],
    scoring: Optional[Dict[str, Any]],
    verdicts_data: Dict[str, Any],
    task_name: str,
) -> Dict[str, Any]:
    """Combine extracted parts into a complete PolicyYAML structure.

    Args:
        terms: List of term definitions
        scoring: Scoring configuration or None
        verdicts_data: Verdicts and rules
        task_name: Description of the policy task

    Returns:
        Complete PolicyYAML dict
    """
    policy_type = scoring.get("policy_type", "count_rule_based") if scoring else "count_rule_based"

    # Process rules based on policy type
    rules = verdicts_data.get("rules", [])

    if policy_type == "scoring":
        # For scoring policies, PRESERVE score-based conditions
        # Don't convert to counts - let seed_compiler generate proper SUM operations
        processed_rules = []
        for rule in rules:
            new_rule = {"verdict": rule.get("verdict", "")}

            if rule.get("default"):
                new_rule["default"] = True
            elif "condition" in rule:
                # Preserve score-based condition: "score >= X"
                # This will be handled by _build_scoring_compute in seed_compiler
                new_rule["condition"] = rule.get("condition")
            elif "conditions" in rule:
                # Already has conditions list, use as-is
                new_rule["logic"] = rule.get("logic", "ANY")
                new_rule["conditions"] = rule.get("conditions", [])

            # Only add rule if it has condition, conditions, or is default
            if new_rule.get("default") or new_rule.get("condition") or new_rule.get("conditions"):
                processed_rules.append(new_rule)
            else:
                logger.warning(f"Skipping rule without conditions: {rule}")

        rules = processed_rules

    # Filter terms that have no values
    filtered_terms = [t for t in terms if t.get("values")]

    yaml_data = {
        "policy_type": policy_type,
        "task_name": task_name,
        "terms": filtered_terms,
        "verdicts": verdicts_data.get("verdicts", []),
        "rules": rules,
    }

    # Add scoring details if present
    if scoring and scoring.get("policy_type") == "scoring":
        yaml_data["scoring"] = scoring.get("scoring", {})

    return yaml_data


async def generate_formula_seed_parts(
    agenda: str,
    policy_id: str,
    llm: LLMService,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Formula Seed using part-by-part extraction.

    This approach extracts each query section separately:
    1. Terms/Definitions → field definitions
    2. Scoring System → point values (if present)
    3. Verdict Rules → thresholds and conditions

    Benefits over single-shot:
    - Each LLM call is simpler and more focused
    - Validation at each step catches errors early
    - Reduced variance because sub-tasks are well-defined

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

    # Step 0: Parse query into sections
    report("PARSE", 5, "parsing query sections")
    output.status(f"[Phase 1 Parts] Parsing query sections...")

    sections = _parse_query_sections(agenda)
    logger.info(f"Parsed query into sections: {list(sections.keys())}")

    # Extract task name from header
    header = sections.get("header", "")
    task_name_match = re.search(r'^#\s*(.+?)(?:\n|$)', header)
    task_name = task_name_match.group(1).strip() if task_name_match else policy_id

    # Step 1: Extract terms
    report("TERMS", 10, "extracting terms")
    output.status(f"[Phase 1 Parts] Step 1/3: Extracting terms...")

    terms_section = sections.get("terms", "")
    if not terms_section:
        # Fall back to using full header + any definition content
        terms_section = header
        logger.warning("No explicit terms section found, using header")

    terms, usage1 = await _extract_terms_from_section(terms_section, llm, policy_id)
    all_usages.append(usage1)
    output.status(f"[Phase 1 Parts] Extracted {len(terms)} terms")
    report("TERMS", 30, f"{len(terms)} terms")

    # Step 2: Extract scoring (if present)
    report("SCORING", 35, "checking for scoring")
    output.status(f"[Phase 1 Parts] Step 2/3: Checking for scoring system...")

    scoring_section = sections.get("scoring", "")
    if scoring_section:
        scoring, usage2 = await _extract_scoring_from_section(scoring_section, terms, llm, policy_id)
        all_usages.append(usage2)
        is_scoring = scoring.get("policy_type") == "scoring"
        output.status(f"[Phase 1 Parts] Scoring: {'yes' if is_scoring else 'no'}")
    else:
        scoring = {"policy_type": "count_rule_based"}
        output.status(f"[Phase 1 Parts] No scoring section found")
    report("SCORING", 50, scoring.get("policy_type", "unknown"))

    # Step 3: Extract verdicts and rules
    report("VERDICTS", 55, "extracting verdicts")
    output.status(f"[Phase 1 Parts] Step 3/3: Extracting verdict rules...")

    verdicts_section = sections.get("verdicts", "")
    if not verdicts_section:
        # Look for any rules in header
        verdicts_section = header
        logger.warning("No explicit verdicts section found, using header")

    verdicts_data, usage3 = await _extract_verdicts_from_section(
        verdicts_section, terms, scoring, llm, policy_id
    )
    all_usages.append(usage3)
    output.status(f"[Phase 1 Parts] Extracted {len(verdicts_data.get('verdicts', []))} verdicts")
    report("VERDICTS", 70, f"{len(verdicts_data.get('rules', []))} rules")

    # Step 4: Combine parts
    report("COMBINE", 75, "combining parts")
    output.status(f"[Phase 1 Parts] Combining extracted parts...")

    yaml_data = _combine_parts_to_yaml(terms, scoring, verdicts_data, task_name)
    logger.info(f"Combined PolicyYAML: {yaml_data.get('policy_type')}, {len(yaml_data.get('terms', []))} terms")

    # Step 5: Validate PolicyYAML
    report("VALIDATE_YAML", 80, "validating yaml")
    try:
        validate_policy_yaml(yaml_data)
        output.status(f"[Phase 1 Parts] PolicyYAML validation: passed")
    except PolicyYAMLValidationError as e:
        logger.warning(f"PolicyYAML validation failed: {e}")
        output.warn(f"[Phase 1 Parts] PolicyYAML validation failed: {e}")
        raise ValueError(f"PolicyYAML validation failed: {e}")
    report("VALIDATE_YAML", 85, "passed")

    # Step 6: Compile to Formula Seed
    report("COMPILE", 88, "compiling seed")
    output.status(f"[Phase 1 Parts] Compiling to Formula Seed...")

    try:
        seed = compile_yaml_to_seed(yaml_data, validate=False)
    except Exception as e:
        logger.error(f"Seed compilation failed: {e}")
        raise ValueError(f"Failed to compile PolicyYAML to seed: {e}")

    seed["_approach"] = "parts"
    seed["_policy_yaml"] = yaml_data
    report("COMPILE", 92, "complete")

    # Step 7: Validate Formula Seed
    report("VALIDATE_SEED", 94, "validating seed")
    errors = validate_formula_seed(seed)

    if errors:
        output.warn(f"[Phase 1 Parts] Seed validation: {len(errors)} errors")
        logger.warning(f"Compiled seed has validation errors: {errors}")
        raise ValueError(f"Compiled seed validation failed: {errors}")
    else:
        output.status(f"[Phase 1 Parts] Seed validation: passed")
    report("VALIDATE_SEED", 98, "passed")

    # Return seed
    total_usage = accumulate_usage(all_usages)
    wall_clock_ms = (time.perf_counter() - start_time) * 1000
    total_usage["wall_clock_ms"] = wall_clock_ms

    # Keep _policy_yaml for saving
    seed_clean = {k: v for k, v in seed.items() if not k.startswith("_") or k == "_policy_yaml"}

    n_terms = len(yaml_data.get("terms", []))
    n_rules = len(yaml_data.get("rules", []))
    output.status(f"[Phase 1 Parts] Complete: {n_terms} terms, {n_rules} rules")

    return seed_clean, total_usage
