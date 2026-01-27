"""Phase 1: Formula Seed Generation.

LLM reads agenda/query and produces a Formula Seed (executable JSON specification).

Uses part-by-part extraction with 3 focused LLM calls:
0. OBSERVE - Format-agnostic semantic analysis (guides downstream steps)
1. Extract terms/field definitions
2. Extract verdict rules

Entry point: generate_formula_seed()

The OBSERVE step (Step 0) enables format-agnostic parsing - it can handle
markdown, XML, prose, or other structured formats and extract semantic
information to guide the downstream extraction steps.

Helper functions in phase1_helpers.py:
- extract_yaml_from_response(), parse_yaml_safely()
- validate_formula_seed()
- accumulate_usage()
"""

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from addm.llm import LLMService
from addm.utils.debug_logger import get_debug_logger
from addm.utils.output import output

from .phase1_prompts import (
    OBSERVE_PROMPT,
    EXTRACT_TERMS_PROMPT,
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
# OBSERVE: Format-Agnostic Semantic Analysis (Step 0)
# =============================================================================

async def _observe(
    agenda: str,
    llm: "LLMService",
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Analyze agenda structure format-agnostically.

    This is Step 0 of Phase 1. It analyzes the agenda (in any format: markdown,
    XML, prose) and extracts semantic information to guide downstream extraction.

    Args:
        agenda: The task agenda (policy description in any format)
        llm: LLM service for API calls
        policy_id: Policy identifier for logging

    Returns:
        Tuple of (observations_dict, usage_dict)
        observations_dict contains:
        - policy_type: "count_rule_based" | "scoring" | "signal_rule_based"
        - extraction_fields: List of {name, description, values}
        - verdict_rules: List of rule dicts
        - verdicts: List of verdict labels
        - has_scoring: bool
        - key_concepts: List of relevant terms
    """
    import json

    prompt = OBSERVE_PROMPT.format(agenda=agenda)
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={
            "sample_id": policy_id,
            "phase": "phase1_observe",
            "step": "observe",
            "policy_id": policy_id,
        },
    )
    usage["latency_ms"] = (time.time() - start_time) * 1000

    # Extract JSON from response
    json_str = response
    if "```json" in response:
        json_str = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        json_str = response.split("```")[1].split("```")[0].strip()

    try:
        observations = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"OBSERVE JSON parse failed: {e}")
        # Return minimal observations on parse failure
        observations = {
            "policy_type": "count_rule_based",
            "extraction_fields": [],
            "verdict_rules": [],
            "verdicts": [],
            "has_scoring": False,
            "key_concepts": [],
        }

    logger.info(f"OBSERVE: policy_type={observations.get('policy_type')}, "
                f"{len(observations.get('extraction_fields', []))} fields, "
                f"{len(observations.get('verdicts', []))} verdicts")

    return observations, usage

# Type alias for progress callback
ProgressCallback = Callable[[int, str, float, str], None]


# =============================================================================
# ENTRY POINT
# =============================================================================

async def generate_formula_seed(
    agenda: str,
    policy_id: str,
    llm: LLMService,
    config: "AMOSConfig" = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Formula Seed from policy agenda.

    Entry point for Phase 1. Extracts each query section separately:
    1. Terms/Definitions → field definitions
    2. Scoring System → point values (if present)
    3. Verdict Rules → thresholds and conditions

    Args:
        agenda: The task agenda/query prompt (natural language policy description)
        policy_id: Policy identifier (e.g., "G1_allergy_V2")
        llm: LLM service for API calls
        config: AMOSConfig (unused, kept for API compatibility)
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (formula_seed, usage_dict)

    Raises:
        ValueError: If seed generation or validation fails
    """
    start_time = time.perf_counter()
    all_usages = []

    # Set debug logger to Phase 1 mode - all LLM calls go to debug/phase1.jsonl
    if debug_logger := get_debug_logger():
        debug_logger.set_phase1_mode()

    def report(step: str, progress: float, detail: str = "") -> None:
        if progress_callback:
            progress_callback(1, step, progress, detail)

    # Step 0: OBSERVE - Format-agnostic semantic analysis
    report("OBSERVE", 5, "analyzing agenda structure")
    output.status(f"[Phase 1] Step 0/3: Analyzing agenda structure (OBSERVE)...")

    observations, usage0 = await _observe(agenda, llm, policy_id)
    all_usages.append(usage0)

    policy_type_hint = observations.get("policy_type", "count_rule_based")
    known_fields = {f["name"].upper() for f in observations.get("extraction_fields", [])}

    output.status(f"[Phase 1] OBSERVE: {policy_type_hint}, {len(known_fields)} fields identified")
    logger.info(f"OBSERVE hints: policy_type={policy_type_hint}, known_fields={known_fields}")

    # Parse query into sections (using observations as hints)
    report("PARSE", 10, "parsing query sections")
    output.status(f"[Phase 1] Parsing query sections...")

    sections = _parse_query_sections(agenda)
    logger.info(f"Parsed query into sections: {list(sections.keys())}")

    # Extract task name from header
    header = sections.get("header", "")
    task_name_match = re.search(r'^#\s*(.+?)(?:\n|$)', header)
    task_name = task_name_match.group(1).strip() if task_name_match else policy_id

    # Step 1: Extract terms (guided by OBSERVE hints)
    report("TERMS", 15, "extracting terms")
    output.status(f"[Phase 1] Step 1/3: Extracting terms...")

    terms_section = sections.get("terms", "")
    if not terms_section:
        terms_section = header
        logger.warning("No explicit terms section found, using header")

    terms, usage1 = await _extract_terms_from_section(terms_section, llm, policy_id)
    all_usages.append(usage1)
    output.status(f"[Phase 1] Extracted {len(terms)} terms")
    report("TERMS", 40, f"{len(terms)} terms")

    # Step 2: Extract verdicts and rules (guided by OBSERVE hints)
    report("VERDICTS", 45, "extracting verdicts")
    output.status(f"[Phase 1] Step 2/3: Extracting verdict rules...")

    verdicts_section = sections.get("verdicts", "")
    if not verdicts_section:
        verdicts_section = header
        logger.warning("No explicit verdicts section found, using header")

    verdicts_data, usage2 = await _extract_verdicts_from_section(
        verdicts_section, terms, llm, policy_id
    )
    all_usages.append(usage2)
    output.status(f"[Phase 1] Extracted {len(verdicts_data.get('verdicts', []))} verdicts")
    report("VERDICTS", 70, f"{len(verdicts_data.get('rules', []))} rules")

    # Merge discovered terms from verdict rules into terms list
    discovered_terms = verdicts_data.pop("discovered_terms", [])
    if discovered_terms:
        logger.info(f"Adding {len(discovered_terms)} discovered terms from verdict rules")
        output.status(f"[Phase 1] Discovered {len(discovered_terms)} additional fields from verdict rules")
        terms = terms + discovered_terms

    # Step 3: Combine parts
    report("COMBINE", 75, "combining parts")
    output.status(f"[Phase 1] Step 3/3: Combining extracted parts...")

    yaml_data = _combine_parts_to_yaml(terms, verdicts_data, task_name)
    logger.info(f"Combined PolicyYAML: {len(yaml_data.get('terms', []))} terms")

    # Step 5: Validate PolicyYAML
    report("VALIDATE_YAML", 80, "validating yaml")
    try:
        validate_policy_yaml(yaml_data)
        output.status(f"[Phase 1] PolicyYAML validation: passed")
    except PolicyYAMLValidationError as e:
        logger.warning(f"PolicyYAML validation failed: {e}")
        output.warn(f"[Phase 1] PolicyYAML validation failed: {e}")
        raise ValueError(f"PolicyYAML validation failed: {e}")
    report("VALIDATE_YAML", 85, "passed")

    # Step 6: Compile to Formula Seed
    report("COMPILE", 88, "compiling seed")
    output.status(f"[Phase 1] Compiling to Formula Seed...")

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
        output.warn(f"[Phase 1] Seed validation: {len(errors)} errors")
        logger.warning(f"Compiled seed has validation errors: {errors}")
        raise ValueError(f"Compiled seed validation failed: {errors}")
    else:
        output.status(f"[Phase 1] Seed validation: passed")
    report("VALIDATE_SEED", 98, "passed")

    # Return seed
    total_usage = accumulate_usage(all_usages)
    wall_clock_ms = (time.perf_counter() - start_time) * 1000
    total_usage["wall_clock_ms"] = wall_clock_ms

    seed_clean = {k: v for k, v in seed.items() if not k.startswith("_") or k == "_policy_yaml"}

    n_terms = len(yaml_data.get("terms", []))
    n_rules = len(yaml_data.get("rules", []))
    output.status(f"[Phase 1] Complete: {n_terms} terms, {n_rules} rules")

    return seed_clean, total_usage


# Backward compatibility alias
generate_formula_seed_with_config = generate_formula_seed


# =============================================================================
# Query Parsing Helpers
# =============================================================================

def _parse_query_sections(query: str) -> Dict[str, str]:
    """Parse query into sections based on markdown headers."""
    sections = {}
    parts = re.split(r'\n##\s+', query)

    if parts:
        sections["header"] = parts[0].strip()

    for part in parts[1:]:
        lines = part.split('\n', 1)
        if lines:
            section_name = lines[0].strip().lower()
            section_content = lines[1].strip() if len(lines) > 1 else ""

            if "definition" in section_name or "term" in section_name:
                sections["terms"] = section_content
            elif "verdict" in section_name or "rule" in section_name:
                sections["verdicts"] = section_content
            else:
                sections[section_name] = section_content

    return sections


def _discover_fields_from_rules(
    rules: List[Dict[str, Any]],
    known_fields: set,
) -> Dict[str, Dict[str, Any]]:
    """Discover fields used in rule conditions that aren't in extracted terms.

    This handles cases where verdict rules reference fields not in the Definitions
    section (e.g., date_outcome in V1 policies). We infer the field type and values
    from how they're used in conditions.

    Args:
        rules: List of extracted verdict rules with conditions
        known_fields: Set of field names already extracted from terms (uppercase)

    Returns:
        Dict mapping discovered field names to their inferred definitions:
        {
            "DATE_OUTCOME": {
                "name": "DATE_OUTCOME",
                "type": "enum",
                "values": {"positive", "negative", ...}
            }
        }
    """
    discovered = {}

    for rule in rules:
        # Skip default rules and score-condition rules
        if rule.get("default") or "condition" in rule:
            continue

        conditions = rule.get("conditions", [])
        for cond in conditions:
            field = cond.get("field", "").upper()
            if not field:
                continue

            # Skip if field is already known
            if field in known_fields:
                continue

            # Check if it's a fuzzy match to known fields
            is_fuzzy_match = False
            for kf in known_fields:
                if field in kf or kf in field:
                    is_fuzzy_match = True
                    break
            if is_fuzzy_match:
                continue

            # This is a truly new field - collect its values
            cond_values = cond.get("values", [])
            if not cond_values:
                continue

            if field not in discovered:
                discovered[field] = {
                    "name": field,
                    "type": "enum",
                    "values": set(),
                }

            # Add values from this condition
            for v in cond_values:
                discovered[field]["values"].add(str(v).lower())

    # Convert value sets to lowercase sets (consistent with field_values format)
    return discovered


# =============================================================================
# LLM Extraction Functions
# =============================================================================

async def _extract_terms_from_section(
    section: str,
    llm: LLMService,
    policy_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract terms/field definitions from the Definitions section."""
    prompt = EXTRACT_TERMS_PROMPT.format(section=section)
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={
            "sample_id": policy_id,
            "phase": "phase1_extract_terms",
            "step": "extract_terms",
            "policy_id": policy_id,
        },
    )
    usage["latency_ms"] = (time.time() - start_time) * 1000

    # Note: LLMService handles debug logging automatically

    yaml_str = extract_yaml_from_response(response)
    try:
        data = parse_yaml_safely(yaml_str)
    except ValueError as e:
        logger.error(f"YAML parse failed for terms extraction")
        logger.debug(f"Raw response:\n{response[:1000]}")
        raise ValueError(f"Failed to parse terms YAML: {e}")

    terms = data.get("terms", [])
    if not terms:
        logger.warning(f"No terms found in response. Data: {data}")
        raise ValueError("No terms extracted from section")

    logger.info(f"Extracted {len(terms)} terms from definitions section")
    return terms, usage


async def _extract_verdicts_from_section(
    section: str,
    terms: List[Dict[str, Any]],
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract verdict rules from the Verdict Rules section."""
    # Build context from terms
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

    context_parts.append("\n## Policy type: count_rule_based")
    context = "\n".join(context_parts)

    prompt = EXTRACT_VERDICTS_PROMPT.format(section=section, context=context)
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={
            "sample_id": policy_id,
            "phase": "phase1_extract_verdicts",
            "step": "extract_verdicts",
            "policy_id": policy_id,
        },
    )
    usage["latency_ms"] = (time.time() - start_time) * 1000

    # Note: LLMService handles debug logging automatically

    yaml_str = extract_yaml_from_response(response)
    data = parse_yaml_safely(yaml_str)

    verdicts = data.get("verdicts", [])
    rules = data.get("rules", [])

    if not verdicts:
        raise ValueError("No verdicts extracted from section")
    if not rules:
        raise ValueError("No rules extracted from section")

    # Build lookup tables for validation (from extracted terms only)
    valid_fields = {t.get("name", "").upper() for t in terms if t.get("name")}
    field_values = {}
    for t in terms:
        name = t.get("name", "").upper()
        values = t.get("values", [])
        if isinstance(values, dict):
            field_values[name] = {str(v).lower() for v in values.keys()}
        elif isinstance(values, list):
            field_values[name] = {str(v).lower() for v in values}

    # Discover fields from rule conditions that aren't in extracted terms
    # This handles fields like date_outcome that appear in verdict rules but not definitions
    discovered_fields = _discover_fields_from_rules(rules, valid_fields)
    if discovered_fields:
        logger.info(f"Discovered {len(discovered_fields)} fields from rule conditions: {list(discovered_fields.keys())}")
        # Add discovered fields to valid_fields and field_values
        for field_name, field_data in discovered_fields.items():
            valid_fields.add(field_name)
            field_values[field_name] = field_data["values"]

    # Repair rules if needed
    repaired_rules = []
    for rule in rules:
        if rule.get("default"):
            repaired_rules.append(rule)
            continue

        # For scoring policies, rules may have "condition: score >= X" format
        if "condition" in rule and "conditions" not in rule:
            repaired_rules.append(rule)
            continue

        # Validate conditions
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
                    matched = False
                    for av in allowed_values:
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
            if rule.get("verdict") == verdicts[-1]:
                repaired_rules.append({"verdict": rule.get("verdict"), "default": True})
                logger.warning(f"Converted rule to default: {rule.get('verdict')}")

    # Ensure at least one default rule exists
    has_default = any(r.get("default") for r in repaired_rules)
    if not has_default and verdicts:
        repaired_rules.append({"verdict": verdicts[-1], "default": True})
        logger.warning(f"Added missing default rule for: {verdicts[-1]}")

    logger.info(f"Extracted {len(verdicts)} verdicts and {len(repaired_rules)} rules")

    # Convert discovered fields to term format for inclusion in seed
    discovered_terms = []
    for field_name, field_data in discovered_fields.items():
        # Convert to dict format with values as descriptions
        values_dict = {v: f"{field_name} is {v}" for v in field_data["values"]}
        discovered_terms.append({
            "name": field_name,
            "type": "enum",
            "values": values_dict,
            "_discovered": True,  # Mark as discovered (for debugging)
        })

    return {"verdicts": verdicts, "rules": repaired_rules, "discovered_terms": discovered_terms}, usage


# =============================================================================
# Combine Parts
# =============================================================================

def _combine_parts_to_yaml(
    terms: List[Dict[str, Any]],
    verdicts_data: Dict[str, Any],
    task_name: str,
) -> Dict[str, Any]:
    """Combine extracted parts into a complete PolicyYAML structure."""
    rules = verdicts_data.get("rules", [])
    filtered_terms = [t for t in terms if t.get("values")]

    yaml_data = {
        "policy_type": "count_rule_based",
        "task_name": task_name,
        "terms": filtered_terms,
        "verdicts": verdicts_data.get("verdicts", []),
        "rules": rules,
    }

    return yaml_data
