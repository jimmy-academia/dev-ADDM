"""Phase 1: Agenda -> Terms + Verdict Rules.

Input: agenda string (natural language)
Output: terms + verdict spec (no scoring)
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from addm.llm import LLMService
from addm.utils.debug_logger import get_debug_logger
from addm.utils.output import output

from .phase1_prompts import (
    OBSERVE_PROMPT,
    EXTRACT_TERMS_PROMPT,
    EXTRACT_VERDICTS_PROMPT,
    REFINE_VERDICTS_PROMPT,
)
from .phase1_helpers import (
    extract_json_from_response,
    parse_json_safely,
    normalize_terms,
    normalize_verdict_rules,
    validate_terms,
    validate_verdict_spec,
    referenced_fields_from_rules,
    accumulate_usage,
)

logger = logging.getLogger(__name__)

# Type alias for progress callback
ProgressCallback = Callable[[int, str, float, str], None]


# =============================================================================
# Step 0: OBSERVE (locate sections)
# =============================================================================

async def _observe(
    agenda: str,
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Locate terms block and verdict rules block."""
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

    json_str = extract_json_from_response(response)
    data = parse_json_safely(json_str)

    return data, usage


# =============================================================================
# Step 1: Extract Verdict Rules
# =============================================================================

async def _extract_verdicts(
    verdict_rules_block: str,
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract verdict labels and rules from verdict rules block."""
    prompt = EXTRACT_VERDICTS_PROMPT.format(verdict_rules_block=verdict_rules_block)
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={
            "sample_id": policy_id,
            "phase": "phase1_verdicts",
            "step": "extract_verdicts",
            "policy_id": policy_id,
        },
    )
    usage["latency_ms"] = (time.time() - start_time) * 1000

    json_str = extract_json_from_response(response)
    data = parse_json_safely(json_str)

    if not data.get("verdicts") or not data.get("rules"):
        raise ValueError("No verdicts or rules extracted from verdict rules block")

    return data, usage


# =============================================================================
# Step 2: Ground / Repair Verdict Rules
# =============================================================================

async def _refine_verdicts(
    verdict_rules_block: str,
    terms_block: str,
    verdict_data: Dict[str, Any],
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Refine verdict rules using definitions to ground implicit qualifiers."""
    verdict_json = json.dumps(verdict_data, indent=2, ensure_ascii=True)
    prompt = REFINE_VERDICTS_PROMPT.format(
        terms_block=terms_block,
        verdict_rules_block=verdict_rules_block,
        verdict_json=verdict_json,
    )
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={
            "sample_id": policy_id,
            "phase": "phase1_verdicts_refine",
            "step": "refine_verdicts",
            "policy_id": policy_id,
        },
    )
    usage["latency_ms"] = (time.time() - start_time) * 1000

    json_str = extract_json_from_response(response)
    data = parse_json_safely(json_str)

    if not data.get("verdicts") or not data.get("rules"):
        raise ValueError("No verdicts or rules extracted from refine step")

    return data, usage


# =============================================================================
# Step 3: Extract Terms
# =============================================================================

def _format_required_fields(fields: List[str]) -> str:
    """Format required fields for prompt readability."""
    if not fields:
        return "None"
    return "\n".join(f"- {f}" for f in fields)


async def _extract_terms(
    terms_block: str,
    required_fields: List[str],
    llm: LLMService,
    policy_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract term definitions from definitions block."""
    required_fields_text = _format_required_fields(required_fields)
    prompt = EXTRACT_TERMS_PROMPT.format(
        terms_block=terms_block,
        required_fields=required_fields_text,
    )
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={
            "sample_id": policy_id,
            "phase": "phase1_terms",
            "step": "extract_terms",
            "policy_id": policy_id,
        },
    )
    usage["latency_ms"] = (time.time() - start_time) * 1000

    json_str = extract_json_from_response(response)
    data = parse_json_safely(json_str)

    terms = data.get("terms", [])
    if not terms:
        raise ValueError("No terms extracted from definitions block")

    return terms, usage


# =============================================================================
# Entry Point
# =============================================================================

async def generate_verdict_and_terms(
    agenda: str,
    policy_id: str,
    llm: LLMService,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Phase 1 outputs: verdict spec + term definitions.

    Returns:
        (spec, usage)
        spec = {
          "terms": [TermDef, ...],
          "verdict": {"verdicts": [...], "rules": [...]}
        }
    """
    all_usages = []

    # Route debug logs to debug/phase1.jsonl if enabled
    if debug_logger := get_debug_logger():
        debug_logger.set_phase1_mode()

    def report(step: str, progress: float, detail: str = "") -> None:
        if progress_callback:
            progress_callback(1, step, progress, detail)

    # Step 0: OBSERVE
    report("OBSERVE", 5, "locating sections")
    output.status("[Phase 1] Step 0/4: OBSERVE (locating sections)...")
    observations, usage0 = await _observe(agenda, llm, policy_id)
    all_usages.append(usage0)

    terms_block = observations.get("terms_block", "") or observations.get("definitions_block", "")
    verdict_block = observations.get("verdict_rules_block", "") or observations.get("verdicts_block", "")
    verdict_labels = observations.get("verdict_labels", []) or []
    if verdict_labels:
        output.status(f"[Phase 1] OBSERVE: {len(verdict_labels)} verdict labels found")

    if not terms_block:
        logger.warning("OBSERVE returned empty terms_block; using full agenda")
        terms_block = agenda
    if not verdict_block:
        logger.warning("OBSERVE returned empty verdict_rules_block; using full agenda")
        verdict_block = agenda

    # Step 1: Verdicts (raw)
    report("VERDICTS", 25, "extracting verdict rules")
    output.status("[Phase 1] Step 1/4: Extracting verdict rules...")
    verdict_data_raw, usage1 = await _extract_verdicts(verdict_block, llm, policy_id)
    all_usages.append(usage1)

    # Step 2: Ground / Repair Verdicts with definitions
    report("GROUND_RULES", 50, "grounding verdict rules to definitions")
    output.status("[Phase 1] Step 2/4: Grounding verdict rules to definitions...")
    verdict_data, usage2 = await _refine_verdicts(
        verdict_block,
        terms_block,
        verdict_data_raw,
        llm,
        policy_id,
    )
    all_usages.append(usage2)

    verdicts = verdict_data.get("verdicts", [])
    rules = normalize_verdict_rules(verdict_data.get("rules", []))
    output.status(f"[Phase 1] Extracted {len(verdicts)} verdicts, {len(rules)} rules")

    referenced = referenced_fields_from_rules(rules)
    if not referenced:
        raise ValueError("No referenced fields found in verdict rules")
    output.status(f"[Phase 1] Referenced fields: {', '.join(referenced)}")

    # Step 3: Terms (only those referenced by rules)
    report("TERMS", 75, "extracting terms")
    output.status("[Phase 1] Step 3/4: Extracting terms...")
    raw_terms, usage3 = await _extract_terms(terms_block, referenced, llm, policy_id)
    all_usages.append(usage3)
    terms = normalize_terms(raw_terms)

    # Keep only required fields and ensure all are present
    terms = [t for t in terms if t.get("name") in referenced]
    missing_terms = [f for f in referenced if f not in {t.get("name") for t in terms}]
    if missing_terms:
        raise ValueError(f"Missing term definitions for: {missing_terms}")

    output.status(f"[Phase 1] Extracted {len(terms)} terms")

    term_errors = validate_terms(terms)
    if term_errors:
        raise ValueError(f"Term validation failed: {term_errors}")

    verdict_errors = validate_verdict_spec(verdicts, rules, terms)
    if verdict_errors:
        raise ValueError(f"Verdict validation failed: {verdict_errors}")

    spec = {
        "terms": terms,
        "verdict": {
            "verdicts": verdicts,
            "rules": rules,
        },
    }

    usage = accumulate_usage(all_usages)

    # Clear phase1 mode (avoid leaking into other phases)
    if debug_logger := get_debug_logger():
        debug_logger.clear_context()

    output.status(f"[Phase 1] Complete: {len(terms)} terms, {len(rules)} rules")

    return spec, usage


__all__ = [
    "generate_verdict_and_terms",
]
