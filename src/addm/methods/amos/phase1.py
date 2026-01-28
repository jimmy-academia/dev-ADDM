"""Phase 1: Agenda -> Agenda Spec (terms + verdict rules).

Input: agenda string (natural language)
Output: agenda_spec (no scoring)
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from addm.llm import LLMService
from addm.utils.debug_logger import get_debug_logger
from addm.utils.output import output

from .phase1_prompts import (
    SEGMENT_AGENDA_PROMPT,
    EXTRACT_VERDICT_LABELS_PROMPT,
    EXTRACT_VERDICT_OUTLINE_PROMPT,
    SELECT_REQUIRED_TERMS_PROMPT,
    EXTRACT_TERM_DEF_PROMPT,
    COMPILE_CLAUSE_PROMPT,
)
from .phase1_helpers import (
    extract_json_from_response,
    parse_json_safely,
    normalize_segment_blocks,
    validate_segment_blocks,
    normalize_verdict_labels,
    validate_verdict_labels,
    normalize_verdict_outline,
    validate_verdict_outline,
    normalize_required_terms,
    validate_required_terms,
    normalize_term_definition,
    validate_term_definition,
    normalize_clause_spec,
    validate_clause_spec,
    dedupe_preserve_order,
    accumulate_usage,
)

logger = logging.getLogger(__name__)

# Type alias for progress callback
ProgressCallback = Callable[[int, str, float, str], None]


# =============================================================================
# LLM call helper with validation retries
# =============================================================================

def _with_errors(base_prompt: str, errors: List[str], data: Dict[str, Any]) -> str:
    error_lines = "\n".join(f"- {e}" for e in errors)
    data_json = json.dumps(data, indent=2, ensure_ascii=True)
    return (
        f"{base_prompt}\n\n"
        f"VALIDATION ERRORS:\n{error_lines}\n\n"
        f"CURRENT OUTPUT (JSON):\n{data_json}\n\n"
        "Fix the output to satisfy the errors. Return JSON ONLY."
    )


async def _call_llm_json_with_retries(
    prompt: str,
    llm: LLMService,
    context: Dict[str, Any],
    validator: Callable[[Dict[str, Any]], List[str]],
    max_retries: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    usages: List[Dict[str, Any]] = []
    base_prompt = prompt
    attempt = 0

    while True:
        messages = [{"role": "user", "content": prompt}]
        start_time = time.time()
        response, usage = await llm.call_async_with_usage(
            messages,
            context={**context, "attempt": attempt},
        )
        usage["latency_ms"] = (time.time() - start_time) * 1000
        usages.append(usage)

        data: Dict[str, Any] = {}
        errors: List[str] = []
        try:
            json_str = extract_json_from_response(response)
            data = parse_json_safely(json_str)
            errors = validator(data)
        except Exception as e:  # noqa: BLE001 - surface error via retries
            errors = [str(e)]

        if not errors:
            return data, usages

        attempt += 1
        if attempt > max_retries:
            raise ValueError(
                f"Validation failed after {max_retries} retries: {errors}"
            )

        prompt = _with_errors(base_prompt, errors, data)


# =============================================================================
# Step 0: Segment agenda
# =============================================================================

async def _segment_agenda(
    agenda: str,
    llm: LLMService,
    policy_id: str,
    max_retries: int,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    prompt = SEGMENT_AGENDA_PROMPT.format(agenda=agenda)

    def validator(data: Dict[str, Any]) -> List[str]:
        definitions, verdicts = normalize_segment_blocks(data)
        return validate_segment_blocks(definitions, verdicts)

    data, usages = await _call_llm_json_with_retries(
        prompt,
        llm,
        context={
            "sample_id": policy_id,
            "phase": "phase1_segment",
            "step": "segment_agenda",
            "policy_id": policy_id,
        },
        validator=validator,
        max_retries=max_retries,
    )

    definitions, verdicts = normalize_segment_blocks(data)
    return definitions, verdicts, usages


# =============================================================================
# Step 1: Extract verdict labels
# =============================================================================

async def _extract_verdict_labels(
    verdict_text: str,
    llm: LLMService,
    policy_id: str,
    max_retries: int,
) -> Tuple[List[str], str, List[Dict[str, Any]]]:
    prompt = EXTRACT_VERDICT_LABELS_PROMPT.format(verdict_text=verdict_text)

    def validator(data: Dict[str, Any]) -> List[str]:
        verdicts, default_verdict = normalize_verdict_labels(data)
        return validate_verdict_labels(verdicts, default_verdict)

    data, usages = await _call_llm_json_with_retries(
        prompt,
        llm,
        context={
            "sample_id": policy_id,
            "phase": "phase1_verdict_labels",
            "step": "extract_verdict_labels",
            "policy_id": policy_id,
        },
        validator=validator,
        max_retries=max_retries,
    )

    verdicts, default_verdict = normalize_verdict_labels(data)
    return verdicts, default_verdict, usages


# =============================================================================
# Step 2: Extract verdict outline (per verdict)
# =============================================================================

async def _extract_verdict_outline(
    verdict_text: str,
    target_verdict: str,
    default_verdict: str,
    llm: LLMService,
    policy_id: str,
    max_retries: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prompt = EXTRACT_VERDICT_OUTLINE_PROMPT.format(
        verdict_text=verdict_text,
        target_verdict=target_verdict,
        default_verdict=default_verdict,
    )

    def validator(data: Dict[str, Any]) -> List[str]:
        outline = normalize_verdict_outline(data, target_verdict, default_verdict)
        return validate_verdict_outline(outline, target_verdict, default_verdict)

    data, usages = await _call_llm_json_with_retries(
        prompt,
        llm,
        context={
            "sample_id": policy_id,
            "phase": "phase1_verdict_outline",
            "step": f"outline_{target_verdict}",
            "policy_id": policy_id,
        },
        validator=validator,
        max_retries=max_retries,
    )

    outline = normalize_verdict_outline(data, target_verdict, default_verdict)
    return outline, usages


# =============================================================================
# Step 3: Select required terms (per non-default verdict)
# =============================================================================

async def _select_required_terms(
    definitions_text: str,
    outline: Dict[str, Any],
    llm: LLMService,
    policy_id: str,
    max_retries: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    outline_json = json.dumps(outline, indent=2, ensure_ascii=True)
    prompt = SELECT_REQUIRED_TERMS_PROMPT.format(
        definitions_text=definitions_text,
        outline_json=outline_json,
    )

    def validator(data: Dict[str, Any]) -> List[str]:
        required = normalize_required_terms(data)
        return validate_required_terms(required, definitions_text)

    data, usages = await _call_llm_json_with_retries(
        prompt,
        llm,
        context={
            "sample_id": policy_id,
            "phase": "phase1_required_terms",
            "step": f"required_terms_{outline.get('verdict')}",
            "policy_id": policy_id,
        },
        validator=validator,
        max_retries=max_retries,
    )

    required = normalize_required_terms(data)
    return required, usages


# =============================================================================
# Step 4: Extract term definition (per term)
# =============================================================================

async def _extract_term_definition(
    definitions_text: str,
    term_title: str,
    llm: LLMService,
    policy_id: str,
    max_retries: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prompt = EXTRACT_TERM_DEF_PROMPT.format(
        definitions_text=definitions_text,
        term_title=term_title,
    )

    def validator(data: Dict[str, Any]) -> List[str]:
        term = normalize_term_definition(data, term_title)
        return validate_term_definition(term, term_title)

    data, usages = await _call_llm_json_with_retries(
        prompt,
        llm,
        context={
            "sample_id": policy_id,
            "phase": "phase1_term_def",
            "step": f"term_{term_title}",
            "policy_id": policy_id,
        },
        validator=validator,
        max_retries=max_retries,
    )

    term = normalize_term_definition(data, term_title)
    return term, usages


# =============================================================================
# Step 5: Compile clause (per clause)
# =============================================================================

async def _compile_clause(
    clause_text: str,
    terms_payload: List[Dict[str, Any]],
    llm: LLMService,
    policy_id: str,
    max_retries: int,
    step_label: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    terms_json = json.dumps(terms_payload, indent=2, ensure_ascii=True)
    prompt = COMPILE_CLAUSE_PROMPT.format(
        clause_text=clause_text,
        terms_json=terms_json,
    )

    term_values_map = {t["field"]: t["values"] for t in terms_payload}

    def validator(data: Dict[str, Any]) -> List[str]:
        clause = normalize_clause_spec(data)
        return validate_clause_spec(clause, term_values_map)

    data, usages = await _call_llm_json_with_retries(
        prompt,
        llm,
        context={
            "sample_id": policy_id,
            "phase": "phase1_compile_clause",
            "step": step_label,
            "policy_id": policy_id,
        },
        validator=validator,
        max_retries=max_retries,
    )

    clause = normalize_clause_spec(data)
    return clause, usages


# =============================================================================
# Entry Point
# =============================================================================

async def generate_verdict_and_terms(
    agenda: str,
    policy_id: str,
    llm: LLMService,
    progress_callback: Optional[ProgressCallback] = None,
    max_retries: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Phase 1 agenda spec.

    Returns:
        (agenda_spec, usage)
    """
    all_usages: List[Dict[str, Any]] = []

    # Route debug logs to debug/phase1.jsonl if enabled
    if debug_logger := get_debug_logger():
        debug_logger.set_phase1_mode()

    def report(step: str, progress: float, detail: str = "") -> None:
        if progress_callback:
            progress_callback(1, step, progress, detail)

    # Step 0: Segment
    report("SEGMENT", 5, "segmenting agenda")
    output.status("[Phase 1] Step 0/6: Segmenting agenda...")
    definitions_blocks, verdict_blocks, usages0 = await _segment_agenda(
        agenda,
        llm,
        policy_id,
        max_retries,
    )
    all_usages.extend(usages0)

    definitions_text = "\n\n".join(definitions_blocks)
    verdict_text = "\n\n".join(verdict_blocks)

    # Step 1: Verdict labels
    report("VERDICT_LABELS", 15, "extracting verdict labels")
    output.status("[Phase 1] Step 1/6: Extracting verdict labels...")
    verdicts, default_verdict, usages1 = await _extract_verdict_labels(
        verdict_text,
        llm,
        policy_id,
        max_retries,
    )
    all_usages.extend(usages1)
    output.status(
        f"[Phase 1] Verdicts: {len(verdicts)} (default={default_verdict})"
    )

    # Step 2: Verdict outlines (parallel per verdict)
    report("VERDICT_OUTLINES", 30, "extracting verdict outlines")
    output.status("[Phase 1] Step 2/6: Extracting verdict outlines...")
    outline_tasks = [
        _extract_verdict_outline(
            verdict_text,
            v,
            default_verdict,
            llm,
            policy_id,
            max_retries,
        )
        for v in verdicts
    ]
    outline_results = await asyncio.gather(*outline_tasks)
    outlines = []
    for outline, usages in outline_results:
        outlines.append(outline)
        all_usages.extend(usages)

    # Step 3: Required terms (parallel per non-default verdict)
    report("REQUIRED_TERMS", 45, "selecting required terms")
    output.status("[Phase 1] Step 3/6: Selecting required terms...")
    required_tasks = [
        _select_required_terms(
            definitions_text,
            o,
            llm,
            policy_id,
            max_retries,
        )
        for o in outlines
        if not o.get("default")
    ]
    required_results = await asyncio.gather(*required_tasks) if required_tasks else []
    required_titles: List[str] = []
    for titles, usages in required_results:
        required_titles.extend(titles)
        all_usages.extend(usages)
    required_titles = dedupe_preserve_order(required_titles)
    output.status(f"[Phase 1] Required terms: {len(required_titles)}")

    # Step 4: Term definitions (parallel per term)
    report("TERM_DEFS", 60, "extracting term definitions")
    output.status("[Phase 1] Step 4/6: Extracting term definitions...")
    term_tasks = [
        _extract_term_definition(
            definitions_text,
            title,
            llm,
            policy_id,
            max_retries,
        )
        for title in required_titles
    ]
    term_results = await asyncio.gather(*term_tasks) if term_tasks else []
    terms: List[Dict[str, Any]] = []
    for term, usages in term_results:
        terms.append(term)
        all_usages.extend(usages)
    output.status(f"[Phase 1] Extracted {len(terms)} terms")

    # Prepare terms payload for clause compilation
    terms_payload = [
        {
            "field": t.get("field"),
            "title": t.get("title"),
            "values": t.get("values", []),
            "descriptions": t.get("descriptions", {}),
        }
        for t in terms
    ]

    # Step 5: Compile clauses (parallel per clause)
    report("COMPILE_CLAUSES", 80, "compiling clauses")
    output.status("[Phase 1] Step 5/6: Compiling clauses...")
    groups: List[Dict[str, Any]] = []
    for outline in outlines:
        if outline.get("default"):
            groups.append({"verdict": outline["verdict"], "default": True})
            continue

        clause_texts = outline.get("clause_texts", [])
        clause_tasks = []
        for idx, clause_text in enumerate(clause_texts):
            step_label = f"{outline.get('verdict')}_clause_{idx + 1}"
            clause_tasks.append(
                _compile_clause(
                    clause_text,
                    terms_payload,
                    llm,
                    policy_id,
                    max_retries,
                    step_label=step_label,
                )
            )

        clause_results = await asyncio.gather(*clause_tasks) if clause_tasks else []
        clauses: List[Dict[str, Any]] = []
        for clause, usages in clause_results:
            clauses.append(clause)
            all_usages.extend(usages)

        groups.append({
            "verdict": outline["verdict"],
            "connective": outline["connective"],
            "clauses": clauses,
        })

    # Step 6: Assemble + validate
    report("ASSEMBLE", 95, "assembling agenda spec")
    output.status("[Phase 1] Step 6/6: Assembling agenda spec...")

    verdict_spec = {
        "verdicts": verdicts,
        "groups": groups,
    }

    agenda_spec = {
        "terms": terms,
        "verdict": verdict_spec,
    }

    # Final validation: ensure exactly one default group and all verdicts covered
    default_groups = [g for g in groups if g.get("default")]
    if len(default_groups) != 1:
        raise ValueError("Expected exactly one default verdict group")

    group_verdicts = [g.get("verdict") for g in groups]
    if set(group_verdicts) != set(verdicts):
        raise ValueError("Verdict groups do not cover all verdict labels")

    usage = accumulate_usage(all_usages)

    # Clear phase1 mode (avoid leaking into other phases)
    if debug_logger := get_debug_logger():
        debug_logger.clear_context()

    output.status(
        f"[Phase 1] Complete: {len(terms)} terms, {len(groups)} groups"
    )

    return agenda_spec, usage


__all__ = [
    "generate_verdict_and_terms",
]
