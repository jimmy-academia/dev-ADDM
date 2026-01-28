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
    LOCATE_BLOCKS_PROMPT,
    EXTRACT_VERDICT_RULES_PROMPT,
    SELECT_TERM_BLOCKS_PROMPT,
    EXTRACT_TERM_ENUM_PROMPT,
    COMPILE_CLAUSE_PROMPT,
)
from .phase1_helpers import (
    extract_json_from_response,
    parse_json_safely,
    slice_block_by_anchors,
    normalize_block_anchors,
    validate_block_anchors,
    normalize_verdict_rules,
    validate_verdict_rules,
    normalize_term_blocks,
    validate_term_blocks,
    normalize_term_enum,
    validate_term_enum,
    normalize_clause_spec,
    validate_clause_spec,
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

        # Record each validation failure attempt.
        if debug_logger := get_debug_logger():
            sample_id = (
                context.get("policy_id")
                or context.get("sample_id")
                or "unknown"
            )
            debug_logger.log_event(
                sample_id=sample_id,
                event_type="phase1.validation_error",
                data={
                    "phase": context.get("phase"),
                    "step": context.get("step"),
                    "attempt": attempt,
                    "errors": errors,
                },
            )

        # Give up after the final attempt; proceed best-effort and warn.
        if attempt >= max_retries:
            step = context.get("step") or context.get("phase") or "unknown_step"
            sample_id = (
                context.get("policy_id")
                or context.get("sample_id")
                or "unknown"
            )
            if debug_logger := get_debug_logger():
                debug_logger.log_event(
                    sample_id=sample_id,
                    event_type="phase1.validation_giveup",
                    data={
                        "phase": context.get("phase"),
                        "step": step,
                        "attempts": attempt + 1,
                        "max_retries": max_retries,
                        "errors": errors,
                    },
                )
            # NOTE: output.warn is suppressed under suppress_output; output.print is not.
            output.print(
                f"  [yellow]⚠[/yellow] {sample_id} P1 {step}: "
                f"validation failed after {max_retries} retries; continuing"
            )
            return data, usages

        attempt += 1
        prompt = _with_errors(base_prompt, errors, data)


# =============================================================================
# Step 0: Locate blocks (anchors only)
# =============================================================================

async def _locate_blocks(
    agenda: str,
    llm: LLMService,
    policy_id: str,
    max_retries: int,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    prompt = LOCATE_BLOCKS_PROMPT.format(agenda=agenda)

    def validator(data: Dict[str, Any]) -> List[str]:
        definitions, verdicts = normalize_block_anchors(data)
        return validate_block_anchors(definitions, verdicts, agenda)

    data, usages = await _call_llm_json_with_retries(
        prompt,
        llm,
        context={
            "sample_id": policy_id,
            "phase": "phase1_locate_blocks",
            "step": "locate_blocks",
            "policy_id": policy_id,
        },
        validator=validator,
        max_retries=max_retries,
    )

    definitions, verdicts = normalize_block_anchors(data)
    definition_blocks: List[str] = []
    for idx, b in enumerate(definitions):
        try:
            definition_blocks.append(
                slice_block_by_anchors(agenda, b["start_quote"], b["end_quote"])
            )
        except Exception as e:  # noqa: BLE001 - best-effort fallback
            if debug_logger := get_debug_logger():
                debug_logger.log_event(
                    sample_id=policy_id,
                    event_type="phase1.slice_warning",
                    data={
                        "step": "locate_blocks",
                        "group": "definitions_blocks",
                        "index": idx,
                        "error": str(e),
                    },
                )
            output.print(
                f"  [yellow]⚠[/yellow] {policy_id} P1 locate_blocks: "
                f"failed to slice definitions_blocks[{idx}]; using full agenda"
            )

    verdict_blocks: List[str] = []
    for idx, b in enumerate(verdicts):
        try:
            verdict_blocks.append(
                slice_block_by_anchors(agenda, b["start_quote"], b["end_quote"])
            )
        except Exception as e:  # noqa: BLE001 - best-effort fallback
            if debug_logger := get_debug_logger():
                debug_logger.log_event(
                    sample_id=policy_id,
                    event_type="phase1.slice_warning",
                    data={
                        "step": "locate_blocks",
                        "group": "verdict_blocks",
                        "index": idx,
                        "error": str(e),
                    },
                )
            output.print(
                f"  [yellow]⚠[/yellow] {policy_id} P1 locate_blocks: "
                f"failed to slice verdict_blocks[{idx}]; using full agenda"
            )

    # Fallback: if slicing failed completely, pass the full agenda forward.
    if not definition_blocks:
        definition_blocks = [agenda]
    if not verdict_blocks:
        verdict_blocks = [agenda]
    return definition_blocks, verdict_blocks, usages


# =============================================================================
# Step 1: Extract verdict rules skeleton
# =============================================================================

async def _extract_verdict_rules(
    verdict_text: str,
    agenda: str,
    llm: LLMService,
    policy_id: str,
    max_retries: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prompt = EXTRACT_VERDICT_RULES_PROMPT.format(verdict_text=verdict_text)

    def validator(data: Dict[str, Any]) -> List[str]:
        verdict_rules = normalize_verdict_rules(data)
        return validate_verdict_rules(verdict_rules, agenda)

    data, usages = await _call_llm_json_with_retries(
        prompt,
        llm,
        context={
            "sample_id": policy_id,
            "phase": "phase1_verdict_rules",
            "step": "extract_verdict_rules",
            "policy_id": policy_id,
        },
        validator=validator,
        max_retries=max_retries,
    )

    verdict_rules = normalize_verdict_rules(data)
    return verdict_rules, usages


# =============================================================================
# Step 2: Select term blocks (anchors only)
# =============================================================================

async def _select_term_blocks(
    definitions_text: str,
    verdict_rules: Dict[str, Any],
    agenda: str,
    llm: LLMService,
    policy_id: str,
    max_retries: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    verdict_rules_json = json.dumps(verdict_rules, indent=2, ensure_ascii=True)
    prompt = SELECT_TERM_BLOCKS_PROMPT.format(
        definitions_text=definitions_text,
        verdict_rules_json=verdict_rules_json,
    )

    non_default_rules = [
        r for r in verdict_rules.get("rules", []) if not r.get("default")
    ]
    allow_empty = not non_default_rules

    def validator(data: Dict[str, Any]) -> List[str]:
        terms = normalize_term_blocks(data)
        return validate_term_blocks(terms, agenda, definitions_text, allow_empty)

    data, usages = await _call_llm_json_with_retries(
        prompt,
        llm,
        context={
            "sample_id": policy_id,
            "phase": "phase1_term_blocks",
            "step": "select_term_blocks",
            "policy_id": policy_id,
        },
        validator=validator,
        max_retries=max_retries,
    )

    terms = normalize_term_blocks(data)
    return terms, usages


# =============================================================================
# Step 3: Extract term enum (per term)
# =============================================================================

async def _extract_term_enum(
    term_title: str,
    term_block: str,
    llm: LLMService,
    policy_id: str,
    max_retries: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prompt = EXTRACT_TERM_ENUM_PROMPT.format(
        term_title=term_title,
        term_block=term_block,
    )

    def validator(data: Dict[str, Any]) -> List[str]:
        term = normalize_term_enum(data, term_title)
        return validate_term_enum(term, term_title, term_block)

    data, usages = await _call_llm_json_with_retries(
        prompt,
        llm,
        context={
            "sample_id": policy_id,
            "phase": "phase1_term_enum",
            "step": f"term_{term_title}",
            "policy_id": policy_id,
        },
        validator=validator,
        max_retries=max_retries,
    )

    term = normalize_term_enum(data, term_title)
    return term, usages


# =============================================================================
# Step 4: Compile clause (per clause)
# =============================================================================

async def _compile_clause(
    clause_quote: str,
    allowed_terms: List[Dict[str, Any]],
    llm: LLMService,
    policy_id: str,
    max_retries: int,
    step_label: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    terms_json = json.dumps(allowed_terms, indent=2, ensure_ascii=True)
    prompt = COMPILE_CLAUSE_PROMPT.format(
        clause_quote=clause_quote,
        terms_json=terms_json,
    )

    allowed_map = {
        t["field_id"]: [v["value_id"] for v in t.get("values", [])]
        for t in allowed_terms
    }

    def validator(data: Dict[str, Any]) -> List[str]:
        clause = normalize_clause_spec(data)
        return validate_clause_spec(clause, allowed_map)

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

    def log_event(event_type: str, data: Dict[str, Any]) -> None:
        if debug_logger := get_debug_logger():
            debug_logger.log_event(policy_id, event_type, data)

    # Step 0: Locate blocks
    report("LOCATE_BLOCKS", 5, "locating blocks")
    output.status("[Phase 1] Step 0/6: Locating blocks...")
    definitions_blocks, verdict_blocks, usages0 = await _locate_blocks(
        agenda,
        llm,
        policy_id,
        max_retries,
    )
    all_usages.extend(usages0)

    definitions_text = "\n\n".join(definitions_blocks)
    verdict_text = "\n\n".join(verdict_blocks)
    log_event(
        "phase1.locate_blocks.summary",
        {
            "definitions_blocks": len(definitions_blocks),
            "verdict_blocks": len(verdict_blocks),
            "definitions_lengths": [len(b) for b in definitions_blocks],
            "verdict_lengths": [len(b) for b in verdict_blocks],
        },
    )

    # Step 1: Verdict rules skeleton
    report("VERDICT_RULES", 15, "extracting verdict rules")
    output.status("[Phase 1] Step 1/6: Extracting verdict rules...")
    verdict_rules, usages1 = await _extract_verdict_rules(
        verdict_text,
        agenda,
        llm,
        policy_id,
        max_retries,
    )
    all_usages.extend(usages1)
    output.status(
        f"[Phase 1] Labels: {len(verdict_rules.get('labels', []))} "
        f"(default={verdict_rules.get('default_label')})"
    )
    log_event(
        "phase1.verdict_rules.summary",
        {
            "labels": verdict_rules.get("labels", []),
            "default_label": verdict_rules.get("default_label"),
            "order": verdict_rules.get("order", []),
            "rules": [
                {
                    "label": r.get("label"),
                    "default": r.get("default"),
                    "connective": r.get("connective"),
                    "clauses": len(r.get("clause_quotes", [])),
                }
                for r in verdict_rules.get("rules", [])
            ],
        },
    )

    # Step 2: Select term blocks
    report("TERM_BLOCKS", 35, "selecting term blocks")
    output.status("[Phase 1] Step 2/6: Selecting term blocks...")
    non_default_rules = [
        r for r in verdict_rules.get("rules", []) if not r.get("default")
    ]
    if non_default_rules:
        term_blocks, usages2 = await _select_term_blocks(
            definitions_text,
            verdict_rules,
            agenda,
            llm,
            policy_id,
            max_retries,
        )
        all_usages.extend(usages2)
    else:
        term_blocks = []
    output.status(f"[Phase 1] Term blocks: {len(term_blocks)}")
    log_event(
        "phase1.term_blocks.summary",
        {
            "terms": [
                {
                    "term_title": t.get("term_title"),
                    "start_len": len(t.get("start_quote", "") or ""),
                    "end_len": len(t.get("end_quote", "") or ""),
                }
                for t in term_blocks
            ],
        },
    )

    # Slice term blocks
    term_inputs: List[Tuple[str, str]] = []
    for t in term_blocks:
        term_title = t.get("term_title", "")
        try:
            term_block = slice_block_by_anchors(
                agenda,
                t.get("start_quote", ""),
                t.get("end_quote", ""),
            )
        except Exception as e:  # noqa: BLE001 - best-effort fallback
            if debug_logger := get_debug_logger():
                debug_logger.log_event(
                    sample_id=policy_id,
                    event_type="phase1.slice_warning",
                    data={
                        "step": "term_blocks",
                        "term_title": term_title,
                        "error": str(e),
                    },
                )
            output.print(
                f"  [yellow]⚠[/yellow] {policy_id} P1 term_blocks[{term_title}]: "
                "failed to slice; using definitions_text"
            )
            term_block = definitions_text
        term_inputs.append((term_title, term_block))

    if term_inputs:
        log_event(
            "phase1.term_blocks.sliced",
            {
                "terms": [
                    {"term_title": title, "block_length": len(block)}
                    for title, block in term_inputs
                ],
            },
        )

    # Step 3: Extract term enums (parallel per term)
    report("TERM_ENUMS", 55, "extracting term enums")
    output.status("[Phase 1] Step 3/6: Extracting term enums...")
    term_tasks = [
        _extract_term_enum(
            title,
            block,
            llm,
            policy_id,
            max_retries,
        )
        for title, block in term_inputs
    ]
    term_results = await asyncio.gather(*term_tasks) if term_tasks else []
    terms: List[Dict[str, Any]] = []
    for term, usages in term_results:
        terms.append(term)
        all_usages.extend(usages)
    output.status(f"[Phase 1] Extracted {len(terms)} terms")
    log_event(
        "phase1.term_enums.summary",
        {
            "terms": [
                {
                    "term_title": t.get("term_title"),
                    "field_id": t.get("field_id"),
                    "value_count": len(t.get("values", [])),
                }
                for t in terms
            ],
        },
    )

    # Prepare allowed terms payload for clause compilation
    allowed_terms = [
        {
            "field_id": t.get("field_id"),
            "term_title": t.get("term_title"),
            "values": [
                {
                    "value_id": v.get("value_id"),
                    "source_value": v.get("source_value"),
                }
                for v in t.get("values", [])
            ],
        }
        for t in terms
    ]

    # Step 4: Compile clauses (parallel per clause)
    report("COMPILE_CLAUSES", 75, "compiling clauses")
    output.status("[Phase 1] Step 4/6: Compiling clauses...")
    rules_out: List[Dict[str, Any]] = []
    for rule in verdict_rules.get("rules", []):
        if rule.get("default"):
            rules_out.append({
                "label": rule.get("label"),
                "default": True,
                "default_quote": rule.get("default_quote"),
                "hints": rule.get("hints", []),
            })
            continue

        clause_quotes = rule.get("clause_quotes", [])
        clause_tasks = []
        for idx, clause_quote in enumerate(clause_quotes):
            step_label = f"{rule.get('label')}_clause_{idx + 1}"
            clause_tasks.append(
                _compile_clause(
                    clause_quote,
                    allowed_terms,
                    llm,
                    policy_id,
                    max_retries,
                    step_label=step_label,
                )
            )

        clause_results = await asyncio.gather(*clause_tasks) if clause_tasks else []
        clauses: List[Dict[str, Any]] = []
        for idx, (clause, usages) in enumerate(clause_results):
            clauses.append({
                "clause_quote": clause_quotes[idx],
                "min_count": clause.get("min_count"),
                "logic": clause.get("logic"),
                "conditions": clause.get("conditions"),
            })
            all_usages.extend(usages)

        rules_out.append({
            "label": rule.get("label"),
            "default": False,
            "connective": rule.get("connective"),
            "connective_quote": rule.get("connective_quote"),
            "clauses": clauses,
        })
        log_event(
            "phase1.compile_clause.summary",
            {
                "label": rule.get("label"),
                "connective": rule.get("connective"),
                "clauses": [
                    {
                        "min_count": c.get("min_count"),
                        "logic": c.get("logic"),
                        "fields": [cond.get("field_id") for cond in c.get("conditions", [])],
                    }
                    for c in clauses
                ],
            },
        )

    # Step 5: Assemble + prune + validate
    report("ASSEMBLE", 95, "assembling agenda spec")
    output.status("[Phase 1] Step 5/6: Assembling agenda spec...")

    # Prune unused terms (only those referenced by non-default clauses)
    used_fields = set()
    for rule in rules_out:
        if rule.get("default"):
            continue
        for clause in rule.get("clauses", []):
            for cond in clause.get("conditions", []):
                if cond.get("field_id"):
                    used_fields.add(cond["field_id"])

    pruned_terms = [
        t for t in terms if t.get("field_id") in used_fields
    ]

    agenda_spec = {
        "terms": pruned_terms,
        "verdict_rules": {
            "labels": verdict_rules.get("labels", []),
            "default_label": verdict_rules.get("default_label", ""),
            "order": verdict_rules.get("order", []),
            "rules": rules_out,
        },
    }
    log_event(
        "phase1.assemble.summary",
        {
            "terms": len(pruned_terms),
            "rules": len(rules_out),
            "used_fields": sorted(used_fields),
        },
    )

    final_warnings: List[str] = []

    # Final validation (warn-only): ensure exactly one default rule and all labels covered
    default_rules = [r for r in rules_out if r.get("default")]
    if len(default_rules) != 1:
        final_warnings.append("Expected exactly one default rule")
    rule_labels = [r.get("label") for r in rules_out if r.get("label")]
    if set(rule_labels) != set(verdict_rules.get("labels", [])):
        final_warnings.append("Rules do not cover all verdict labels")
    if (
        verdict_rules.get("order")
        and verdict_rules.get("default_label")
        and verdict_rules["order"][-1] != verdict_rules.get("default_label")
    ):
        final_warnings.append("order must end with default_label")

    # Ensure all conditions reference allowed terms/values after pruning
    term_values_map = {
        t["field_id"]: [v["value_id"] for v in t.get("values", [])]
        for t in pruned_terms
    }
    for rule in rules_out:
        if rule.get("default"):
            continue
        if rule.get("connective") not in ("ANY", "ALL"):
            final_warnings.append("Rule connective must be ANY or ALL")
        for clause in rule.get("clauses", []):
            if clause.get("logic") not in ("ANY", "ALL"):
                final_warnings.append("Clause logic must be ANY or ALL")
            for cond in clause.get("conditions", []):
                field_id = cond.get("field_id")
                if field_id not in term_values_map:
                    final_warnings.append(
                        f"Condition field_id '{field_id}' not in terms"
                    )
                    continue
                for v in cond.get("values", []):
                    if v not in term_values_map[field_id]:
                        final_warnings.append(
                            f"Condition value '{v}' not in {field_id} values"
                        )

    if final_warnings:
        log_event(
            "phase1.validation_final_warning",
            {"warnings": final_warnings},
        )
        output.print(
            f"  [yellow]⚠[/yellow] {policy_id} P1 final validation: "
            f"{len(final_warnings)} warning(s) (see debug/phase1.jsonl)"
        )

    usage = accumulate_usage(all_usages)

    # Clear phase1 mode (avoid leaking into other phases)
    if debug_logger := get_debug_logger():
        debug_logger.clear_context()

    output.status(
        f"[Phase 1] Complete: {len(pruned_terms)} terms, {len(rules_out)} rules"
    )

    return agenda_spec, usage


__all__ = [
    "generate_verdict_and_terms",
]
