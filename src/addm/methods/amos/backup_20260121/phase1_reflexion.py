"""Phase 1 Reflexion Approach: Self-Improving Formula Seed Generation.

Initial Generation (Plan-and-Act) → [Quality Analysis → Reflection → Revision]* → Validate

This approach generates an initial seed, then applies domain-agnostic quality
criteria to identify weaknesses and revise iteratively.

Quality Criteria (domain-agnostic):
- Q1: Keyword Specificity (not too generic?)
- Q2: Keyword Coverage (all concepts covered?)
- Q3: Severity Differentiation (clear boundaries?)
- Q4: Scoring-Verdict Alignment (consistent?)
- Q5: Field Completeness (all info captured?)
- Q6: False Positive Risk (likely to match irrelevant?)

No domain-specific hints - all analysis derived from query text.
"""

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from addm.llm import LLMService

from .phase1_plan_and_act import generate_plan_and_act

logger = logging.getLogger(__name__)


# =============================================================================
# Quality Analysis Prompt (Domain-Agnostic)
# =============================================================================

QUALITY_ANALYSIS_PROMPT = '''Analyze this Formula Seed against quality criteria.

## TASK AGENDA

{agenda}

## CURRENT FORMULA SEED

{seed}

## QUALITY CRITERIA

Rate each criterion 1-5 (1=poor, 5=excellent) and explain your rating.

### Q1: Keyword Specificity
Are keywords specific enough to avoid matching irrelevant reviews?
- Bad: Generic words like "food", "service", "bad" that match everything
- Good: Specific phrases like "allergic reaction", "nut allergy", "celiac"

### Q2: Keyword Coverage
Do keywords cover all concepts mentioned in the agenda?
- Check: Every concept from the agenda should have corresponding keywords
- Check: Variants (singular/plural, verb forms) are included

### Q3: Severity Differentiation
Are severity indicators distinct and mutually exclusive?
- Check: Clear distinction between mild/moderate/severe indicators
- Check: No overlap that could cause misclassification

### Q4: Scoring-Verdict Alignment
Does the scoring logic match the agenda's threshold requirements?
- Check: Point values match what the agenda specifies
- Check: Thresholds produce correct verdict labels

### Q5: Field Completeness
Are all necessary extraction fields present?
- Required: ACCOUNT_TYPE, INCIDENT_SEVERITY, SPECIFIC_INCIDENT
- Check: Any modifiers from agenda have corresponding fields

### Q6: False Positive Risk
Could keywords match reviews that aren't actually relevant?
- Check: Keywords are contextual (e.g., "nut allergy" vs just "nut")
- Check: Account for menu descriptions vs actual incidents

## RESPONSE FORMAT

```json
{{
  "scores": {{
    "Q1_specificity": {{"score": <1-5>, "reasoning": "<why>"}},
    "Q2_coverage": {{"score": <1-5>, "reasoning": "<why>"}},
    "Q3_differentiation": {{"score": <1-5>, "reasoning": "<why>"}},
    "Q4_alignment": {{"score": <1-5>, "reasoning": "<why>"}},
    "Q5_completeness": {{"score": <1-5>, "reasoning": "<why>"}},
    "Q6_false_positives": {{"score": <1-5>, "reasoning": "<why>"}}
  }},
  "overall_score": <average>,
  "critical_issues": ["<issues that must be fixed>"],
  "recommendations": ["<specific improvements>"]
}}
```

Output ONLY the JSON:

```json
'''


# =============================================================================
# Reflection Prompt
# =============================================================================

REFLECTION_PROMPT = '''Reflect on the quality analysis and identify specific improvements.

## TASK AGENDA

{agenda}

## CURRENT FORMULA SEED

{seed}

## QUALITY ANALYSIS

{quality_analysis}

## YOUR JOB

Based on the quality analysis, identify SPECIFIC changes to improve the Formula Seed.
Focus on the critical issues and lowest-scoring criteria.

For each change, explain:
1. What's wrong with the current state
2. What the fix should be
3. Why this fix addresses the issue

## RESPONSE FORMAT

```json
{{
  "reflection": {{
    "main_weaknesses": ["<top issues to address>"],
    "root_causes": ["<why these issues exist>"]
  }},

  "planned_changes": [
    {{
      "target": "filter.keywords" | "extract.fields" | "compute" | "search_strategy",
      "change_type": "add" | "remove" | "modify",
      "current_value": "<what exists now>",
      "new_value": "<what it should be>",
      "rationale": "<why this change>"
    }}
  ],

  "expected_improvements": {{
    "Q1_specificity": "<how this improves specificity>",
    "Q2_coverage": "<how this improves coverage>",
    "Q3_differentiation": "<how this improves differentiation>",
    "Q4_alignment": "<how this improves alignment>",
    "Q5_completeness": "<how this improves completeness>",
    "Q6_false_positives": "<how this reduces false positives>"
  }},

  "skip_revision": <true if overall_score >= 4.5 and no critical issues>
}}
```

Output ONLY the JSON:

```json
'''


# =============================================================================
# Revision Prompt
# =============================================================================

REVISION_PROMPT = '''Apply the planned changes to produce an improved Formula Seed.

## TASK AGENDA

{agenda}

## CURRENT FORMULA SEED

{seed}

## PLANNED CHANGES

{planned_changes}

## YOUR JOB

Apply ALL the planned changes to produce a revised Formula Seed.
Ensure the output is valid JSON with the correct structure.

REQUIREMENTS:
- Keep required fields: ACCOUNT_TYPE, INCIDENT_SEVERITY, SPECIFIC_INCIDENT
- Maintain valid compute operations (count, sum, expr, case)
- Ensure field references in compute match extraction field names

Output the COMPLETE revised Formula Seed:

```json
{{
  "task_name": "<policy identifier>",

  "filter": {{
    "keywords": [<revised keywords>]
  }},

  "extract": {{
    "fields": [<revised fields>]
  }},

  "compute": [<revised compute operations>],

  "output": ["VERDICT", "SCORE", "N_INCIDENTS"],

  "search_strategy": {{
    <revised search strategy>
  }},

  "expansion_hints": {{
    "domain": "<from original>",
    "expand_on": [<categories>]
  }}
}}
```

Output ONLY the JSON:

```json
'''


def _extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response."""
    import re

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

    # Clean up common issues
    fixed = re.sub(r',\s*([}\]])', r'\1', response)
    fixed = re.sub(r',\s*,', ',', fixed)
    fixed = re.sub(r'//.*$', '', fixed, flags=re.MULTILINE)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return json.loads(response)


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


async def _analyze_quality(
    agenda: str,
    seed: Dict[str, Any],
    llm: LLMService,
    policy_id: str,
    iteration: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Analyze Formula Seed quality against domain-agnostic criteria.

    Args:
        agenda: The task agenda
        seed: Current Formula Seed
        llm: LLM service
        policy_id: Policy identifier
        iteration: Current reflection iteration

    Returns:
        Tuple of (quality_analysis, usage)
    """
    prompt = QUALITY_ANALYSIS_PROMPT.format(
        agenda=agenda,
        seed=json.dumps(seed, indent=2),
    )

    messages = [{"role": "user", "content": prompt}]

    response, usage = await llm.call_async_with_usage(
        messages,
        context={
            "phase": "phase1_reflexion",
            "step": f"quality_analysis_{iteration}",
            "policy_id": policy_id,
        },
    )

    try:
        analysis = _extract_json_from_response(response)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Quality analysis parse failed: {e}")
        analysis = {
            "scores": {},
            "overall_score": 3.0,
            "critical_issues": ["Parse error"],
            "recommendations": [],
        }

    return analysis, usage


async def _reflect(
    agenda: str,
    seed: Dict[str, Any],
    quality_analysis: Dict[str, Any],
    llm: LLMService,
    policy_id: str,
    iteration: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Reflect on quality analysis and plan improvements.

    Args:
        agenda: The task agenda
        seed: Current Formula Seed
        quality_analysis: Output from quality analysis
        llm: LLM service
        policy_id: Policy identifier
        iteration: Current reflection iteration

    Returns:
        Tuple of (reflection, usage)
    """
    prompt = REFLECTION_PROMPT.format(
        agenda=agenda,
        seed=json.dumps(seed, indent=2),
        quality_analysis=json.dumps(quality_analysis, indent=2),
    )

    messages = [{"role": "user", "content": prompt}]

    response, usage = await llm.call_async_with_usage(
        messages,
        context={
            "phase": "phase1_reflexion",
            "step": f"reflection_{iteration}",
            "policy_id": policy_id,
        },
    )

    try:
        reflection = _extract_json_from_response(response)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Reflection parse failed: {e}")
        reflection = {
            "reflection": {"main_weaknesses": ["Parse error"], "root_causes": []},
            "planned_changes": [],
            "expected_improvements": {},
            "skip_revision": True,
        }

    return reflection, usage


async def _revise(
    agenda: str,
    seed: Dict[str, Any],
    planned_changes: List[Dict[str, Any]],
    llm: LLMService,
    policy_id: str,
    iteration: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply planned changes to produce revised Formula Seed.

    Args:
        agenda: The task agenda
        seed: Current Formula Seed
        planned_changes: List of changes to apply
        llm: LLM service
        policy_id: Policy identifier
        iteration: Current reflection iteration

    Returns:
        Tuple of (revised_seed, usage)
    """
    prompt = REVISION_PROMPT.format(
        agenda=agenda,
        seed=json.dumps(seed, indent=2),
        planned_changes=json.dumps(planned_changes, indent=2),
    )

    messages = [{"role": "user", "content": prompt}]

    response, usage = await llm.call_async_with_usage(
        messages,
        context={
            "phase": "phase1_reflexion",
            "step": f"revision_{iteration}",
            "policy_id": policy_id,
        },
    )

    try:
        revised_seed = _extract_json_from_response(response)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Revision parse failed: {e}, keeping original")
        revised_seed = deepcopy(seed)

    return revised_seed, usage


def _ensure_required_structure(seed: Dict[str, Any], policy_id: str) -> Dict[str, Any]:
    """Ensure seed has all required structure elements."""
    seed = deepcopy(seed)

    # Ensure task_name
    if "task_name" not in seed:
        seed["task_name"] = policy_id

    # Ensure filter
    if "filter" not in seed:
        seed["filter"] = {"keywords": []}
    if "keywords" not in seed["filter"]:
        seed["filter"]["keywords"] = []

    # Ensure extract
    if "extract" not in seed:
        seed["extract"] = {"fields": []}
    if "fields" not in seed["extract"]:
        seed["extract"]["fields"] = []

    # Ensure required extraction fields
    field_names = {f.get("name") for f in seed["extract"]["fields"]}

    required_fields = [
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

    for field in required_fields:
        if field["name"] not in field_names:
            seed["extract"]["fields"].append(field)

    # Ensure compute
    if "compute" not in seed:
        seed["compute"] = []

    # Ensure output
    if "output" not in seed:
        seed["output"] = ["VERDICT", "SCORE", "N_INCIDENTS"]

    # Ensure search_strategy
    if "search_strategy" not in seed:
        seed["search_strategy"] = {}

    return seed


async def generate_reflexion(
    agenda: str,
    policy_id: str,
    llm: LLMService,
    max_iterations: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Generate Formula Seed using Reflexion approach.

    Initial Generation (Plan-and-Act) → [Quality Analysis → Reflection → Revision]*

    Args:
        agenda: The task agenda/query prompt
        policy_id: Policy identifier (e.g., "G1_allergy_V2")
        llm: LLM service for API calls
        max_iterations: Maximum reflection iterations (default 2)

    Returns:
        Tuple of (formula_seed, intermediates, usage)
        - formula_seed: The generated Formula Seed specification
        - intermediates: Dict with reflection history for debugging
        - usage: Combined token/cost usage from all LLM calls
    """
    all_usages = []
    reflection_history = []

    # Step 1: Initial generation using Plan-and-Act
    logger.info(f"Reflexion: Initial generation for {policy_id}")
    seed, plan_act_intermediates, initial_usage = await generate_plan_and_act(
        agenda, policy_id, llm
    )
    all_usages.append(initial_usage)

    # Ensure structure
    seed = _ensure_required_structure(seed, policy_id)

    # Reflection loop
    for iteration in range(1, max_iterations + 1):
        logger.info(f"Reflexion: Iteration {iteration}/{max_iterations} for {policy_id}")

        # Quality Analysis
        quality_analysis, qa_usage = await _analyze_quality(
            agenda, seed, llm, policy_id, iteration
        )
        all_usages.append(qa_usage)

        overall_score = quality_analysis.get("overall_score", 0)
        critical_issues = quality_analysis.get("critical_issues", [])
        logger.debug(f"Reflexion: Overall score {overall_score}, critical issues: {len(critical_issues)}")

        # Check if good enough
        if overall_score >= 4.5 and not critical_issues:
            logger.info(f"Reflexion: Quality threshold met at iteration {iteration}")
            reflection_history.append({
                "iteration": iteration,
                "quality_analysis": quality_analysis,
                "action": "accepted",
            })
            break

        # Reflection
        reflection, refl_usage = await _reflect(
            agenda, seed, quality_analysis, llm, policy_id, iteration
        )
        all_usages.append(refl_usage)

        # Check if should skip revision
        if reflection.get("skip_revision", False):
            logger.info(f"Reflexion: Skipping revision at iteration {iteration}")
            reflection_history.append({
                "iteration": iteration,
                "quality_analysis": quality_analysis,
                "reflection": reflection,
                "action": "skipped",
            })
            break

        # Revision
        planned_changes = reflection.get("planned_changes", [])
        if planned_changes:
            revised_seed, rev_usage = await _revise(
                agenda, seed, planned_changes, llm, policy_id, iteration
            )
            all_usages.append(rev_usage)

            # Ensure structure after revision
            revised_seed = _ensure_required_structure(revised_seed, policy_id)
            seed = revised_seed

            reflection_history.append({
                "iteration": iteration,
                "quality_analysis": quality_analysis,
                "reflection": reflection,
                "action": "revised",
                "changes_applied": len(planned_changes),
            })
        else:
            logger.debug(f"Reflexion: No changes planned at iteration {iteration}")
            reflection_history.append({
                "iteration": iteration,
                "quality_analysis": quality_analysis,
                "reflection": reflection,
                "action": "no_changes",
            })

    # Final quality check
    final_analysis, final_usage = await _analyze_quality(
        agenda, seed, llm, policy_id, max_iterations + 1
    )
    all_usages.append(final_usage)

    intermediates = {
        "plan_and_act": plan_act_intermediates,
        "reflection_history": reflection_history,
        "final_quality": final_analysis,
        "total_iterations": len(reflection_history),
    }

    total_usage = _accumulate_usage(all_usages)

    return seed, intermediates, total_usage
