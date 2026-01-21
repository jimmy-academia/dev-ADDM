"""Phase 1 Plan-and-Act Approach: Dynamic Formula Seed Generation.

OBSERVE → PLAN → ACT pipeline with NO domain-specific hints.
Keywords are discovered from the agenda text, not pattern-matched.

This is a fixed 3-step pipeline (same cost as original approach):
1. OBSERVE: Analyze query to extract concepts, signals, requirements
2. PLAN: Create explicit reasoning about what keywords/fields/rules are needed
3. ACT: Generate Formula Seed following the plan
"""

import json
import logging
from typing import Any, Dict, List, Tuple

from addm.llm import LLMService

logger = logging.getLogger(__name__)


# =============================================================================
# Step 1: OBSERVE - Query Analysis (NO domain hints)
# =============================================================================

OBSERVE_PROMPT = '''Analyze this task agenda and extract EXACTLY what it specifies.

## TASK AGENDA

{agenda}

## YOUR JOB

Read the agenda carefully. Extract the components it EXPLICITLY mentions.
Do NOT add assumptions or domain knowledge - only extract what the text says.

Identify:
1. **Core Concepts**: What specific things does the agenda ask about? (Extract exact terms used)
2. **Evaluation Criteria**: How does the agenda define what counts/doesn't count?
3. **Severity Levels**: What severity classifications does the agenda specify?
4. **Account Types**: Does it distinguish firsthand vs secondhand accounts?
5. **Modifiers**: Any additional factors that affect scoring?
6. **Thresholds**: What numerical thresholds determine verdicts?
7. **Verdict Labels**: Exact text for each verdict level?

Output a JSON object with your observations:

```json
{{
  "core_concepts": {{
    "primary_topic": "<the main subject of evaluation, in the agenda's own words>",
    "explicit_terms": ["<terms the agenda explicitly uses>"],
    "related_concepts": ["<concepts the agenda explicitly relates to the topic>"]
  }},

  "evaluation_criteria": {{
    "what_counts": "<exact criteria from agenda>",
    "what_does_not_count": "<exclusions from agenda, if any>",
    "special_cases": ["<any special handling mentioned>"]
  }},

  "severity_framework": {{
    "levels": [
      {{"level": "none", "description": "<from agenda>", "points": 0}},
      {{"level": "mild", "description": "<from agenda>", "points": "<from agenda>"}},
      {{"level": "moderate", "description": "<from agenda>", "points": "<from agenda>"}},
      {{"level": "severe", "description": "<from agenda>", "points": "<from agenda>"}}
    ],
    "scoring_notes": "<any scoring details from agenda>"
  }},

  "account_handling": {{
    "types": [
      {{"type": "firsthand", "weight": 1.0, "description": "<from agenda>"}},
      {{"type": "secondhand", "weight": "<from agenda>", "description": "<from agenda>"}},
      {{"type": "general", "weight": "<from agenda>", "description": "<from agenda>"}}
    ]
  }},

  "modifiers": [
    {{"name": "<modifier from agenda>", "condition": "<when it applies>", "effect": "<point adjustment>"}}
  ],

  "verdict_thresholds": {{
    "critical": {{"min_score": "<from agenda>", "label": "<exact label from agenda>"}},
    "high": {{"min_score": "<from agenda>", "label": "<exact label from agenda>"}},
    "low": {{"min_score": "<from agenda or 0>", "label": "<exact label from agenda>"}}
  }},

  "temporal_requirements": {{
    "has_recency_weighting": "<true/false based on agenda>",
    "decay_rules": "<if mentioned in agenda>"
  }}
}}
```

IMPORTANT: Only include what the agenda EXPLICITLY states. Use null or empty values for anything not specified.

Output ONLY the JSON:

```json
'''


# =============================================================================
# Step 2: PLAN - Keyword and Field Strategy (NO domain hints)
# =============================================================================

PLAN_PROMPT = '''Create a detailed plan for generating a Formula Seed based on these observations.

## OBSERVATIONS FROM QUERY ANALYSIS

{observations}

## YOUR JOB

Think carefully about what keywords, fields, and rules will be needed.
Base your reasoning ONLY on what was observed in the query - do NOT add external domain knowledge.

For keywords, reason about:
- What words would reviewers use when discussing the concepts from the observations?
- What terms indicate each severity level according to the agenda's definitions?
- What phrases distinguish firsthand from secondhand accounts?

Output your plan:

```json
{{
  "keyword_strategy": {{
    "reasoning": "<explain how you derived keywords from the observed concepts>",

    "from_core_concepts": {{
      "primary_terms": ["<words derived from primary_topic>"],
      "related_terms": ["<words derived from related_concepts>"],
      "derivation_notes": "<how these connect to the agenda>"
    }},

    "from_severity_definitions": {{
      "severe_indicators": ["<words that match 'severe' definition from agenda>"],
      "moderate_indicators": ["<words that match 'moderate' definition>"],
      "mild_indicators": ["<words that match 'mild' definition>"],
      "derivation_notes": "<how these connect to severity descriptions>"
    }},

    "from_evaluation_criteria": {{
      "positive_indicators": ["<words indicating something counts>"],
      "negative_indicators": ["<words indicating something doesn't count>"],
      "derivation_notes": "<how these connect to criteria>"
    }},

    "account_type_indicators": {{
      "firsthand_phrases": ["<phrases indicating personal experience>"],
      "secondhand_phrases": ["<phrases indicating heard from others>"],
      "general_phrases": ["<phrases indicating general statements>"]
    }},

    "priority_terms": ["<highest-signal keywords for early stopping>"]
  }},

  "field_strategy": {{
    "reasoning": "<explain what extraction fields are needed>",

    "required_fields": [
      {{
        "name": "ACCOUNT_TYPE",
        "purpose": "Distinguish firsthand/secondhand/general",
        "based_on": "<which observation this addresses>"
      }},
      {{
        "name": "INCIDENT_SEVERITY",
        "purpose": "Classify severity level",
        "based_on": "<which observation this addresses>"
      }},
      {{
        "name": "SPECIFIC_INCIDENT",
        "purpose": "Capture what happened",
        "based_on": "<which observation this addresses>"
      }}
    ],

    "additional_fields": [
      {{
        "name": "<field_name>",
        "purpose": "<why needed>",
        "based_on": "<which modifier/criteria requires this>"
      }}
    ]
  }},

  "compute_strategy": {{
    "reasoning": "<explain the scoring approach based on observations>",

    "scoring_approach": {{
      "base_scoring": "<how to compute base points per observation>",
      "modifier_scoring": "<how to apply modifiers if any>",
      "aggregation": "<how to combine scores>"
    }},

    "verdict_rules": {{
      "threshold_logic": "<exact threshold rules from observations>",
      "edge_cases": "<any special handling needed>"
    }}
  }},

  "search_strategy": {{
    "early_stop_score": "<score at which verdict is definite>",
    "priority_logic": "<how to prioritize reviews>"
  }}
}}
```

REMEMBER: Derive everything from the observations. Do NOT inject external domain knowledge.

Output ONLY the JSON:

```json
'''


# =============================================================================
# Step 3: ACT - Generate Formula Seed (Following the Plan)
# =============================================================================

ACT_PROMPT = '''Generate the complete Formula Seed following this plan.

## ORIGINAL OBSERVATIONS

{observations}

## GENERATION PLAN

{plan}

## REQUIREMENTS

You MUST use these EXACT field names in extraction schema:
- ACCOUNT_TYPE: "firsthand" | "secondhand" | "general"
- INCIDENT_SEVERITY: "none" | "mild" | "moderate" | "severe"
- SPECIFIC_INCIDENT: string describing what happened (or null if none)

Additional fields may be added based on the plan's field_strategy.

Use this compute structure (adjust values based on plan):
1. Count incidents: {{"name": "N_INCIDENTS", "op": "count", "where": {...}}}
2. Sum base points: {{"name": "BASE_POINTS", "op": "sum", "expr": "CASE WHEN...", "where": {...}}}
3. Sum modifier points (if needed): {{"name": "MODIFIER_POINTS", "op": "sum", "expr": "...", "where": {...}}}
4. Total score: {{"name": "SCORE", "op": "expr", "expr": "BASE_POINTS + MODIFIER_POINTS"}}
5. Verdict: {{"name": "VERDICT", "op": "case", "source": "SCORE", "rules": [...]}}

## YOUR JOB

Generate the Formula Seed using:
- Keywords from keyword_strategy (flatten all categories)
- Fields from field_strategy
- Scoring from compute_strategy
- Thresholds from observations

Output the complete Formula Seed:

```json
{{
  "task_name": "<policy identifier>",

  "filter": {{
    "keywords": ["<all keywords from plan, flattened>"]
  }},

  "extract": {{
    "fields": [
      {{
        "name": "ACCOUNT_TYPE",
        "type": "enum",
        "values": {{
          "firsthand": "Reviewer personally experienced it",
          "secondhand": "Reviewer heard from others",
          "general": "General statement without specific incident"
        }}
      }},
      {{
        "name": "INCIDENT_SEVERITY",
        "type": "enum",
        "values": {{
          "none": "No relevant incident",
          "mild": "<from severity_framework>",
          "moderate": "<from severity_framework>",
          "severe": "<from severity_framework>"
        }}
      }},
      {{
        "name": "SPECIFIC_INCIDENT",
        "type": "string",
        "description": "Brief description of what happened"
      }}
    ]
  }},

  "compute": [
    <compute operations per plan>
  ],

  "output": ["VERDICT", "SCORE", "N_INCIDENTS"],

  "search_strategy": {{
    "priority_keywords": ["<from plan.keyword_strategy.priority_terms>"],
    "priority_expr": "<from plan.search_strategy>",
    "stopping_condition": "<from plan.search_strategy>",
    "early_verdict_expr": "<build from verdict_thresholds>",
    "use_embeddings_when": "len(keyword_matched) < 5"
  }},

  "expansion_hints": {{
    "domain": "<from observations.core_concepts.primary_topic>",
    "expand_on": ["<categories to expand in Phase 2>"]
  }}
}}
```

Follow the plan exactly. Output ONLY the JSON:

```json
'''


def _extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
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

    # Try to parse as-is first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Common fixes
    fixed = re.sub(r',\s*([}\]])', r'\1', response)
    fixed = re.sub(r',\s*,', ',', fixed)
    fixed = re.sub(r'//.*$', '', fixed, flags=re.MULTILINE)

    return json.loads(fixed)


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
    prompt = OBSERVE_PROMPT.format(agenda=agenda)
    messages = [{"role": "user", "content": prompt}]

    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_plan_act", "step": "observe", "policy_id": policy_id},
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
    prompt = PLAN_PROMPT.format(observations=json.dumps(observations, indent=2))
    messages = [{"role": "user", "content": prompt}]

    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_plan_act", "step": "plan", "policy_id": policy_id},
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
    prompt = ACT_PROMPT.format(
        observations=json.dumps(observations, indent=2),
        plan=json.dumps(plan, indent=2),
    )
    messages = [{"role": "user", "content": prompt}]

    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_plan_act", "step": "act", "policy_id": policy_id},
    )

    seed = _extract_json_from_response(response)
    return seed, usage


async def generate_plan_and_act(
    agenda: str,
    policy_id: str,
    llm: LLMService,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Generate Formula Seed using Plan-and-Act approach.

    Fixed 3-step pipeline: OBSERVE → PLAN → ACT
    No domain-specific hints - keywords derived from query analysis.

    Args:
        agenda: The task agenda/query prompt
        policy_id: Policy identifier (e.g., "G1_allergy_V2")
        llm: LLM service for API calls

    Returns:
        Tuple of (formula_seed, intermediates, usage)
        - formula_seed: The generated Formula Seed specification
        - intermediates: Dict with observe/plan step outputs for debugging
        - usage: Combined token/cost usage from all LLM calls
    """
    all_usages = []

    # Step 1: OBSERVE
    logger.info(f"Plan-and-Act Step 1: OBSERVE for {policy_id}")
    observations, usage1 = await _observe(agenda, llm, policy_id)
    all_usages.append(usage1)
    logger.debug(f"Observed primary topic: {observations.get('core_concepts', {}).get('primary_topic', 'unknown')}")

    # Step 2: PLAN
    logger.info(f"Plan-and-Act Step 2: PLAN for {policy_id}")
    plan, usage2 = await _plan(observations, llm, policy_id)
    all_usages.append(usage2)

    # Step 3: ACT
    logger.info(f"Plan-and-Act Step 3: ACT for {policy_id}")
    seed, usage3 = await _act(observations, plan, llm, policy_id)
    all_usages.append(usage3)

    # Store intermediates
    intermediates = {
        "observations": observations,
        "plan": plan,
    }

    total_usage = _accumulate_usage(all_usages)

    return seed, intermediates, total_usage
