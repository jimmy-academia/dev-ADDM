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

Read the agenda carefully. There are TWO types of verdict logic - identify which one this agenda uses:

### TYPE A: SCORING-BASED (uses point values)
The agenda specifies point values for categories (e.g., "Severe: 15 points, Moderate: 5 points").
Verdict is determined by total score vs thresholds (e.g., "Critical if score >= 8").

### TYPE B: RULE-BASED (uses count thresholds)
The agenda specifies count thresholds for verdict rules (e.g., "Critical if 2+ severe incidents").
NO point values are assigned to categories. Verdict is determined by counting occurrences.

## WHAT TO EXTRACT

### 1. Policy Type
Determine if this is "scoring" (point values given) or "rule_based" (count thresholds only).

### 2. Severity/Outcome Categories
What categories exist? (e.g., "Mild", "Moderate", "Severe" or "Poor", "Fair", "Good", "Excellent")

### 3. For SCORING-BASED policies:
- Extract EXACT point values for each category (can be positive or negative)
- Extract modifier point adjustments

### 4. For RULE-BASED policies:
- Extract count threshold rules (e.g., "2+ severe incidents" → "Critical")
- No point values - these use counts directly

### 5. Verdict Rules
- For SCORING: thresholds on total score (e.g., ">= 8" → "Critical")
- For RULE-BASED: count conditions (e.g., "severe_count >= 2" → "Critical")

## OUTPUT FORMAT

```json
{{
  "core_concepts": {{
    "primary_topic": "<main subject>",
    "explicit_terms": ["<terms from agenda>"]
  }},

  "policy_type": "<scoring OR rule_based>",

  "evaluation_criteria": {{
    "what_counts": "<from agenda>",
    "what_does_not_count": "<from agenda>"
  }},

  "categories": {{
    "field_name": "<what to call this: INCIDENT_SEVERITY, OUTCOME, QUALITY_LEVEL, etc.>",
    "values": [
      {{"name": "<category name>", "description": "<from agenda>"}}
    ]
  }},

  "scoring_system": {{
    "has_scoring": <true if point values are specified, false otherwise>,
    "description": "<how scoring works, or 'N/A - rule-based' if no scoring>",
    "base_point_categories": [
      {{"category": "<exact name>", "points": <exact number, can be negative>}}
    ],
    "modifiers": [
      {{"name": "<modifier name>", "condition": "<when it applies>", "points": <exact number>}}
    ]
  }},

  "verdict_rules": {{
    "type": "<scoring OR rule_based>",
    "verdicts": ["<verdict1>", "<verdict2>", "<verdict3>"],
    "rules": [
      {{
        "verdict": "<verdict label>",
        "condition_type": "<score_threshold OR count_threshold>",
        "condition": "<exact condition: '>= 8' for scoring, or 'severe_count >= 2' for rule-based>",
        "description": "<human readable rule from agenda>"
      }}
    ],
    "default_verdict": "<verdict when no rules match>"
  }},

  "account_handling": {{
    "field_name": "<what to call the account type field - e.g., ACCOUNT_TYPE, EVIDENCE_SOURCE, REPORTER_TYPE>",
    "types": [
      // Extract ALL account types defined in the agenda - do NOT assume firsthand/secondhand/general
      // {{"type": "<name from agenda>", "description": "<from agenda>", "counts_for_verdict": <from agenda>}}
    ]
  }},

  "description_field": {{
    "name": "<what to call the incident description field - e.g., SPECIFIC_INCIDENT, DESCRIPTION, DETAILS>",
    "purpose": "<purpose from agenda context>"
  }},

  "extraction_fields": [
    // List other domain-specific fields from agenda (severity categories, modifiers, etc.)
    {{"name": "<field mentioned in agenda>", "values": ["<possible values>"]}}
  ]
}}
```

CRITICAL:
1. First determine if this is "scoring" (has point values) or "rule_based" (count thresholds only)
2. Extract EXACT numbers - don't use template values
3. For rule-based policies, verdict rules use counts (e.g., "severe_count >= 2"), NOT scores

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

IMPORTANT: Check observations.policy_type to know if this is "scoring" or "rule_based":
- scoring: Need BASE_POINTS, MODIFIER_POINTS, SCORE computations
- rule_based: Need severity counts (N_SEVERE, N_MODERATE, etc.) and count-based verdict rules

For keywords, reason about:
- What words would reviewers use when discussing the concepts from the observations?
- What terms indicate each category level according to the agenda's definitions?
- What phrases distinguish firsthand from secondhand accounts?

Output your plan:

```json
{{
  "policy_type": "<copy from observations.policy_type: 'scoring' or 'rule_based'>",

  "keyword_strategy": {{
    "reasoning": "<explain how you derived keywords from the observed concepts>",

    "from_core_concepts": {{
      "primary_terms": ["<words derived from primary_topic>"],
      "related_terms": ["<words derived from related_concepts>"],
      "derivation_notes": "<how these connect to the agenda>"
    }},

    "from_category_definitions": {{
      "category_indicators": {{
        "<category1>": ["<words that match this category's definition>"],
        "<category2>": ["<words that match this category's definition>"]
      }},
      "derivation_notes": "<how these connect to category descriptions>"
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
        "name": "<from observations.account_handling.field_name>",
        "purpose": "Distinguish account types per observations.account_handling.types",
        "values": "<from observations.account_handling.types>",
        "based_on": "observations.account_handling"
      }},
      {{
        "name": "<from observations.categories.field_name>",
        "purpose": "Classify category level",
        "values": "<from observations.categories.values>",
        "based_on": "observations.categories"
      }},
      {{
        "name": "<from observations.description_field.name>",
        "purpose": "<from observations.description_field.purpose>",
        "based_on": "observations.description_field"
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
    "reasoning": "<explain the approach based on policy_type>",

    "for_scoring_type": {{
      "base_scoring": "<how to compute base points using observations.scoring_system.base_point_categories>",
      "modifier_scoring": "<how to apply modifiers using observations.scoring_system.modifiers>",
      "aggregation": "SCORE = BASE_POINTS + MODIFIER_POINTS"
    }},

    "for_rule_based_type": {{
      "counts_needed": ["<N_SEVERE>", "<N_MODERATE>", "<etc based on verdict rules>"],
      "count_logic": "<how to count each category>"
    }},

    "verdict_rules": {{
      "type": "<scoring or rule_based>",
      "rules_from_observations": "<copy observations.verdict_rules.rules>",
      "edge_cases": "<any special handling needed>"
    }}
  }},

  "search_strategy": {{
    "early_stop_condition": "<for scoring: 'SCORE >= X', for rule_based: 'N_SEVERE >= Y'>",
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

ACT_PROMPT = '''Generate the complete Formula Seed based on observations.

## ORIGINAL OBSERVATIONS

{observations}

## GENERATION PLAN

{plan}

## CRITICAL: CHECK POLICY TYPE FIRST

Look at observations.policy_type to determine the structure:
- If "scoring": Use point-based BASE_POINTS, MODIFIER_POINTS, SCORE, and score thresholds for VERDICT
- If "rule_based": Use severity counts and count thresholds for VERDICT (NO points)

---

## FOR SCORING-BASED POLICIES (observations.policy_type == "scoring")

### EXTRACTION FIELDS
- <observations.account_handling.field_name>: enum (values from observations.account_handling.types)
- <CATEGORY_FIELD> (from observations.categories.field_name): enum with category values
- <observations.description_field.name>: string for <observations.description_field.purpose>
- Modifier fields: one per modifier in observations.scoring_system.modifiers

### COMPUTE OPERATIONS
1. N_INCIDENTS: count where <account_field> = <counting_account_type>
2. BASE_POINTS: sum using EXACT point values from observations.scoring_system.base_point_categories
   Example: "CASE WHEN OUTCOME = 'severe' THEN 15 WHEN OUTCOME = 'moderate' THEN 5 ... ELSE 0 END"
3. MODIFIER_POINTS: sum using EXACT point values from observations.scoring_system.modifiers
   Example: "(CASE WHEN MOD1 = 'yes' THEN 3 ELSE 0 END) + (CASE WHEN MOD2 = 'yes' THEN -5 ELSE 0 END)"
4. SCORE: BASE_POINTS + MODIFIER_POINTS
5. VERDICT: case on SCORE using observations.verdict_rules.rules conditions

### VERDICT FORMAT (SCORING)

**CRITICAL - EXACT VERDICT LABELS**: Copy verdict labels EXACTLY from observations.verdict_rules.verdicts.
Do NOT simplify or modify labels. If observations say "Critical Risk", use "Critical Risk" - NOT "Critical".
The verdicts array contains the EXACT strings to use in your rules.

Use observations.verdict_rules.rules to build:
```json
{{"name": "VERDICT", "op": "case", "source": "SCORE", "rules": [
  {{"when": ">= 8", "then": "<EXACT label from observations.verdict_rules.verdicts[0]>"}},
  {{"when": ">= 4", "then": "<EXACT label from observations.verdict_rules.verdicts[1]>"}},
  {{"else": "<EXACT label from observations.verdict_rules.verdicts[2]>"}}
]}}
```

---

## FOR RULE-BASED POLICIES (observations.policy_type == "rule_based")

### EXTRACTION FIELDS
- <observations.account_handling.field_name>: enum (values from observations.account_handling.types)
- <CATEGORY_FIELD> (from observations.categories.field_name): enum with category values
- <observations.description_field.name>: string for <observations.description_field.purpose>
- Any additional fields needed for compound conditions

### COMPUTE OPERATIONS
1. N_INCIDENTS: count where <account_field> = <counting_account_type>
2. N_SEVERE: count where <account_field> = <counting_account_type> AND <CATEGORY_FIELD> = "severe"
3. N_MODERATE: count where <account_field> = <counting_account_type> AND <CATEGORY_FIELD> = "moderate"
4. (Add more counts as needed for verdict rules)
5. VERDICT: case using COUNT conditions (NOT score)

### VERDICT FORMAT (RULE-BASED)

**CRITICAL - EXACT VERDICT LABELS**: Copy verdict labels EXACTLY from observations.verdict_rules.verdicts.
Do NOT simplify or modify labels. If observations say "Critical Risk", use "Critical Risk" - NOT "Critical".
The verdicts array contains the EXACT strings to use in your rules.

Build verdict from observations.verdict_rules.rules using count conditions:
```json
{{"name": "VERDICT", "op": "case", "source": "N_SEVERE", "rules": [
  {{"when": ">= 2", "then": "<EXACT from observations.verdict_rules.verdicts>"}},
  {{"else": null}}
], "fallback_source": "N_MODERATE", "fallback_rules": [
  {{"when": ">= 2", "then": "<EXACT from observations.verdict_rules.verdicts>"}},
  {{"else": "<EXACT from observations.verdict_rules.verdicts>"}}
]}}
```

OR use compound conditions in a single case:
```json
{{"name": "VERDICT", "op": "case", "rules": [
  {{"when": "N_SEVERE >= 2", "then": "<EXACT from observations.verdict_rules.verdicts[0]>"}},
  {{"when": "N_MODERATE >= 2", "then": "<EXACT from observations.verdict_rules.verdicts[1]>"}},
  {{"else": "<EXACT from observations.verdict_rules.verdicts[2]>"}}
]}}
```

---

## OUTPUT FORMAT - FOLLOW EXACTLY

```json
{{
  "task_name": "<from observations.core_concepts.primary_topic>",

  "filter": {{
    "keywords": ["<from plan.keyword_strategy - combine all relevant terms>"]
  }},

  "extract": {{
    "fields": [
      // REQUIRED: Account type field with values as DICT (not array!)
      {{"name": "<observations.account_handling.field_name>", "type": "enum", "values": {{
        "<type1>": "<description1>",
        "<type2>": "<description2>"
      }}}},
      // REQUIRED: Category field with values as DICT
      {{"name": "<observations.categories.field_name>", "type": "enum", "values": {{
        "<cat1>": "<cat1 description>",
        "<cat2>": "<cat2 description>"
      }}}},
      // REQUIRED: Description field
      {{"name": "<observations.description_field.name>", "type": "string", "description": "<observations.description_field.purpose>"}},
      // REQUIRED: Include ALL modifier-related fields from observations.extraction_fields
      // Each field referenced in MODIFIER_POINTS MUST be defined here!
    ]
  }},

  "compute": [
    // EXACT FORMAT FOR EACH OPERATION:
    // count: {{"name": "N_INCIDENTS", "op": "count", "where": {{"<account_field>": "<counting_type>"}}}}
    // sum:   {{"name": "BASE_POINTS", "op": "sum", "expr": "CASE WHEN ... THEN ... ELSE 0 END", "where": {{"<account_field>": "<counting_type>"}}}}
    // expr:  {{"name": "SCORE", "op": "expr", "expr": "BASE_POINTS + MODIFIER_POINTS"}}
    // case:  {{"name": "VERDICT", "op": "case", "source": "SCORE", "rules": [{{"when": ">= 8", "then": "Critical"}}, {{"else": "Low"}}]}}
  ],

  "output": ["VERDICT", "<SCORE or relevant counts>", "N_INCIDENTS"],

  "search_strategy": {{
    "priority_keywords": ["<high-signal terms from plan>"],
    "priority_expr": "len(keyword_hits) * 2 + (1.0 if is_recent else 0.5)",
    "stopping_condition": "<condition based on policy type>",
    "early_verdict_expr": "<Python ternary for early stopping>",
    "use_embeddings_when": "len(keyword_matched) < 5"
  }},

  "expansion_hints": {{
    "domain": "<primary_topic>",
    "expand_on": ["<category names>"]
  }},

  "verdict_metadata": {{
    "verdicts": ["<list all possible verdict labels in order from lowest to highest severity>"],
    "ordered": true,
    "severe_verdicts": ["<highest severity verdict(s) that should trigger early exit>"],
    "score_variable": "<SCORE or primary aggregation variable name>"
  }}
}}
```

## CRITICAL RULES - MUST FOLLOW

1. **Values format**: extract.fields[].values MUST be a DICT (not array!): {{"value1": "desc1", "value2": "desc2"}}
2. **Compute format**: Each compute operation MUST have "op" key (count/sum/expr/case), NOT "expression"
3. **Field references**: ALL fields used in compute MUST exist in extract.fields
   - If MODIFIER_POINTS uses ASSURANCE_OF_SAFETY, add it to extract.fields!
   - If compute references STAFF_RESPONSE, add it to extract.fields!
4. **sum operations**: Use {{"op": "sum", "expr": "CASE WHEN...", "where": {{...}}}} format
5. **case operations**: Use {{"op": "case", "source": "...", "rules": [...]}} format
6. **expr operations**: Use {{"op": "expr", "expr": "..."}} for combining computed values
7. **VERDICT LABELS**: Use EXACT verdict strings from observations.verdict_rules.verdicts - copy character-for-character!
   - If observations say "Critical Risk", use "Critical Risk" NOT "Critical"
   - If observations say "Low Risk", use "Low Risk" NOT "Low"

Output ONLY the JSON:

```json
'''


def _extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks and common errors."""
    import re

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
    # This is a heuristic - replace newlines inside strings
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
        response_format={"type": "json_object"},
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
        response_format={"type": "json_object"},
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
        response_format={"type": "json_object"},
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
