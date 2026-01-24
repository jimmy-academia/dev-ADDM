"""Phase 1: Formula Seed Generation.

LLM reads agenda/query and produces a Formula Seed (executable JSON specification).

Two approaches available (configured via AMOSConfig.phase1_approach):
1. PLAN_AND_ACT (legacy): OBSERVE → PLAN → ACT pipeline, LLM generates seed directly
2. HYBRID (recommended): NL → PolicyYAML → deterministic compiler

The HYBRID approach splits the work:
- LLM is good at: understanding NL, producing structured YAML
- LLM is bad at: complex count operations, enum filtering logic
- Solution: LLM does understanding, compiler does precision

All approaches include validation after generation.
"""

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

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
from .seed_compiler import compile_yaml_to_seed, validate_policy_yaml, PolicyYAMLValidationError

logger = logging.getLogger(__name__)

# Maximum attempts to fix structural errors
MAX_FIX_ATTEMPTS = 3


# =============================================================================
# TEXT2YAML Prompt for Hybrid Approach
# =============================================================================

TEXT2YAML_PROMPT = '''You are a policy specification expert. Analyze this query and generate a PolicyYAML specification.

## QUERY/AGENDA

{agenda}

## YOUR TASK

Read the query carefully and produce a PolicyYAML that captures:
1. What fields/signals to extract from reviews
2. What values each field can have
3. What verdict rules determine the outcome

## OUTPUT FORMAT (YAML)

```yaml
policy_type: count_rule_based  # use "count_rule_based" for counting occurrences, "scoring" for point systems

task_name: "<brief description of what this policy evaluates>"

terms:
  - name: <FIELD_NAME_IN_CAPS>
    values: [<value1>, <value2>, ...]  # all possible values for this field
    descriptions:
      <value1>: "<what this value means>"
      <value2>: "<what this value means>"

verdicts: [<verdict1>, <verdict2>, <verdict3>]  # exactly the verdicts from the query

rules:
  # Non-default rules: checked in order
  - verdict: <verdict_label>
    logic: ANY  # ANY = at least one condition; ALL = all conditions must be true
    conditions:
      - field: <FIELD_NAME>
        values: [<value1>, <value2>]  # which values count toward this verdict
        min_count: <N>  # how many needed (extract from query, e.g., "2 or more" → 2)

  # Default rule: applied when no other rules match
  - verdict: <default_verdict>
    default: true
```

## CRITICAL RULES

1. **FIELD NAMES**: Use UPPERCASE_WITH_UNDERSCORES (e.g., PRICE_PERCEPTION, INCIDENT_SEVERITY)

2. **VALUES**: List ALL possible values for each field. These become enum options for extraction.

3. **VERDICTS**: Use EXACT verdict labels from the query. Copy them character-for-character.

4. **THRESHOLDS**: Extract exact numbers from the query:
   - "2 or more" → min_count: 2
   - "at least 10" → min_count: 10
   - "multiple" → min_count: 2 (standard interpretation)

5. **RULE ORDER**: More specific rules first, default rule last.

6. **DEFAULT RULE**: Exactly ONE rule must have `default: true`

7. **CONDITION VALUES**: The `values` in a condition specify WHICH enum values to count.
   Example: To count "positive reviews", use values that indicate positive sentiment.

## EXAMPLE

For query: "Recommend restaurants with good value. Recommend if 10+ reviews say it's a bargain or good value. Not Recommended if 5+ say it's overpriced or a ripoff. Otherwise Neutral."

```yaml
policy_type: count_rule_based

task_name: "Evaluate restaurant value for money"

terms:
  - name: PRICE_PERCEPTION
    values: [bargain, good_value, fair, overpriced, ripoff]
    descriptions:
      bargain: "Exceptional value, prices much lower than expected for quality"
      good_value: "Good value for money, reasonable prices"
      fair: "Average value, neither good nor bad deal"
      overpriced: "Prices higher than expected for quality"
      ripoff: "Extremely overpriced, poor value"

verdicts: [Recommend, Not Recommended, Neutral]

rules:
  - verdict: Recommend
    logic: ANY
    conditions:
      - field: PRICE_PERCEPTION
        values: [bargain, good_value]
        min_count: 10

  - verdict: Not Recommended
    logic: ANY
    conditions:
      - field: PRICE_PERCEPTION
        values: [overpriced, ripoff]
        min_count: 5

  - verdict: Neutral
    default: true
```

## NOW GENERATE

Analyze the query above and output ONLY the PolicyYAML (no explanation before or after):

```yaml
'''


# =============================================================================
# Part-by-Part Prompts for Structured Extraction
# =============================================================================

EXTRACT_TERMS_PROMPT = '''You are extracting term definitions from a policy query.

## QUERY SECTION

{section}

## YOUR TASK

Extract ALL terms/fields defined in this section. Each term has:
- A name (convert to UPPERCASE_WITH_UNDERSCORES)
- Possible values (the options/categories)
- Descriptions for each value

## OUTPUT FORMAT (YAML)

```yaml
terms:
  - name: FIELD_NAME
    values: [value1, value2, value3]
    descriptions:
      value1: "description of what this value means"
      value2: "description of what this value means"
```

## RULES

1. Use UPPERCASE_WITH_UNDERSCORES for field names (e.g., PRICE_PERCEPTION, INCIDENT_SEVERITY)
2. Use lowercase_with_underscores for values (e.g., good_value, very_poor)
3. Include ALL values mentioned, even neutral ones
4. Copy descriptions verbatim when possible

Output ONLY the YAML, no explanation:

```yaml
'''

EXTRACT_SCORING_PROMPT = '''You are extracting a scoring system from a policy query.

## QUERY SECTION

{section}

## AVAILABLE TERMS AND VALUES (USE ONLY THESE)

{terms_summary}

## YOUR TASK

Extract the scoring rules using ONLY the field names and values listed above.

1. Base points: What points are assigned for each value of the outcome field
2. Modifiers: Additional points for other field values

CRITICAL: The outcome_field MUST be one of the Available Terms above.
CRITICAL: base_points keys MUST use EXACT values from that term's value list.
CRITICAL: modifier field/value pairs MUST use EXACT terms and values from above.

## OUTPUT FORMAT (YAML)

```yaml
policy_type: scoring

scoring:
  outcome_field: FIELD_NAME  # MUST be from Available Terms
  base_points:  # Keys MUST be exact values from that field
    value1: 10
    value2: 5
    value3: 0
    value4: -5
  modifiers:
    - field: OTHER_FIELD  # MUST be from Available Terms
      value: some_value   # MUST be from that field's values
      points: 3
```

## RULES

1. Extract EXACT point values from the query (e.g., "+10 points", "-5 points")
2. outcome_field: Pick the MAIN outcome field from Available Terms (usually has severity-like values)
3. base_points: Map the outcome field's VALUES to points
4. modifiers: Use OTHER field+value pairs from Available Terms
5. If no scoring system exists, output: `policy_type: count_rule_based`

Output ONLY the YAML, no explanation:

```yaml
'''

EXTRACT_VERDICTS_PROMPT = '''You are extracting verdict rules from a policy query.

## QUERY SECTION

{section}

## AVAILABLE TERMS (use ONLY these field names)

{context}

## YOUR TASK

Extract the verdict rules that determine the final outcome.

CRITICAL: You may ONLY use field names listed in "Available Terms" above.
Do NOT invent new field names.

## OUTPUT FORMAT (YAML)

For SCORING policies (point-based):
```yaml
verdicts: [Verdict1, Verdict2, Verdict3]

rules:
  - verdict: Verdict1
    condition: score >= 44
  - verdict: Verdict2
    condition: score <= -13
  - verdict: Verdict3
    default: true
```

For COUNT-BASED policies:
```yaml
verdicts: [Verdict1, Verdict2, Verdict3]

rules:
  - verdict: Verdict1
    logic: ANY
    conditions:
      - field: FIELD_NAME
        values: [value1, value2]
        min_count: 10
  - verdict: Verdict2
    logic: ALL
    conditions:
      - field: FIELD_NAME
        values: [value3]
        min_count: 5
  - verdict: Verdict3
    default: true
```

## RULES

1. Use EXACT verdict labels from the query (copy character-for-character)
2. Extract EXACT threshold numbers (e.g., "44 or higher" → >= 44)
3. For count-based: min_count is how many reviews needed
4. Exactly ONE rule must have `default: true`
5. Default rule should be the "neutral" or "middle" verdict
6. ONLY use field names from "Available Terms" - NEVER invent new names

Output ONLY the YAML, no explanation:

```yaml
'''


# =============================================================================
# JSON/YAML Extraction and Usage Accumulation
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


def _extract_yaml_from_response(response: str) -> str:
    """Extract YAML content from LLM response.

    Handles:
    - Markdown code blocks (```yaml ... ```)
    - Plain YAML text
    - Response with explanations before/after
    """
    response = response.strip()

    # Try to find YAML in markdown code block
    if "```yaml" in response:
        start = response.find("```yaml") + 7
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    # Try generic code block
    if "```" in response:
        start = response.find("```") + 3
        # Skip language identifier if present
        first_newline = response.find("\n", start)
        if first_newline > start:
            start = first_newline + 1
        end = response.find("```", start)
        if end > start:
            return response[start:end].strip()

    # Assume entire response is YAML
    return response


def _parse_yaml_safely(yaml_str: str) -> Dict[str, Any]:
    """Parse YAML string with error handling."""
    try:
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")
        return data
    except yaml.YAMLError as e:
        # Try to fix common issues
        fixed = yaml_str

        # Fix unquoted strings that look like numbers
        fixed = re.sub(r":\s*(\d+\.\d+)(?=\s*$)", r': "\1"', fixed, flags=re.MULTILINE)

        # Fix single quotes inside single-quoted strings by converting to double quotes
        # Pattern: 'text with 'inner' quote' -> "text with 'inner' quote"
        def fix_single_quotes(line):
            # If line has a value after colon, try to fix quote issues
            if ':' in line:
                key, _, value = line.partition(':')
                value = value.strip()
                # If value starts with single quote but has issues, use double quotes
                if value.startswith("'") and value.count("'") > 2:
                    # Extract content and re-quote with double quotes
                    content = value[1:-1] if value.endswith("'") else value[1:]
                    # Escape any double quotes in content
                    content = content.replace('"', '\\"')
                    return f'{key}: "{content}"'
            return line

        fixed_lines = [fix_single_quotes(line) for line in fixed.split('\n')]
        fixed = '\n'.join(fixed_lines)

        try:
            data = yaml.safe_load(fixed)
            if isinstance(data, dict):
                return data
        except yaml.YAMLError:
            pass

        # Try even more aggressive fixing - remove problematic value content entirely
        # Just keep the key with a placeholder
        lines_simplified = []
        for line in yaml_str.split('\n'):
            if ':' in line and "'" in line:
                key = line.split(':')[0]
                # Keep just the key with a generic value
                lines_simplified.append(f'{key}: "value"')
            else:
                lines_simplified.append(line)

        try:
            data = yaml.safe_load('\n'.join(lines_simplified))
            if isinstance(data, dict):
                logger.warning("Used simplified YAML parsing, some descriptions may be lost")
                return data
        except yaml.YAMLError:
            pass

        raise ValueError(f"YAML parse error: {e}")


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
    yaml_str = _extract_yaml_from_response(response)
    try:
        data = _parse_yaml_safely(yaml_str)
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
    yaml_str = _extract_yaml_from_response(response)
    data = _parse_yaml_safely(yaml_str)

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
    yaml_str = _extract_yaml_from_response(response)
    data = _parse_yaml_safely(yaml_str)

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
        # For scoring policies, convert score-based conditions to count-based
        # The rules might have "condition: score >= X" format
        # Convert to conditions with the outcome field
        scoring_info = scoring.get("scoring", {}) if scoring else {}
        outcome_field = scoring_info.get("outcome_field", "INCIDENT_SEVERITY")
        base_points = scoring_info.get("base_points", {})

        processed_rules = []
        for rule in rules:
            new_rule = {"verdict": rule.get("verdict", "")}

            if rule.get("default"):
                new_rule["default"] = True
            elif "condition" in rule:
                # Score-based condition: "score >= 44" or "score <= -13"
                cond = rule.get("condition", "")

                # Parse threshold
                threshold = None
                is_positive = ">=" in cond
                match = re.search(r'(-?\d+)', cond)
                if match:
                    threshold = int(match.group(1))

                if threshold is not None:
                    # Determine which values contribute positively/negatively
                    if is_positive:
                        # Positive threshold: count values with positive points
                        positive_values = [v for v, pts in base_points.items() if pts > 0]
                        if positive_values:
                            # Estimate min_count from threshold and average points
                            avg_pts = sum(base_points[v] for v in positive_values) / len(positive_values)
                            min_count = max(2, int(threshold / avg_pts)) if avg_pts > 0 else 5
                            new_rule["logic"] = "ANY"
                            new_rule["conditions"] = [{
                                "field": outcome_field,
                                "values": positive_values,
                                "min_count": min_count,
                            }]
                    else:
                        # Negative threshold: count values with negative points
                        negative_values = [v for v, pts in base_points.items() if pts < 0]
                        if negative_values:
                            # Estimate min_count from threshold and average points
                            avg_pts = abs(sum(base_points[v] for v in negative_values) / len(negative_values))
                            min_count = max(2, int(abs(threshold) / avg_pts)) if avg_pts > 0 else 3
                            new_rule["logic"] = "ANY"
                            new_rule["conditions"] = [{
                                "field": outcome_field,
                                "values": negative_values,
                                "min_count": min_count,
                            }]
            elif "conditions" in rule:
                # Already has conditions list, use as-is
                new_rule["logic"] = rule.get("logic", "ANY")
                new_rule["conditions"] = rule.get("conditions", [])

            # Only add rule if it has conditions or is default
            if new_rule.get("default") or new_rule.get("conditions"):
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
    errors = _validate_formula_seed(seed)

    if errors:
        output.warn(f"[Phase 1 Parts] Seed validation: {len(errors)} errors")
        logger.warning(f"Compiled seed has validation errors: {errors}")
        raise ValueError(f"Compiled seed validation failed: {errors}")
    else:
        output.status(f"[Phase 1 Parts] Seed validation: passed")
    report("VALIDATE_SEED", 98, "passed")

    # Return seed
    total_usage = _accumulate_usage(all_usages)
    wall_clock_ms = (time.perf_counter() - start_time) * 1000
    total_usage["wall_clock_ms"] = wall_clock_ms

    # Keep _policy_yaml for saving
    seed_clean = {k: v for k, v in seed.items() if not k.startswith("_") or k == "_policy_yaml"}

    n_terms = len(yaml_data.get("terms", []))
    n_rules = len(yaml_data.get("rules", []))
    output.status(f"[Phase 1 Parts] Complete: {n_terms} terms, {n_rules} rules")

    return seed_clean, total_usage


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
        if not u:
            continue
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
# Plan-and-Act Pipeline Steps (Legacy Approach)
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
# Hybrid Approach: Text → YAML → Compiled Seed
# =============================================================================

async def _generate_policy_yaml(
    agenda: str,
    llm: LLMService,
    policy_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate PolicyYAML from natural language agenda using LLM.

    This is the first step of the hybrid approach.

    Args:
        agenda: The task agenda/query prompt
        llm: LLM service for API calls
        policy_id: Policy identifier for context

    Returns:
        Tuple of (policy_yaml_dict, usage)
    """
    output.status(f"[Phase 1 Hybrid] Step 1/2: Generating PolicyYAML...")

    prompt = TEXT2YAML_PROMPT.format(agenda=agenda)
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response, usage = await llm.call_async_with_usage(
        messages,
        context={"phase": "phase1_hybrid", "step": "text2yaml", "policy_id": policy_id},
    )
    latency_ms = (time.time() - start_time) * 1000
    usage["latency_ms"] = latency_ms

    # Log to debug file
    if debug_logger := get_debug_logger():
        debug_logger.log_llm_call(
            sample_id=policy_id,
            phase="phase1_text2yaml",
            prompt=prompt,
            response=response,
            model=llm._config.get("model", "unknown"),
            latency_ms=latency_ms,
        )

    # Extract and parse YAML
    yaml_str = _extract_yaml_from_response(response)
    logger.debug(f"Extracted YAML ({len(yaml_str)} chars):\n{yaml_str[:500]}...")

    try:
        yaml_data = _parse_yaml_safely(yaml_str)
    except ValueError as e:
        logger.error(f"Failed to parse YAML response: {e}")
        logger.debug(f"Raw response:\n{response}")
        raise ValueError(f"Failed to parse LLM response as YAML: {e}")

    n_terms = len(yaml_data.get("terms", []))
    n_rules = len(yaml_data.get("rules", []))
    output.status(f"[Phase 1 Hybrid] PolicyYAML generated: {n_terms} terms, {n_rules} rules")

    return yaml_data, usage


async def generate_formula_seed_hybrid(
    agenda: str,
    policy_id: str,
    llm: LLMService,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate Formula Seed using hybrid approach: NL → PolicyYAML → compiled seed.

    This approach splits the work:
    - LLM generates simple PolicyYAML (good at understanding, bad at precision)
    - Deterministic compiler transforms to Formula Seed (guaranteed correct)

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

    # Step 1: LLM generates PolicyYAML from agenda
    report("TEXT2YAML", 10, "generating yaml")
    logger.info(f"Phase 1 Hybrid: Step 1/2 TEXT2YAML for {policy_id}")

    yaml_data, usage1 = await _generate_policy_yaml(agenda, llm, policy_id)
    all_usages.append(usage1)
    report("TEXT2YAML", 40, "complete")

    # Step 2: Validate PolicyYAML
    report("VALIDATE_YAML", 45, "validating yaml")
    try:
        validate_policy_yaml(yaml_data)
        output.status(f"[Phase 1 Hybrid] PolicyYAML validation: passed")
    except PolicyYAMLValidationError as e:
        logger.warning(f"PolicyYAML validation failed: {e}")
        output.warn(f"[Phase 1 Hybrid] PolicyYAML validation failed: {e}")
        raise ValueError(f"PolicyYAML validation failed: {e}")
    report("VALIDATE_YAML", 50, "passed")

    # Step 3: Deterministic compilation to Formula Seed
    report("COMPILE", 55, "compiling seed")
    output.status(f"[Phase 1 Hybrid] Step 2/2: Compiling to Formula Seed...")
    logger.info(f"Phase 1 Hybrid: Step 2/2 COMPILE for {policy_id}")

    try:
        seed = compile_yaml_to_seed(yaml_data, validate=False)  # Already validated
    except Exception as e:
        logger.error(f"Seed compilation failed: {e}")
        raise ValueError(f"Failed to compile PolicyYAML to seed: {e}")

    seed["_approach"] = "hybrid"
    seed["_policy_yaml"] = yaml_data
    report("COMPILE", 75, "complete")

    # Step 4: Validate the compiled Formula Seed
    report("VALIDATE_SEED", 80, "validating seed")
    errors = _validate_formula_seed(seed)

    if errors:
        output.warn(f"[Phase 1 Hybrid] Seed validation: {len(errors)} errors")
        logger.warning(f"Compiled seed has validation errors: {errors}")
        # For hybrid approach, compilation errors are bugs - don't try to fix via LLM
        raise ValueError(f"Compiled seed validation failed: {errors}")
    else:
        output.status(f"[Phase 1 Hybrid] Seed validation: passed")
        report("VALIDATE_SEED", 95, "passed")

    # Return seed
    total_usage = _accumulate_usage(all_usages)
    wall_clock_ms = (time.perf_counter() - start_time) * 1000
    total_usage["wall_clock_ms"] = wall_clock_ms

    # Keep _policy_yaml for saving, remove other intermediates
    seed_clean = {k: v for k, v in seed.items() if not k.startswith("_") or k == "_policy_yaml"}

    return seed_clean, total_usage


# =============================================================================
# Main Entry Points
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

    Selects between approaches based on config.phase1_approach:
    - PARTS: Part-by-part extraction (recommended) - extracts terms, scoring, verdicts separately
    - HYBRID: NL → PolicyYAML → deterministic compiler - single-shot YAML generation
    - PLAN_AND_ACT: Legacy 3-step pipeline (OBSERVE → PLAN → ACT)

    Args:
        agenda: The task agenda/query prompt
        policy_id: Policy identifier (e.g., "G1_allergy_V2")
        llm: LLM service for API calls
        config: AMOSConfig with phase1_approach setting
        progress_callback: Optional callback for progress updates.

    Returns:
        Tuple of (formula_seed, usage_dict)
    """
    from .config import Phase1Approach

    if config.phase1_approach == Phase1Approach.PARTS:
        return await generate_formula_seed_parts(
            agenda=agenda,
            policy_id=policy_id,
            llm=llm,
            progress_callback=progress_callback,
        )
    elif config.phase1_approach == Phase1Approach.HYBRID:
        return await generate_formula_seed_hybrid(
            agenda=agenda,
            policy_id=policy_id,
            llm=llm,
            progress_callback=progress_callback,
        )
    else:
        # PLAN_AND_ACT
        return await generate_formula_seed(
            agenda=agenda,
            policy_id=policy_id,
            llm=llm,
            progress_callback=progress_callback,
        )
