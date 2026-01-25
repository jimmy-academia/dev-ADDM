"""Deterministic PolicyYAML → Formula Seed compiler.

This module transforms PolicyYAML (simple, LLM-generated) into Formula Seed
(complex, execution-ready). The transformation is fully deterministic:
same YAML always produces same seed.

The PolicyYAML format is designed to be easy for LLMs to generate:
- Explicit field→values mappings
- Simple rule structures with field references
- No complex compute expressions

The compiler generates:
- Proper count operations with correct enum filtering
- CASE-based verdict rules
- All required Formula Seed metadata
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PolicyYAMLValidationError(Exception):
    """Raised when PolicyYAML structure is invalid."""
    pass


def validate_policy_yaml(yaml_data: Dict[str, Any]) -> None:
    """Validate PolicyYAML structure before compilation.

    Checks:
    1. Required keys present (terms, verdicts, rules)
    2. Field names in rules match terms
    3. Rule values exist in term definitions
    4. Default verdict exists

    Args:
        yaml_data: Parsed PolicyYAML dict

    Raises:
        PolicyYAMLValidationError: If validation fails
    """
    errors = []

    # Check required keys
    required_keys = ["terms", "verdicts", "rules"]
    for key in required_keys:
        if key not in yaml_data:
            errors.append(f"Missing required key: '{key}'")

    if errors:
        raise PolicyYAMLValidationError("; ".join(errors))

    # Build term name -> values mapping
    terms = yaml_data.get("terms", [])
    term_map: Dict[str, List[str]] = {}
    for term in terms:
        name = term.get("name")
        values = term.get("values", [])
        if not name:
            errors.append("Term missing 'name' field")
            continue
        if not values:
            errors.append(f"Term '{name}' has no values")
            continue
        # Convert values to strings (YAML may parse true/false as bools)
        term_map[name.upper()] = [str(v).lower() for v in values]

    # Validate verdicts
    verdicts = yaml_data.get("verdicts", [])
    if not verdicts:
        errors.append("'verdicts' list is empty")

    # Validate rules
    rules = yaml_data.get("rules", [])
    has_default = False

    for i, rule in enumerate(rules):
        verdict = rule.get("verdict")
        if not verdict:
            errors.append(f"Rule {i} missing 'verdict'")
            continue

        # Check verdict is in verdicts list
        if verdict not in verdicts:
            errors.append(f"Rule {i} verdict '{verdict}' not in verdicts list")

        # Check for default rule
        if rule.get("default"):
            has_default = True
            continue

        # Validate conditions
        # For scoring policies, rules may have "condition" (singular) with score expression
        # For count policies, rules have "conditions" (plural) with field/values
        condition_singular = rule.get("condition", "")  # e.g., "score >= 12"
        conditions = rule.get("conditions", [])

        if not conditions and not condition_singular and not rule.get("default"):
            errors.append(f"Rule {i} ('{verdict}') has no conditions and is not default")

        # If this is a score-based condition, skip field validation (no field references)
        if condition_singular and "score" in condition_singular.lower():
            continue

        for j, cond in enumerate(conditions):
            field = cond.get("field")
            values = cond.get("values", [])

            if not field:
                errors.append(f"Rule {i} condition {j} missing 'field'")
                continue

            # Check field exists in terms
            field_upper = field.upper()
            if field_upper not in term_map:
                errors.append(
                    f"Rule {i} condition {j} references unknown field '{field}'. "
                    f"Available terms: {list(term_map.keys())}"
                )
                continue

            # Check values exist in term
            term_values = term_map[field_upper]
            for v in values:
                if str(v).lower() not in term_values:
                    errors.append(
                        f"Rule {i} condition {j}: value '{v}' not in {field} values. "
                        f"Available: {term_values}"
                    )

    if not has_default:
        errors.append("No default rule found (one rule must have 'default: true')")

    if errors:
        raise PolicyYAMLValidationError("; ".join(errors))


def _generate_count_name(field: str, values: List[str]) -> str:
    """Generate a descriptive count operation name.

    Examples:
        ("PRICE_PERCEPTION", ["bargain", "good_value"]) -> "N_PRICE_POSITIVE"
        ("SEVERITY", ["severe"]) -> "N_SEVERITY_SEVERE"
        ("QUALITY", ["poor", "terrible"]) -> "N_QUALITY_NEGATIVE"

    Args:
        field: Field name
        values: Values being counted

    Returns:
        Count variable name (e.g., "N_FIELD_SUFFIX")
    """
    field_upper = field.upper()

    # Common positive/negative value patterns
    positive_indicators = {"bargain", "steal", "good", "excellent", "great", "positive", "high"}
    negative_indicators = {"poor", "terrible", "overpriced", "ripoff", "bad", "negative", "low"}

    values_lower = {str(v).lower() for v in values}

    if values_lower & positive_indicators:
        suffix = "POSITIVE"
    elif values_lower & negative_indicators:
        suffix = "NEGATIVE"
    elif len(values) == 1:
        # Single value: use value name
        suffix = str(values[0]).upper().replace(" ", "_").replace("-", "_")
    else:
        # Multiple values: use generic suffix
        suffix = "MATCHED"

    return f"N_{field_upper}_{suffix}"


def _build_extract_fields(
    terms: List[Dict[str, Any]],
    account_handling: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build extract.fields from terms.

    Args:
        terms: List of term definitions from PolicyYAML
        account_handling: Optional account type configuration

    Returns:
        List of field definitions for Formula Seed
    """
    fields = []

    # Determine account field name
    if account_handling:
        account_field_name = account_handling.get("field_name", "ACCOUNT_TYPE")
        types = account_handling.get("types", [
            {"type": "Firsthand", "description": "Personal experience"},
            {"type": "Secondhand", "description": "Heard from others"},
            {"type": "General", "description": "General statement"},
        ])
        values = {t["type"]: t.get("description", t["type"]) for t in types}
        fields.append({
            "name": account_field_name,
            "type": "enum",
            "values": values,
        })
    else:
        account_field_name = "ACCOUNT_TYPE"
        # Default account type field
        fields.append({
            "name": account_field_name,
            "type": "enum",
            "values": {
                "Firsthand": "Direct personal experience",
                "Secondhand": "Information heard from others",
                "General": "General statement without personal experience",
            },
        })

    # Track added field names to avoid duplicates
    added_fields = {account_field_name.upper()}

    # Add term fields (skip if duplicate)
    for term in terms:
        name = term.get("name", "")
        values = term.get("values", [])
        descriptions = term.get("descriptions", {})

        if not name or not values:
            continue

        name_upper = name.upper()

        # Skip if this term duplicates any already-added field
        if name_upper in added_fields:
            logger.debug(f"Skipping term '{name}' - duplicates existing field")
            continue

        # Build case-insensitive description lookup
        # This handles mismatches like values=["mild"] with descriptions={"Mild": "..."}
        desc_lookup = {}
        for k, v_desc in descriptions.items():
            desc_lookup[str(k).lower()] = v_desc

        # Build values dict (convert to strings for YAML bool handling)
        values_dict = {}
        for v in values:
            v_str = str(v)
            v_lower = v_str.lower()
            # Try case-insensitive lookup first, fall back to generic description
            desc = desc_lookup.get(v_lower, f"Indicates {v_str}")
            values_dict[v_str] = desc

        fields.append({
            "name": name_upper,
            "type": "enum",
            "values": values_dict,
        })
        added_fields.add(name_upper)

    # Add description field (if not already added by terms)
    if "DESCRIPTION" not in added_fields:
        fields.append({
            "name": "DESCRIPTION",
            "type": "string",
            "description": "Brief description of the relevant finding",
        })

    return fields


def _determine_outcome_field(terms: List[Dict[str, Any]]) -> tuple:
    """Determine outcome field and none values from terms.

    Args:
        terms: List of term definitions

    Returns:
        Tuple of (outcome_field_name, none_values_list)
    """
    # Look for common outcome field patterns
    outcome_keywords = {"severity", "quality", "outcome", "level", "perception", "rating"}
    # None indicators to catch patterns that mean "no incident/signal to report":
    # - Explicit none: none, n/a, absent
    # - Negation prefix: no_, not_
    # - Note: DON'T include neutral signals like "average", "fair" as these are legitimate
    none_indicators = {
        "none", "n/a", "absent", "unknown",
        "no ", "no_", "no-", "not ", "not_",
        "nothing_special", "generic_forgettable",  # Specific values that mean "nothing noteworthy"
    }

    outcome_field = None
    none_values = []

    for term in terms:
        name = term.get("name", "").lower()
        values = term.get("values", [])

        # Check if this looks like an outcome field
        if any(kw in name for kw in outcome_keywords):
            outcome_field = term.get("name", "").upper()

            # Find none values
            for v in values:
                v_lower = str(v).lower()
                if any(ind in v_lower for ind in none_indicators):
                    none_values.append(str(v))

            break

    # Fallback: use first term as outcome field
    if not outcome_field and terms:
        outcome_field = terms[0].get("name", "OUTCOME").upper()
        for v in terms[0].get("values", []):
            if str(v).lower() in ("none", "n/a", "neutral"):
                none_values.append(str(v))

    if not none_values:
        none_values = ["none", "n/a"]

    return outcome_field, none_values


def _build_scoring_compute(
    rules: List[Dict[str, Any]],
    scoring_config: Dict[str, Any],
    account_field: str,
    counting_account_type: str,
) -> List[Dict[str, Any]]:
    """Build compute operations for scoring policies using SUM.

    This generates proper SUM-based scoring operations instead of converting
    to count-based, preserving the original threshold semantics.

    Args:
        rules: List of verdict rules (may have "condition: score >= X")
        scoring_config: Scoring configuration from PolicyYAML (base_points, modifiers, etc.)
        account_field: Name of account type field
        counting_account_type: Account type that counts for verdicts

    Returns:
        List of compute operations for Formula Seed
    """
    import re

    compute = []

    outcome_field = scoring_config.get("outcome_field", "INCIDENT_SEVERITY")
    base_points = scoring_config.get("base_points", {})
    modifiers = scoring_config.get("modifiers", [])

    # 1. BASE_POINTS: Sum severity points using CASE expression
    if base_points:
        case_parts = [f"WHEN {outcome_field}='{v}' THEN {pts}" for v, pts in base_points.items()]
        compute.append({
            "name": "BASE_POINTS",
            "op": "sum",
            "expr": f"CASE {' '.join(case_parts)} ELSE 0 END",
            "where": {account_field: [counting_account_type]},
        })

    # 2. MODIFIER operations (one per modifier, unique names)
    modifier_names = []
    used_mod_names = set()
    for i, mod in enumerate(modifiers):
        field = mod.get("field", "")
        value = mod.get("value", "")
        points = mod.get("points", 0)
        if field and value:
            # Generate unique name including both field and value
            # Sanitize value for variable name (uppercase, replace non-alphanumeric)
            value_sanitized = re.sub(r'[^A-Za-z0-9_]', '_', str(value).upper())
            base_name = f"MOD_{field.upper()}_{value_sanitized}"

            # Ensure uniqueness by adding suffix if needed
            name = base_name
            suffix = 1
            while name in used_mod_names:
                name = f"{base_name}_{suffix}"
                suffix += 1
            used_mod_names.add(name)

            modifier_names.append(name)
            compute.append({
                "name": name,
                "op": "sum",
                "expr": f"CASE WHEN {field.upper()}='{value}' THEN {points} ELSE 0 END",
                "where": {account_field: [counting_account_type]},
            })

    # 3. SCORE = BASE_POINTS + modifiers
    if base_points:
        expr = "BASE_POINTS"
        if modifier_names:
            expr += "".join(f" + {m}" for m in modifier_names)
        compute.append({
            "name": "SCORE",
            "op": "expr",
            "expr": expr,
        })

    # 4. VERDICT case on SCORE with original thresholds
    verdict_rules = []
    for rule in rules:
        verdict = rule.get("verdict", "")

        if rule.get("default"):
            verdict_rules.append({"else": verdict})
        elif "condition" in rule:
            # Parse "score >= X" -> {"when": ">= X", "then": verdict}
            cond = rule["condition"]
            # Match patterns like "score >= 12", "score >= 2", "score <= -5"
            match = re.search(r'score\s*([><=]+)\s*(-?\d+)', cond.lower())
            if match:
                op = match.group(1)
                threshold = match.group(2)
                verdict_rules.append({
                    "when": f"SCORE {op} {threshold}",
                    "then": verdict,
                })
            else:
                logger.warning(f"Could not parse score condition: {cond}")
        elif "conditions" in rule:
            # Fall back to count-based logic for rules with explicit conditions
            # This shouldn't happen for pure scoring policies but handles edge cases
            logger.warning(f"Scoring policy has count-based conditions, skipping: {rule}")

    compute.append({
        "name": "VERDICT",
        "op": "case",
        "source": "SCORE",
        "rules": verdict_rules,
    })

    return compute


def _build_compute_operations(
    rules: List[Dict[str, Any]],
    policy_type: str,
    account_field: str = "ACCOUNT_TYPE",
    counting_account_type: str = "Firsthand",
    scoring_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build compute operations from rules.

    For count_rule_based policies:
    - Creates count operations for each unique field+values combination
    - Builds VERDICT case operation with proper conditions

    For scoring policies with score-based conditions:
    - Routes to _build_scoring_compute for SUM-based operations
    - Preserves original score thresholds (no conversion to counts)

    Args:
        rules: List of verdict rules from PolicyYAML
        policy_type: "count_rule_based" or "scoring"
        account_field: Name of account type field
        counting_account_type: Account type that counts for verdicts
        scoring_config: Optional scoring configuration for scoring policies

    Returns:
        List of compute operations for Formula Seed
    """
    # Check if this is a scoring policy with score-based conditions
    # These should use SUM operations, not COUNT
    has_score_conditions = any(
        "condition" in r and "score" in r.get("condition", "").lower()
        for r in rules if not r.get("default")
    )

    if policy_type == "scoring" and has_score_conditions and scoring_config:
        # Route to dedicated scoring compute builder
        logger.info("Using SUM-based scoring operations (score conditions detected)")
        return _build_scoring_compute(
            rules=rules,
            scoring_config=scoring_config,
            account_field=account_field,
            counting_account_type=counting_account_type,
        )

    # Otherwise use count-based logic (for count_rule_based or scoring without score conditions)
    compute = []
    count_vars = {}  # Track count variable names to avoid duplicates

    if policy_type in ("count_rule_based", "scoring"):
        # Extract all unique field+values combinations and create count ops
        for rule in rules:
            if rule.get("default"):
                continue

            conditions = rule.get("conditions", [])
            for cond in conditions:
                field = cond.get("field", "")
                values = cond.get("values", [])

                if not field or not values:
                    continue

                # Generate unique count variable name
                key = (field.upper(), tuple(sorted(str(v).lower() for v in values)))
                if key in count_vars:
                    continue

                count_name = _generate_count_name(field, values)

                # Ensure uniqueness
                base_name = count_name
                counter = 1
                while count_name in [v for v in count_vars.values()]:
                    count_name = f"{base_name}_{counter}"
                    counter += 1

                count_vars[key] = count_name

                # Create count operation
                compute.append({
                    "name": count_name,
                    "op": "count",
                    "where": {
                        account_field: [counting_account_type],
                        field.upper(): [str(v) for v in values],
                    },
                })

        # Build VERDICT case operation
        verdict_rules = []
        for rule in rules:
            verdict = rule.get("verdict", "")

            if rule.get("default"):
                verdict_rules.append({"else": verdict})
                continue

            conditions = rule.get("conditions", [])
            logic = rule.get("logic", "ANY").upper()

            if not conditions:
                continue

            # Build condition expression
            cond_exprs = []
            for cond in conditions:
                field = cond.get("field", "")
                values = cond.get("values", [])
                min_count = cond.get("min_count", 1)

                if not field or not values:
                    continue

                # Skip ACCOUNT_TYPE-only conditions - they measure evidence quantity,
                # not content, and shouldn't alone determine verdicts
                if field.upper() == account_field.upper():
                    logger.debug(f"Skipping account-type-only condition: {field}={values}")
                    continue

                key = (field.upper(), tuple(sorted(str(v).lower() for v in values)))
                count_var = count_vars.get(key)
                if count_var:
                    cond_exprs.append(f"{count_var} >= {min_count}")

            if not cond_exprs:
                continue

            # Combine conditions based on logic
            if logic == "ALL":
                when_expr = " and ".join(cond_exprs)
            else:  # ANY
                # For OR rules, filter out conditions with threshold = 1 if there are
                # other stronger conditions. This prevents overly permissive rules like
                # "SEVERE >= 1 OR MILD >= 1" which triggers on any incident.
                strong_conds = [c for c in cond_exprs if not c.endswith(">= 1")]
                weak_conds = [c for c in cond_exprs if c.endswith(">= 1")]

                # If we have both strong and weak conditions, only keep strong ones
                # for non-default verdicts (keeps rules meaningful)
                if strong_conds and weak_conds:
                    logger.debug(
                        f"Filtering weak OR conditions for {verdict}: "
                        f"keeping {strong_conds}, dropping {weak_conds}"
                    )
                    cond_exprs = strong_conds

                when_expr = " or ".join(cond_exprs)

            verdict_rules.append({
                "when": when_expr,
                "then": verdict,
                "_verdict_type": "negative" if any(kw in verdict.lower() for kw in {"poor", "bad", "low", "negative", "terrible"}) else "positive" if any(kw in verdict.lower() for kw in {"good", "excellent", "great", "positive"}) else "neutral",
            })

        # Post-process verdict rules for consistency:
        # 1. Detect if default is an extreme verdict (should be middle/neutral)
        # 2. For 3-verdict policies, ensure middle verdict is the default
        # 3. Sort non-default rules appropriately
        if verdict_rules:
            # Separate default from non-default rules
            default_rules = [r for r in verdict_rules if "else" in r]
            non_default_rules = [r for r in verdict_rules if "else" not in r]

            # Get all verdict labels
            all_verdicts = [r.get("then") for r in non_default_rules]
            all_verdicts += [r.get("else") for r in default_rules]

            # Detect extreme verdicts (these shouldn't be defaults)
            extreme_positive = {"excellent", "highly unique", "great", "best", "recommended", "outstanding"}
            extreme_negative = {"generic", "poor", "terrible", "worst", "needs improvement", "not recommended"}

            # Check if current default is extreme
            current_default = default_rules[0].get("else", "") if default_rules else ""
            current_default_lower = current_default.lower()
            default_is_extreme = (
                any(kw in current_default_lower for kw in extreme_negative) or
                any(kw in current_default_lower for kw in extreme_positive)
            )

            # If default is extreme and there's a neutral verdict available, swap
            if default_is_extreme and len(all_verdicts) == 3:
                # Find the middle verdict (not extreme positive or negative)
                middle_verdict = None
                for v in all_verdicts:
                    v_lower = v.lower() if v else ""
                    is_extreme = (
                        any(kw in v_lower for kw in extreme_negative) or
                        any(kw in v_lower for kw in extreme_positive)
                    )
                    if not is_extreme:
                        middle_verdict = v
                        break

                if middle_verdict:
                    # Swap: make middle the default, move old default to a rule
                    logger.info(
                        f"Swapping default verdict: '{current_default}' → '{middle_verdict}' "
                        f"(extreme default detected)"
                    )
                    # Find the rule for middle verdict and remove it
                    new_non_default = []
                    for r in non_default_rules:
                        if r.get("then") == middle_verdict:
                            # Convert old default to this rule
                            new_non_default.append({
                                "when": r.get("when"),
                                "then": current_default,
                                "_verdict_type": r.get("_verdict_type", "neutral"),
                            })
                        else:
                            new_non_default.append(r)
                    non_default_rules = new_non_default
                    default_rules = [{"else": middle_verdict}]

            # Sort non-default: positive verdicts first, then negative, then neutral
            def rule_sort_key(r):
                vtype = r.get("_verdict_type", "neutral")
                if vtype == "positive":
                    return 0
                elif vtype == "negative":
                    return 2
                else:
                    return 1

            non_default_rules.sort(key=rule_sort_key)

            # Rebuild rules list with proper ordering
            verdict_rules = non_default_rules + default_rules

            compute.append({
                "name": "VERDICT",
                "op": "case",
                "rules": verdict_rules,
            })

    return compute


def compile_yaml_to_seed(
    yaml_data: Dict[str, Any],
    k: int = 200,
    validate: bool = True,
) -> Dict[str, Any]:
    """Compile PolicyYAML to Formula Seed deterministically.

    This is the main entry point for the compiler. It transforms
    a simple PolicyYAML structure into a full Formula Seed.

    Args:
        yaml_data: Parsed PolicyYAML dict
        k: Context size (number of reviews)
        validate: Whether to validate before compiling

    Returns:
        Formula Seed dict ready for Phase 2 execution

    Raises:
        PolicyYAMLValidationError: If validation fails (when validate=True)
    """
    if validate:
        validate_policy_yaml(yaml_data)

    # Extract configuration
    policy_type = yaml_data.get("policy_type", "count_rule_based")
    terms = yaml_data.get("terms", [])
    verdicts = yaml_data.get("verdicts", [])
    rules = yaml_data.get("rules", [])
    account_handling = yaml_data.get("account_handling")
    task_name = yaml_data.get("task_name", "policy evaluation")

    # Get account field configuration
    if account_handling:
        account_field = account_handling.get("field_name", "ACCOUNT_TYPE")
        # Find which account type counts for verdict
        counting_types = [
            t["type"] for t in account_handling.get("types", [])
            if t.get("counts_for_verdict", True)
        ]
        counting_account_type = counting_types[0] if counting_types else "Firsthand"
    else:
        account_field = "ACCOUNT_TYPE"
        counting_account_type = "Firsthand"

    # Build extract section
    extract_fields = _build_extract_fields(terms, account_handling)
    outcome_field, none_values = _determine_outcome_field(terms)

    # Get scoring config for scoring policies
    scoring_config = yaml_data.get("scoring", {})

    # Build compute section
    compute_ops = _build_compute_operations(
        rules=rules,
        policy_type=policy_type,
        account_field=account_field,
        counting_account_type=counting_account_type,
        scoring_config=scoring_config,
    )

    # Build extraction guidelines from terms and rules
    guidelines = _build_extraction_guidelines(terms, rules, verdicts)

    # Build output list
    output_fields = ["VERDICT"]
    # Add computed variables to output for transparency
    for op in compute_ops:
        op_type = op.get("op")
        op_name = op.get("name", "")
        if op_type == "count":
            output_fields.append(op_name)
        elif op_type in ("sum", "expr") and op_name:
            # Include SCORE, BASE_POINTS, modifiers for scoring policies
            output_fields.append(op_name)

    # Assemble Formula Seed
    seed = {
        "task_name": task_name,
        "extraction_guidelines": guidelines,
        "extract": {
            "fields": extract_fields,
            "outcome_field": outcome_field,
            "none_values": none_values,
        },
        "compute": compute_ops,
        "output": output_fields,
        "_compiled_from": "PolicyYAML",
        "_compiler_version": "1.0",
    }

    # Preserve recency_rules for V3 policies (used by phase2 for weighting)
    if scoring_config.get("recency_rules"):
        seed["scoring"] = {
            "recency_rules": scoring_config["recency_rules"],
        }

    return seed


def _build_extraction_guidelines(
    terms: List[Dict[str, Any]],
    rules: List[Dict[str, Any]],
    verdicts: List[str],
) -> str:
    """Build extraction guidelines from PolicyYAML components.

    Args:
        terms: Term definitions
        rules: Verdict rules
        verdicts: List of possible verdicts

    Returns:
        Multi-line guidelines string
    """
    lines = []

    # What to look for
    term_names = [t.get("name", "") for t in terms if t.get("name")]
    if term_names:
        lines.append(f"WHAT TO EXTRACT: Look for information about {', '.join(term_names)}.")

    # Classification guidance
    lines.append("\nCLASSIFICATION:")
    for term in terms:
        name = term.get("name", "")
        values = term.get("values", [])
        if name and values:
            # Convert values to strings (YAML may parse true/false as bools)
            values_str = [str(v) for v in values]
            lines.append(f"  {name}: Classify as one of [{', '.join(values_str)}]")

    # Account types
    lines.append("\nACCOUNT TYPES:")
    lines.append("  Firsthand: Direct personal experience by the reviewer")
    lines.append("  Secondhand: Information heard from others")
    lines.append("  General: General statements without personal experience")
    lines.append("  Only Firsthand accounts contribute to verdict counts.")

    # Evidence requirements
    lines.append("\nEVIDENCE REQUIREMENTS:")
    lines.append("  - You MUST provide a supporting_quote from the review text")
    lines.append("  - If no relevant evidence exists, set is_relevant: false")
    lines.append("  - Quotes must be verbatim from the source text")

    return "\n".join(lines)


def get_policy_yaml_schema() -> str:
    """Get the PolicyYAML schema documentation.

    Returns:
        YAML schema documentation string
    """
    return '''
PolicyYAML Schema
=================

policy_type: count_rule_based  # or: scoring

task_name: "Brief description of the evaluation task"

terms:
  - name: <FIELD_NAME>           # e.g., PRICE_PERCEPTION, QUALITY_LEVEL
    values: [<value1>, <value2>, ...]  # enum values to extract
    descriptions:                # optional: descriptions for each value
      <value1>: "Description"
      <value2>: "Description"

account_handling:               # optional: customize account types
  field_name: ACCOUNT_TYPE
  types:
    - type: Firsthand
      description: "Direct personal experience"
      counts_for_verdict: true
    - type: Secondhand
      description: "Heard from others"
      counts_for_verdict: false

verdicts: [<verdict1>, <verdict2>, <verdict3>]  # possible verdicts

rules:
  - verdict: <verdict_label>
    logic: ANY  # or: ALL
    conditions:
      - field: <FIELD_NAME>
        values: [<value1>, <value2>]  # which values to count
        min_count: <N>  # threshold

  - verdict: <default_verdict>
    default: true  # exactly one rule must be the default
'''
