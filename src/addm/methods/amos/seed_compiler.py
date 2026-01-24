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
        conditions = rule.get("conditions", [])
        if not conditions and not rule.get("default"):
            errors.append(f"Rule {i} ('{verdict}') has no conditions and is not default")

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

        # Build values dict (convert to strings for YAML bool handling)
        values_dict = {}
        for v in values:
            v_str = str(v)
            desc = descriptions.get(v, descriptions.get(v_str, f"Indicates {v_str}"))
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
    none_indicators = {"none", "n/a", "neutral", "no ", "absent", "not "}

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


def _build_compute_operations(
    rules: List[Dict[str, Any]],
    policy_type: str,
    account_field: str = "ACCOUNT_TYPE",
    counting_account_type: str = "Firsthand",
) -> List[Dict[str, Any]]:
    """Build compute operations from rules.

    For count_rule_based policies:
    - Creates count operations for each unique field+values combination
    - Builds VERDICT case operation with proper conditions

    For scoring policies:
    - Creates sum operations for point calculations
    - Builds SCORE computation and VERDICT thresholds

    Args:
        rules: List of verdict rules from PolicyYAML
        policy_type: "count_rule_based" or "scoring"
        account_field: Name of account type field
        counting_account_type: Account type that counts for verdicts

    Returns:
        List of compute operations for Formula Seed
    """
    compute = []
    count_vars = {}  # Track count variable names to avoid duplicates

    if policy_type == "scoring":
        # Build scoring operations (TODO: implement full scoring support)
        logger.warning("Scoring policy type not fully implemented, falling back to count-based")
        policy_type = "count_rule_based"

    if policy_type == "count_rule_based":
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
                when_expr = " or ".join(cond_exprs)

            verdict_rules.append({
                "when": when_expr,
                "then": verdict,
            })

        if verdict_rules:
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

    # Build compute section
    compute_ops = _build_compute_operations(
        rules=rules,
        policy_type=policy_type,
        account_field=account_field,
        counting_account_type=counting_account_type,
    )

    # Build extraction guidelines from terms and rules
    guidelines = _build_extraction_guidelines(terms, rules, verdicts)

    # Build output list
    output_fields = ["VERDICT"]
    # Add count variables to output for transparency
    for op in compute_ops:
        if op.get("op") == "count":
            output_fields.append(op.get("name", ""))

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
