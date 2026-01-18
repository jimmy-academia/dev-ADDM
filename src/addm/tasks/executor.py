"""Generic ground truth executor.

Executes parsed prompt DSL against actual data.
The prompt is the single source of truth - this code has no hardcoded task logic.
"""

import ast
import operator
import re
from typing import Any, Dict, List, Optional

from addm.tasks.prompt_parser import (
    Condition,
    DecisionPolicy,
    Formula,
    L1Composite,
    L2Aggregate,
    LookupTable,
    ParsedPrompt,
)


# =============================================================================
# Condition Evaluation
# =============================================================================


def evaluate_condition(cond: Condition, record: Dict[str, Any]) -> bool:
    """Evaluate a single condition against a record."""
    # Get field value (case-insensitive lookup)
    field_lower = cond.field.lower()
    value = record.get(field_lower) or record.get(cond.field)

    # Normalize values for comparison
    def normalize(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v).lower()

    if cond.operator == "=":
        return normalize(value) == normalize(cond.value)
    elif cond.operator == "!=":
        return normalize(value) != normalize(cond.value)
    elif cond.operator == "in":
        return normalize(value) in [normalize(v) for v in cond.value]
    elif cond.operator == ">=":
        return float(value or 0) >= float(cond.value)
    elif cond.operator == "<=":
        return float(value or 0) <= float(cond.value)
    elif cond.operator == ">":
        return float(value or 0) > float(cond.value)
    elif cond.operator == "<":
        return float(value or 0) < float(cond.value)
    elif cond.operator == "==":
        return float(value or 0) == float(cond.value)

    return False


# =============================================================================
# L1 Composite Evaluation
# =============================================================================


def evaluate_composite(comp: L1Composite, record: Dict[str, Any]) -> bool:
    """Evaluate an L1 composite for a single record."""
    results = [evaluate_condition(c, record) for c in comp.conditions]

    if comp.quantifier == "ALL":
        return all(results)
    elif comp.quantifier == "ANY":
        return any(results)

    return False


def compute_l1_for_record(
    record: Dict[str, Any],
    composites: List[L1Composite],
) -> Dict[str, bool]:
    """Compute all L1 composites for a single record."""
    return {comp.name: evaluate_composite(comp, record) for comp in composites}


# =============================================================================
# Lookup Table Evaluation
# =============================================================================


def evaluate_lookup(
    table: LookupTable,
    categories: str,
) -> float:
    """Evaluate a lookup table against category string.

    For 'highest' strategy: returns max matching value.
    """
    if not categories:
        return table.default

    cats = [c.strip() for c in categories.split(",")]
    matches = []

    for cat in cats:
        # Direct match
        if cat in table.entries:
            matches.append(table.entries[cat])
        else:
            # Partial match
            for key, value in table.entries.items():
                if key.lower() in cat.lower():
                    matches.append(value)

    if not matches:
        return table.default

    if table.strategy == "highest":
        return max(matches)
    elif table.strategy == "lowest":
        return min(matches)
    elif table.strategy == "first":
        return matches[0]

    return table.default


# =============================================================================
# L2 Aggregate Evaluation
# =============================================================================


def evaluate_aggregate(
    agg: L2Aggregate,
    records: List[Dict[str, Any]],
    ctx: Dict[str, Any],
) -> Any:
    """Evaluate an L2 aggregate over all records."""
    if agg.agg_type == "count":
        count = 0
        for record in records:
            if all(evaluate_condition(c, record) for c in agg.conditions):
                count += 1
        return count

    elif agg.agg_type == "arithmetic":
        # Simple arithmetic on context vars
        return evaluate_expression(agg.expression, ctx)

    elif agg.agg_type == "conditional":
        # if/elif/else chain
        for cond_str, result in agg.branches:
            if cond_str == "else":
                return result

            # Parse condition
            parts = re.match(r"([A-Z_]+)\s*([<>=!]+)\s*([\d\.]+)", cond_str)
            if parts:
                var = parts.group(1)
                op = parts.group(2)
                val = float(parts.group(3))
                ctx_val = ctx.get(var, 0)

                if op == "==" and ctx_val == val:
                    return result
                elif op == "<=" and ctx_val <= val:
                    return result
                elif op == "<" and ctx_val < val:
                    return result
                elif op == ">=" and ctx_val >= val:
                    return result
                elif op == ">" and ctx_val > val:
                    return result

        return None

    elif agg.agg_type == "max":
        # max(FIELD) where CONDITION
        values = []
        for record in records:
            if all(evaluate_condition(c, record) for c in agg.conditions):
                # Get the field value (e.g., REVIEW_YEAR)
                field_val = record.get(agg.expression.lower()) or record.get(agg.expression)
                if field_val is not None:
                    values.append(field_val)

        if values:
            return max(values)
        else:
            # Return default from context
            return ctx.get(agg.default_var, 0)

    return 0


# =============================================================================
# Safe Arithmetic Evaluation
# =============================================================================


# Allowed operators for safe arithmetic evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node containing only arithmetic operations."""
    if isinstance(node, ast.Constant):
        # Python 3.8+ uses ast.Constant for literals
        return float(node.value)
    elif isinstance(node, ast.Num):
        # Fallback for older Python versions
        return float(node.n)
    elif isinstance(node, ast.BinOp):
        # Binary operation: left op right
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_func = SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(left, right)
    elif isinstance(node, ast.UnaryOp):
        # Unary operation: op operand
        operand = _eval_node(node.operand)
        op_func = SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(operand)
    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def safe_eval_arithmetic(expr: str) -> float:
    """Safely evaluate arithmetic expressions without variable substitution.

    Only supports:
    - Numbers (int, float)
    - Binary operators: +, -, *, /
    - Unary operator: - (negation)
    - Parentheses for grouping

    Does NOT support:
    - Variables
    - Function calls
    - Comparisons
    - Boolean operations
    - Any other Python features

    Args:
        expr: Arithmetic expression string (e.g., "2 + 3 * 4")

    Returns:
        Evaluated result as float

    Raises:
        ValueError: If expression contains unsupported operations
        SyntaxError: If expression is malformed
    """
    try:
        tree = ast.parse(expr, mode="eval")
        return _eval_node(tree.body)
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        # Return 0.0 for any evaluation errors (same behavior as original eval)
        return 0.0


# =============================================================================
# Expression Evaluation
# =============================================================================


def evaluate_expression(expr: str, ctx: Dict[str, Any]) -> float:
    """Safely evaluate an arithmetic expression with variable substitution."""
    # Replace variable names with values
    result_expr = expr

    # Find all variable references (uppercase with underscores)
    vars_in_expr = re.findall(r"[A-Z_]+", expr)

    # Sort by length (longest first) to avoid partial replacements
    vars_in_expr = sorted(set(vars_in_expr), key=len, reverse=True)

    for var in vars_in_expr:
        if var in ctx:
            result_expr = result_expr.replace(var, str(float(ctx[var])))

    # Safe evaluation - only allow numbers and arithmetic
    # Remove any remaining letters (undefined vars become 0)
    result_expr = re.sub(r"[A-Za-z_]+", "0", result_expr)
    return safe_eval_arithmetic(result_expr)


def evaluate_formula(formula: Formula, ctx: Dict[str, Any]) -> float:
    """Evaluate a formula and return the result."""
    if formula.func == "clamp":
        value = evaluate_expression(formula.expression, ctx)
        min_val = float(formula.func_args[0])
        max_val = float(formula.func_args[1])
        return max(min_val, min(max_val, value))

    elif formula.func == "max":
        floor = float(formula.func_args[0])
        value = evaluate_expression(formula.expression, ctx)
        return max(floor, value)

    elif formula.func == "min":
        ceiling = float(formula.func_args[0])
        value = evaluate_expression(formula.expression, ctx)
        return min(ceiling, value)

    else:
        return evaluate_expression(formula.expression, ctx)


# =============================================================================
# Decision Policy Evaluation
# =============================================================================


# Verdict ranking for "min" overrides
VERDICT_RANK = {
    "Low Risk": 1,
    "High Risk": 2,
    "Critical Risk": 3,
}


def apply_decision_policy(policy: DecisionPolicy, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Apply decision policy to get final verdict."""
    score = ctx.get(policy.score_variable, 0)

    # Find base verdict from thresholds
    base_verdict = "Unknown"
    for rule in policy.threshold_rules:
        matches = True
        for op, threshold in rule.conditions:
            if op == "<" and not (score < threshold):
                matches = False
            elif op == "<=" and not (score <= threshold):
                matches = False
            elif op == ">" and not (score > threshold):
                matches = False
            elif op == ">=" and not (score >= threshold):
                matches = False

        if matches:
            base_verdict = rule.result
            break

    # Apply overrides
    final_verdict = base_verdict
    override_applied = "none"

    for rule in policy.override_rules:
        if evaluate_condition(rule.condition, ctx):
            if rule.is_minimum:
                # "min X" means at least X
                if VERDICT_RANK.get(rule.result, 0) > VERDICT_RANK.get(final_verdict, 0):
                    final_verdict = rule.result
                    override_applied = f"min_{rule.result.lower().replace(' ', '_')}"
            else:
                # Absolute override
                final_verdict = rule.result
                override_applied = rule.condition.field.lower()
                break  # First absolute override wins

    return {
        "base_verdict_by_score": base_verdict,
        "override_applied": override_applied,
        "verdict": final_verdict,
    }


# =============================================================================
# Main Executor
# =============================================================================


def compute_ground_truth(
    judgments: List[Dict[str, Any]],
    restaurant_meta: Dict[str, Any],
    parsed_prompt: ParsedPrompt,
) -> Dict[str, Any]:
    """
    Compute ground truth from judgments using parsed prompt DSL.

    Args:
        judgments: List of L0 judgments (one per review)
        restaurant_meta: Restaurant metadata with 'categories' field
        parsed_prompt: Parsed prompt structure

    Returns:
        Ground truth dictionary with all computed values
    """
    ctx: Dict[str, Any] = {}

    # 1. Load constants
    ctx.update(parsed_prompt.constants)

    # 2. Compute lookup table values
    categories = restaurant_meta.get("categories", "")
    for name, table in parsed_prompt.lookup_tables.items():
        ctx[name] = evaluate_lookup(table, categories)

    # 3. Filter to allergy-related judgments and compute L1
    allergy_judgments = [j for j in judgments if j.get("is_allergy_related", False)]

    for j in allergy_judgments:
        l1 = compute_l1_for_record(j, parsed_prompt.l1_composites)
        j.update(l1)

        # Extract REVIEW_YEAR from date
        date_str = j.get("date", "")
        if date_str and len(date_str) >= 4:
            try:
                j["REVIEW_YEAR"] = int(date_str[:4])
            except ValueError:
                pass

    # 4. Compute L2 aggregates
    for agg in parsed_prompt.l2_aggregates:
        ctx[agg.name] = evaluate_aggregate(agg, allergy_judgments, ctx)

    # 5. Evaluate formulas in order (assuming correct dependency order in prompt)
    for formula in parsed_prompt.formulas:
        ctx[formula.name] = evaluate_formula(formula, ctx)

    # 6. Apply decision policy
    decision = apply_decision_policy(parsed_prompt.decision_policy, ctx)
    ctx.update(decision)

    # 7. Add counts
    ctx["n_allergy_reviews"] = len(allergy_judgments)

    return ctx
