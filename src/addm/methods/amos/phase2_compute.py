"""Phase 2 Compute Operations.

Mixin class providing compute operations for FormulaSeedInterpreter.
Extracted from phase2.py for clarity.

Operations: count, sum, case, expr, lookup
"""

import logging
import re
from typing import Any, Dict, List

from addm.methods.amos.phase2_helpers import (
    get_field_value,
    fuzzy_enum_match,
    normalize_actual_value,
    matches_condition,
)

logger = logging.getLogger(__name__)


class ComputeOperationsMixin:
    """Mixin providing compute operations for Formula Seed interpretation.

    Expects the following attributes on self:
    - seed: Dict[str, Any] - Formula Seed
    - _extractions: List[Dict] - Extracted data
    - _namespace: Dict[str, Any] - Computed values
    - _executor: SafeExpressionExecutor - For boolean expressions
    - _get_recency_weight(age_years) - V3 weight function
    """

    def _compute_count(self, op_def: Dict[str, Any]) -> int:
        """Compute count of extractions matching condition."""
        where = op_def.get("where") or op_def.get("condition", {})
        if not where:
            return len(self._extractions)
        return sum(1 for e in self._extractions if matches_condition(e, where, self.seed))

    def _compute_sum(self, op_def: Dict[str, Any]) -> float:
        """Compute sum with optional V3 recency weighting."""
        expr = op_def.get("expr", "1")
        name = op_def.get("name", "SUM")
        where = op_def.get("where") or op_def.get("condition", {})
        is_sql_case = expr.strip().upper().startswith("CASE")
        has_recency = bool(self.seed.get("scoring", {}).get("recency_rules", {}).get("rules"))

        total = 0.0
        for extraction in self._extractions:
            if where and not matches_condition(extraction, where, self.seed):
                continue

            # Evaluate expression
            if is_sql_case:
                value = self._eval_sql_case_expr(expr, extraction)
            else:
                try:
                    safe_ns = {
                        **{k: v for k, v in extraction.items() if not k.startswith("_")},
                        **self._namespace,
                    }
                    value = float(eval(expr, {"__builtins__": {}}, safe_ns))
                except Exception as e:
                    logger.warning(f"[SUM] {name}: eval failed: {e}")
                    continue

            # Apply recency weighting for V3
            if has_recency:
                age = extraction.get("AGE_YEARS", 0.0)
                weight = self._get_recency_weight(age)
                value *= weight

            total += value

        return total

    def _eval_sql_case_expr(self, expr: str, extraction: Dict[str, Any]) -> float:
        """Evaluate SQL-style CASE expression against an extraction."""
        case_pattern = r"CASE\s+(.+?)\s+END"
        case_blocks = re.findall(case_pattern, expr, re.IGNORECASE | re.DOTALL)

        if not case_blocks:
            return 0.0

        total = 0.0
        for case_body in case_blocks:
            total += self._eval_single_case(case_body, extraction)
        return total

    def _eval_single_case(self, case_body: str, extraction: Dict[str, Any]) -> float:
        """Evaluate single CASE body (between CASE and END)."""
        # WHEN field = 'value' THEN number
        when_pattern = r"WHEN\s+(\w+)\s*=\s*['\"]([^'\"]+)['\"]\s+THEN\s+(-?\d+(?:\.\d+)?)"
        matches = re.findall(when_pattern, case_body, re.IGNORECASE)

        # Also unquoted values
        when_unquoted = r"WHEN\s+(\w+)\s*=\s*(\w+)\s+THEN\s+(-?\d+(?:\.\d+)?)"
        matches.extend(re.findall(when_unquoted, case_body, re.IGNORECASE))

        # IN clauses
        in_pattern = r"WHEN\s+(\w+)\s+IN\s*\(([^)]+)\)\s+THEN\s+(-?\d+(?:\.\d+)?)"
        in_matches = re.findall(in_pattern, case_body, re.IGNORECASE)

        for field, value, then_value in matches:
            actual = get_field_value(extraction, field)
            actual = normalize_actual_value(self.seed, field, actual)
            if actual is None:
                actual = ""
            if fuzzy_enum_match(str(actual), value):
                return float(then_value)

        for field, values_str, then_value in in_matches:
            actual = get_field_value(extraction, field)
            actual = normalize_actual_value(self.seed, field, actual)
            if actual is None:
                actual = ""
            values = [v.strip().strip("'\"") for v in values_str.split(",")]
            if str(actual).lower().strip() in [v.lower().strip() for v in values]:
                return float(then_value)

        # ELSE clause
        else_match = re.search(r"ELSE\s+(-?\d+(?:\.\d+)?)", case_body, re.IGNORECASE)
        if else_match:
            return float(else_match.group(1))

        return 0.0

    def _compute_expr(self, op_def: Dict[str, Any]) -> Any:
        """Evaluate mathematical expression with embedded CASE support."""
        expr = op_def.get("expr", "0")

        # Handle embedded CASE blocks
        case_pattern = r"\(?\s*CASE\s+.+?\s+END\s*\)?"
        case_blocks = re.findall(case_pattern, expr, re.IGNORECASE | re.DOTALL)

        if case_blocks:
            for case_block in case_blocks:
                case_sum = sum(self._eval_sql_case_expr(case_block, e) for e in self._extractions)
                expr = expr.replace(case_block, str(case_sum), 1)

        try:
            safe_builtins = {"min": min, "max": max, "abs": abs, "round": round, "sum": sum, "len": len}
            return eval(expr, {"__builtins__": safe_builtins}, self._namespace)
        except Exception:
            return 0

    def _compute_lookup(self, op_def: Dict[str, Any], business: Dict[str, Any]) -> Any:
        """Lookup value from restaurant attributes."""
        source = op_def.get("source", "")
        table = op_def.get("table", {})
        default = op_def.get("default", 1.0)

        if source.startswith("context."):
            source_value = business.get(source[8:], "")
        else:
            source_value = self._namespace.get(source, "")

        if isinstance(source_value, list):
            source_value = " ".join(source_value)

        source_lower = str(source_value).lower()
        for pattern, value in table.items():
            if pattern.lower() in source_lower:
                return value

        return default

    def _compute_case(self, op_def: Dict[str, Any]) -> Any:
        """Apply case rules with threshold/expression support."""
        source = op_def.get("source", "")
        rules = op_def.get("rules", [])
        source_value = self._namespace.get(source, 0)

        for rule in rules:
            if "else" in rule:
                return rule["else"]

            when = rule.get("when", "")
            then = rule.get("then", "")

            # Try full expression evaluation
            try:
                context = {
                    **self._namespace,
                    source: source_value,
                    source.lower(): source_value,
                    source.upper(): source_value,
                }
                result = self._executor.execute_bool(when, context, default=None)
                if result is True:
                    return then
                elif result is not None:
                    continue
            except Exception:
                pass

            # Fallback: threshold comparison
            match = re.match(r"([<>=!]+)\s*(\d+(?:\.\d+)?)", str(when))
            if match:
                op, threshold = match.groups()
                threshold = float(threshold)
                try:
                    num_val = float(source_value) if source_value is not None else 0
                    if self._eval_threshold(op, num_val, threshold):
                        return then
                except (ValueError, TypeError):
                    pass
            elif str(source_value) == str(when):
                return then

        return op_def.get("default_verdict") or op_def.get("default")

    def _eval_threshold(self, op: str, value: float, threshold: float) -> bool:
        """Evaluate threshold comparison."""
        if op == "<":
            return value < threshold
        elif op == "<=":
            return value <= threshold
        elif op == ">":
            return value > threshold
        elif op == ">=":
            return value >= threshold
        elif op == "==":
            return value == threshold
        elif op == "!=":
            return value != threshold
        return False

    def _apply_case_to_extraction(
        self, extraction: Dict[str, Any], source: str, rules: List[Dict[str, Any]]
    ) -> Any:
        """Apply case rules to a single extraction."""
        source_value = get_field_value(extraction, source)
        if source_value is None:
            source_value = "none"

        for rule in rules:
            if "else" in rule:
                return rule["else"]

            when = rule.get("when", "")
            then = rule.get("then", "")

            match = re.match(r"([<>=!]+)\s*(\d+(?:\.\d+)?)", str(when))
            if match:
                op, threshold = match.groups()
                try:
                    num_val = float(source_value) if source_value is not None else 0
                    if self._eval_threshold(op, num_val, float(threshold)):
                        return then
                except (ValueError, TypeError):
                    pass
            elif str(source_value).lower() == str(when).lower():
                return then

        return 0

    def _is_extraction_field(self, field_name: str) -> bool:
        """Check if a field name exists in extraction fields."""
        fields = self.seed.get("extract", {}).get("fields", [])
        return any(f.get("name") == field_name for f in fields)

    def _execute_compute(self, business: Dict[str, Any]) -> None:
        """Execute all compute operations in order."""
        for op_def in self.seed.get("compute", []):
            name = op_def.get("name", "")
            op = op_def.get("op") or op_def.get("operation", "")

            if op == "count":
                self._namespace[name] = self._compute_count(op_def)
            elif op == "sum":
                self._namespace[name] = self._compute_sum(op_def)
            elif op == "expr":
                self._namespace[name] = self._compute_expr(op_def)
            elif op == "lookup":
                self._namespace[name] = self._compute_lookup(op_def, business)
            elif op == "case":
                source = op_def.get("source", "")
                rules = op_def.get("rules", [])

                if self._is_extraction_field(source):
                    # Per-extraction case, sum results
                    total = 0
                    for extraction in self._extractions:
                        result = self._apply_case_to_extraction(extraction, source, rules)
                        if isinstance(result, (int, float)):
                            total += result
                    self._namespace[name] = total
                else:
                    self._namespace[name] = self._compute_case(op_def)
