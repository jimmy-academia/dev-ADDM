"""Phase 2: Formula Seed Interpreter.

Executes the Formula Seed against restaurant data:
1. Filter reviews by keywords (Python)
2. Extract signals via LLM (parallel)
3. Compute aggregations (Python)
4. Execute computation DAG (Python)
5. Return output values
"""

import json
import re
from typing import Any, Dict, List, Optional

from addm.llm import LLMService
from addm.utils.async_utils import gather_with_concurrency


def _build_extraction_prompt(fields: List[Dict[str, Any]], review_text: str, review_id: str) -> str:
    """Build extraction prompt from Formula Seed field definitions."""
    field_defs = []
    for field in fields:
        name = field["name"]
        ftype = field.get("type", "enum")
        values = field.get("values", {})

        if ftype == "enum" and values:
            value_list = ", ".join(values.keys())
            field_defs.append(f"{name.upper()} // one of {{{value_list}}}")
            for value, description in values.items():
                field_defs.append(f"  - {value}: {description}")
            field_defs.append("")
        elif ftype == "int":
            field_defs.append(f"{name.upper()} // integer value")
            if values:
                for value, description in values.items():
                    field_defs.append(f"  - {value}: {description}")
            field_defs.append("")
        elif ftype == "float":
            field_defs.append(f"{name.upper()} // numeric value (0.0-1.0 or as specified)")
            field_defs.append("")
        elif ftype == "bool":
            field_defs.append(f"{name.upper()} // true or false")
            field_defs.append("")

    field_definitions = "\n".join(field_defs)

    # Build output format
    output_fields = ", ".join(f'"{f["name"]}": "<value>"' for f in fields)

    prompt = f"""Extract the following fields from this review.

FIELD DEFINITIONS:
{field_definitions}

REVIEW TEXT:
{review_text}

Output JSON only. If the review is not relevant (no useful content for any field), output:
{{"is_relevant": false}}

Otherwise output:
{{
  "review_id": "{review_id}",
  "is_relevant": true,
  {output_fields}
}}"""

    return prompt


def _parse_extraction_response(response: str) -> Dict[str, Any]:
    """Parse LLM extraction response to JSON."""
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

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        # Return non-relevant for parse failures
        return {"is_relevant": False, "_parse_error": str(e)}


class FormulaSeedInterpreter:
    """Executes Formula Seed against restaurant data."""

    def __init__(
        self,
        seed: Dict[str, Any],
        llm: LLMService,
        max_concurrent: int = 32,
    ):
        """Initialize interpreter.

        Args:
            seed: Formula Seed specification
            llm: LLM service for extraction calls
            max_concurrent: Max concurrent LLM calls for extraction
        """
        self.seed = seed
        self.llm = llm
        self.max_concurrent = max_concurrent

        # Usage tracking
        self._usage_records: List[Dict[str, Any]] = []
        self._extractions: List[Dict[str, Any]] = []

        # Computed values namespace
        self._namespace: Dict[str, Any] = {}

    def _filter_reviews(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter reviews by keywords (Python, no LLM).

        Args:
            reviews: List of review dicts with 'text' field

        Returns:
            List of reviews that contain at least one keyword
        """
        keywords = self.seed.get("filter", {}).get("keywords", [])
        if not keywords:
            return reviews

        # Build regex pattern for keywords (case-insensitive, word boundaries)
        patterns = [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in keywords]

        filtered = []
        for review in reviews:
            text = review.get("text", "")
            # Check if any keyword matches
            for pattern in patterns:
                if pattern.search(text):
                    filtered.append(review)
                    break

        return filtered

    async def _extract_single_review(
        self,
        review: Dict[str, Any],
        fields: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract signals from a single review via LLM.

        Args:
            review: Review dict with 'text' and 'review_id' fields
            fields: Field definitions from Formula Seed

        Returns:
            Extraction result dict
        """
        review_id = review.get("review_id", "unknown")
        review_text = review.get("text", "")

        # Filter out temporal fields that should be populated from metadata
        temporal_field_names = {"REVIEW_DATE", "AGE_YEARS"}
        llm_fields = [f for f in fields if f.get("name") not in temporal_field_names]

        prompt = _build_extraction_prompt(llm_fields, review_text, review_id)
        messages = [{"role": "user", "content": prompt}]

        response, usage = await self.llm.call_async_with_usage(
            messages,
            context={"phase": "phase2_extract", "review_id": review_id},
        )

        self._usage_records.append(usage)

        extraction = _parse_extraction_response(response)
        extraction["review_id"] = review_id

        # Populate temporal fields from review metadata
        if any(f.get("name") in temporal_field_names for f in fields):
            from datetime import datetime

            review_date = review.get("date")  # Expecting ISO format or datetime object
            if review_date:
                # Convert to datetime if string
                if isinstance(review_date, str):
                    try:
                        review_date_dt = datetime.fromisoformat(review_date.replace("Z", "+00:00"))
                    except ValueError:
                        review_date_dt = None
                else:
                    review_date_dt = review_date

                # Populate REVIEW_DATE if requested
                if any(f.get("name") == "REVIEW_DATE" for f in fields) and review_date_dt:
                    extraction["REVIEW_DATE"] = review_date_dt.strftime("%Y-%m-%d")

                # Populate AGE_YEARS if requested
                if any(f.get("name") == "AGE_YEARS" for f in fields) and review_date_dt:
                    age_days = (datetime.now() - review_date_dt).days
                    extraction["AGE_YEARS"] = age_days / 365.25

        return extraction

    async def _extract_signals(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract signals from all reviews via parallel LLM calls.

        Args:
            reviews: List of filtered reviews

        Returns:
            List of extraction results (only relevant ones)
        """
        fields = self.seed.get("extract", {}).get("fields", [])
        if not fields or not reviews:
            return []

        # Run extractions in parallel with concurrency limit
        tasks = [self._extract_single_review(r, fields) for r in reviews]
        results = await gather_with_concurrency(self.max_concurrent, tasks)

        # Filter to only relevant extractions
        relevant = [r for r in results if r.get("is_relevant", False)]
        self._extractions = relevant

        return relevant

    def _matches_condition(self, extraction: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Check if an extraction matches a where condition.

        Args:
            extraction: Single extraction result
            condition: Dict of {field: value} to match

        Returns:
            True if all conditions match
        """
        for field, expected in condition.items():
            actual = extraction.get(field)
            if isinstance(expected, list):
                # Match any in list
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False
        return True

    def _compute_count(self, op_def: Dict[str, Any]) -> int:
        """Compute count aggregation.

        Args:
            op_def: Operation definition with 'where' condition

        Returns:
            Count of matching extractions
        """
        where = op_def.get("where", {})
        if not where:
            return len(self._extractions)

        return sum(1 for e in self._extractions if self._matches_condition(e, where))

    def _compute_sum(self, op_def: Dict[str, Any]) -> float:
        """Compute sum aggregation.

        Args:
            op_def: Operation definition with 'expr' and optional 'where'

        Returns:
            Sum of expression values across matching extractions
        """
        expr = op_def.get("expr", "1")
        where = op_def.get("where", {})

        total = 0.0
        for extraction in self._extractions:
            if where and not self._matches_condition(extraction, where):
                continue

            # Evaluate expression in extraction context
            try:
                # Create safe namespace with extraction values
                safe_namespace = {
                    **{k: v for k, v in extraction.items() if not k.startswith("_")},
                    **self._namespace,
                }
                value = eval(expr, {"__builtins__": {}}, safe_namespace)
                total += float(value)
            except Exception:
                pass  # Skip invalid expressions

        return total

    def _apply_case_to_extraction(
        self, extraction: Dict[str, Any], source: str, rules: List[Dict[str, Any]]
    ) -> Any:
        """Apply case rules to a single extraction.

        Args:
            extraction: Single extraction result
            source: Field name in extraction to use as source
            rules: List of case rules

        Returns:
            Result from first matching rule, or None
        """
        source_value = extraction.get(source, "none")

        for rule in rules:
            if "else" in rule:
                return rule["else"]

            when = rule.get("when", "")
            then = rule.get("then", "")

            # Try numeric threshold comparison
            match = re.match(r"([<>=!]+)\s*(\d+(?:\.\d+)?)", str(when))
            if match:
                op, threshold = match.groups()
                threshold = float(threshold)
                try:
                    numeric_value = float(source_value) if source_value is not None else 0
                    if op == "<" and numeric_value < threshold:
                        return then
                    elif op == "<=" and numeric_value <= threshold:
                        return then
                    elif op == ">" and numeric_value > threshold:
                        return then
                    elif op == ">=" and numeric_value >= threshold:
                        return then
                    elif op == "==" and numeric_value == threshold:
                        return then
                    elif op == "!=" and numeric_value != threshold:
                        return then
                except (ValueError, TypeError):
                    pass
            else:
                # Direct value matching
                if str(source_value) == str(when):
                    return then

        return 0  # Default to 0 for scoring purposes

    def _compute_expr(self, op_def: Dict[str, Any]) -> Any:
        """Evaluate mathematical expression.

        Args:
            op_def: Operation definition with 'expr'

        Returns:
            Expression result
        """
        expr = op_def.get("expr", "0")

        # Safe evaluation with namespace
        try:
            # Allow basic math functions
            safe_builtins = {
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sum": sum,
                "len": len,
            }
            return eval(expr, {"__builtins__": safe_builtins}, self._namespace)
        except Exception as e:
            return 0

    def _compute_lookup(self, op_def: Dict[str, Any], business: Dict[str, Any]) -> Any:
        """Lookup value from restaurant attributes.

        Args:
            op_def: Operation definition with 'source' and 'table'
            business: Restaurant business info

        Returns:
            Looked up value or default
        """
        source = op_def.get("source", "")
        table = op_def.get("table", {})
        default = op_def.get("default", 1.0)

        # Get source value (e.g., "context.categories")
        if source.startswith("context."):
            key = source[8:]
            source_value = business.get(key, "")
        else:
            source_value = self._namespace.get(source, "")

        # Handle list values (e.g., categories)
        if isinstance(source_value, list):
            source_value = " ".join(source_value)

        # String matching in table
        source_lower = str(source_value).lower()
        for pattern, value in table.items():
            if pattern.lower() in source_lower:
                return value

        return default

    def _compute_case(self, op_def: Dict[str, Any]) -> Any:
        """Apply case rules - supports both threshold conditions and value matching.

        Args:
            op_def: Operation definition with 'source' and 'rules'

        Returns:
            Result from first matching rule
        """
        source = op_def.get("source", "")
        rules = op_def.get("rules", [])

        # Get source value from namespace
        source_value = self._namespace.get(source, 0)

        for rule in rules:
            if "else" in rule:
                return rule["else"]

            when = rule.get("when", "")
            then = rule.get("then", "")

            # Try numeric threshold comparison first (e.g., "< 4.0", ">= 8.0")
            match = re.match(r"([<>=!]+)\s*(\d+(?:\.\d+)?)", str(when))
            if match:
                op, threshold = match.groups()
                threshold = float(threshold)

                try:
                    numeric_value = float(source_value) if source_value is not None else 0
                    if op == "<" and numeric_value < threshold:
                        return then
                    elif op == "<=" and numeric_value <= threshold:
                        return then
                    elif op == ">" and numeric_value > threshold:
                        return then
                    elif op == ">=" and numeric_value >= threshold:
                        return then
                    elif op == "==" and numeric_value == threshold:
                        return then
                    elif op == "!=" and numeric_value != threshold:
                        return then
                except (ValueError, TypeError):
                    pass
            else:
                # Direct value matching (e.g., when: "Mild", when: "Thai")
                if str(source_value) == str(when):
                    return then

        return None

    def _is_extraction_field(self, field_name: str) -> bool:
        """Check if a field name exists in extraction fields.

        Args:
            field_name: Field name to check

        Returns:
            True if field is defined in extraction schema
        """
        fields = self.seed.get("extract", {}).get("fields", [])
        return any(f.get("name") == field_name for f in fields)

    def _execute_compute(self, business: Dict[str, Any]) -> None:
        """Execute all compute operations in order.

        Args:
            business: Restaurant business info (for lookups)
        """
        for op_def in self.seed.get("compute", []):
            name = op_def.get("name", "")
            op = op_def.get("op", "")

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

                # If source is an extraction field, apply case per-extraction and sum
                if self._is_extraction_field(source):
                    total = 0
                    for extraction in self._extractions:
                        result = self._apply_case_to_extraction(extraction, source, rules)
                        if isinstance(result, (int, float)):
                            total += result
                    self._namespace[name] = total
                else:
                    # Source is in namespace (already computed value)
                    self._namespace[name] = self._compute_case(op_def)

    def _get_output(self) -> Dict[str, Any]:
        """Get output values as specified in Formula Seed.

        Returns:
            Dict with output values
        """
        output_fields = self.seed.get("output", [])
        result = {}

        for field in output_fields:
            if field in self._namespace:
                result[field] = self._namespace[field]

        # Always include extractions for observability
        result["_extractions"] = self._extractions
        result["_namespace"] = self._namespace

        return result

    def get_usage_metrics(self) -> Dict[str, Any]:
        """Get aggregated usage metrics from all LLM calls.

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens,
            cost_usd, latency_ms, llm_calls
        """
        if not self._usage_records:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms": 0.0,
                "llm_calls": 0,
            }

        prompt_tokens = sum(u.get("prompt_tokens", 0) for u in self._usage_records)
        completion_tokens = sum(u.get("completion_tokens", 0) for u in self._usage_records)
        cost_usd = sum(u.get("cost_usd", 0.0) for u in self._usage_records)
        latency_ms = sum(u.get("latency_ms", 0.0) for u in self._usage_records)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "llm_calls": len(self._usage_records),
        }

    async def execute(
        self,
        reviews: List[Dict[str, Any]],
        business: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute Formula Seed against restaurant data.

        Args:
            reviews: List of review dicts with 'text' and 'review_id' fields
            business: Restaurant business info dict

        Returns:
            Dict with output values and observability data
        """
        # Step 1: Filter reviews by keywords (Python, no LLM)
        filtered = self._filter_reviews(reviews)

        # Step 2: Extract signals via LLM (parallel)
        await self._extract_signals(filtered)

        # Step 3 & 4: Compute aggregations and execute calculation DAG
        self._execute_compute(business)

        # Step 5: Return output values
        result = self._get_output()

        # Add filtering stats
        result["_filter_stats"] = {
            "total_reviews": len(reviews),
            "filtered_reviews": len(filtered),
            "relevant_extractions": len(self._extractions),
        }

        return result
