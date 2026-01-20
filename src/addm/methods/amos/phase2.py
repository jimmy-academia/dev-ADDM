"""Phase 2: Formula Seed Interpreter.

Executes the Formula Seed against restaurant data:
1. Filter reviews by keywords (Python)
2. Extract signals via LLM (parallel or adaptive)
3. Compute aggregations (Python)
4. Execute computation DAG (Python)
5. Return output values

Supports two execution modes:
- Parallel (default): Process all filtered reviews in parallel
- Adaptive: Batch processing with early stopping based on search strategy
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from addm.llm import LLMService
from addm.methods.amos.config import AMOSConfig
from addm.methods.amos.search.executor import SafeExpressionExecutor
from addm.methods.amos.search.embeddings import HybridRetriever
from addm.utils.async_utils import gather_with_concurrency

logger = logging.getLogger(__name__)


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
    """Executes Formula Seed against restaurant data.

    Supports two execution modes:
    - Parallel (default): Process all filtered reviews in parallel for minimum latency
    - Adaptive: Batch processing with early stopping to save tokens

    The search strategy (generated in Phase 1) guides:
    - Review prioritization (high-value reviews first)
    - Early stopping conditions (when verdict is determinable)
    - Hybrid embedding retrieval (when keywords miss important reviews)
    """

    def __init__(
        self,
        seed: Dict[str, Any],
        llm: LLMService,
        max_concurrent: int = 32,
        config: Optional[AMOSConfig] = None,
    ):
        """Initialize interpreter.

        Args:
            seed: Formula Seed specification
            llm: LLM service for extraction calls
            max_concurrent: Max concurrent LLM calls for extraction
            config: AMOS configuration (controls adaptive mode, hybrid retrieval, etc.)
        """
        self.seed = seed
        self.llm = llm
        self.max_concurrent = max_concurrent
        self.config = config or AMOSConfig()

        # Search strategy from Formula Seed
        self.strategy = seed.get("search_strategy", {})

        # Expression executor for LLM-generated expressions
        self._executor = SafeExpressionExecutor()

        # Hybrid retriever (initialized lazily)
        self._retriever: Optional[HybridRetriever] = None

        # Usage tracking
        self._usage_records: List[Dict[str, Any]] = []
        self._extractions: List[Dict[str, Any]] = []

        # Computed values namespace
        self._namespace: Dict[str, Any] = {}

        # Early stopping state
        self._early_verdict: Optional[str] = None
        self._stopped_early: bool = False
        self._reviews_skipped: int = 0

        # Total reviews count (set during execute)
        self._total_reviews: int = 0

        # Keyword match tracking per review
        self._keyword_hits_map: Dict[str, List[str]] = {}

    def _filter_reviews(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter reviews by keywords (Python, no LLM).

        Also tracks which keywords matched for each review (used for prioritization).

        Args:
            reviews: List of review dicts with 'text' field

        Returns:
            List of reviews that contain at least one keyword
        """
        keywords = self.seed.get("filter", {}).get("keywords", [])
        if not keywords:
            # No filtering - all reviews pass but no keyword hits
            for review in reviews:
                review_id = review.get("review_id", "")
                self._keyword_hits_map[review_id] = []
            return reviews

        # Build regex pattern for keywords (case-insensitive, word boundaries)
        patterns = [(kw, re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)) for kw in keywords]

        filtered = []
        for review in reviews:
            text = review.get("text", "")
            review_id = review.get("review_id", "")
            hits = []

            # Check all keywords and track hits
            for kw, pattern in patterns:
                if pattern.search(text):
                    hits.append(kw)

            self._keyword_hits_map[review_id] = hits

            if hits:  # At least one keyword matched
                filtered.append(review)

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

    # =========================================================================
    # Search Strategy Methods (for adaptive mode)
    # =========================================================================

    def _is_recent(self, review: Dict[str, Any], threshold_years: float = 2.0) -> bool:
        """Check if a review is recent (within threshold years).

        Args:
            review: Review dict with optional 'date' field
            threshold_years: Years threshold for "recent" (default: 2.0)

        Returns:
            True if review is recent or has no date
        """
        review_date = review.get("date")
        if not review_date:
            return False  # Assume not recent if no date

        try:
            if isinstance(review_date, str):
                review_date_dt = datetime.fromisoformat(review_date.replace("Z", "+00:00"))
            else:
                review_date_dt = review_date

            age_years = (datetime.now() - review_date_dt).days / 365.25
            return age_years < threshold_years
        except (ValueError, TypeError):
            return False

    def _compute_review_priority(self, review: Dict[str, Any]) -> float:
        """Compute priority score for a review using LLM-generated expression.

        Higher priority = process first (in adaptive mode).

        Args:
            review: Review dict

        Returns:
            Priority score (higher = more important)
        """
        expr = self.strategy.get("priority_expr", "1.0")
        review_id = review.get("review_id", "")

        # Get keyword hits for this review
        keyword_hits = self._keyword_hits_map.get(review_id, [])

        # Check for priority keywords
        priority_keywords = self.strategy.get("priority_keywords", [])
        priority_hits = [kw for kw in keyword_hits if kw in priority_keywords]

        context = {
            "keyword_hits": keyword_hits,
            "priority_hits": priority_hits,
            "is_recent": self._is_recent(review),
            "embedding_sim": review.get("_embedding_sim", 0.0),
        }

        priority = self._executor.execute_float(expr, context, default=1.0)

        # Boost priority for priority keyword matches
        priority += len(priority_hits) * 3.0

        return priority

    def _sort_reviews_by_priority(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort reviews by priority (highest first).

        Args:
            reviews: List of reviews

        Returns:
            Sorted list of reviews (highest priority first)
        """
        # Compute priorities
        priorities = [(self._compute_review_priority(r), i, r) for i, r in enumerate(reviews)]

        # Sort by priority descending (stable sort by original index for ties)
        priorities.sort(key=lambda x: (-x[0], x[1]))

        return [r for _, _, r in priorities]

    def _check_stopping_condition(self, remaining: int) -> Tuple[bool, Optional[str]]:
        """Check if we can stop early using LLM-generated condition.

        Args:
            remaining: Number of reviews remaining to process

        Returns:
            Tuple of (can_stop, early_verdict)
            - can_stop: True if we can stop processing
            - early_verdict: The verdict if determinable, else None
        """
        stopping_expr = self.strategy.get("stopping_condition", "False")

        # Get current score from namespace (try common names)
        score = self._namespace.get("SCORE", 0)
        if score == 0:
            score = self._namespace.get("FINAL_RISK_SCORE", 0)
        if score == 0:
            score = self._namespace.get("RISK_SCORE", 0)

        # Build context with both lowercase and uppercase versions for flexibility
        context = {
            "extractions": self._extractions,
            "score": score,
            "SCORE": score,  # Uppercase for expressions using SCORE
            "remaining": remaining,
            "namespace": self._namespace,
            # Also expose all namespace values directly for convenience
            **{k: v for k, v in self._namespace.items()},
        }

        can_stop = self._executor.execute_bool(stopping_expr, context, default=False)

        if can_stop:
            # Try to compute early verdict
            verdict_expr = self.strategy.get("early_verdict_expr", "None")
            verdict = self._executor.execute_str(verdict_expr, context, default=None)
            return True, verdict

        return False, None

    async def _extract_adaptive(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract signals with adaptive batch processing and early stopping.

        Processes reviews in batches by priority, checking stopping condition
        after each batch. Saves tokens by not processing remaining reviews
        when verdict is determinable.

        Args:
            reviews: List of filtered reviews (will be sorted by priority)

        Returns:
            List of extraction results (only relevant ones)
        """
        fields = self.seed.get("extract", {}).get("fields", [])
        if not fields or not reviews:
            return []

        # Sort reviews by priority (high-value first)
        sorted_reviews = self._sort_reviews_by_priority(reviews)
        batch_size = self.config.batch_size

        all_extractions = []
        processed = 0

        for i in range(0, len(sorted_reviews), batch_size):
            batch = sorted_reviews[i : i + batch_size]

            # Extract this batch in parallel
            tasks = [self._extract_single_review(r, fields) for r in batch]
            results = await gather_with_concurrency(self.max_concurrent, tasks)

            # Filter to relevant extractions
            relevant = [r for r in results if r.get("is_relevant", False)]
            all_extractions.extend(relevant)
            processed += len(batch)

            # Update internal state for stopping check
            self._extractions = all_extractions

            # Recompute namespace values for stopping check
            # (simplified - only run compute ops that don't depend on full extractions)
            self._execute_compute_partial()

            # Check stopping condition
            remaining = len(sorted_reviews) - processed
            can_stop, verdict = self._check_stopping_condition(remaining)

            if can_stop:
                self._early_verdict = verdict
                self._stopped_early = True
                self._reviews_skipped = remaining
                logger.debug(
                    f"Early stopping: processed={processed}, skipped={remaining}, "
                    f"verdict={verdict}"
                )
                break

        return all_extractions

    def _execute_compute_partial(self) -> None:
        """Execute compute operations for intermediate stopping checks.

        Only computes count and simple sum operations (not case verdicts).
        """
        for op_def in self.seed.get("compute", []):
            name = op_def.get("name", "")
            op = op_def.get("op", "")

            # Only run simple aggregations, not verdict computation
            if op == "count":
                self._namespace[name] = self._compute_count(op_def)
            elif op == "sum":
                self._namespace[name] = self._compute_sum(op_def)
            elif op == "expr":
                self._namespace[name] = self._compute_expr(op_def)

    def _matches_condition(self, extraction: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Check if an extraction matches a where condition.

        Supports:
        - Simple conditions: {"field": "value"}
        - List matching: {"field": ["value1", "value2"]}
        - AND conditions: {"and": [{"field": "FIELD", "equals": "value"}, ...]}
        - OR conditions: {"or": [{"field": "FIELD", "equals": "value"}, ...]}

        Args:
            extraction: Single extraction result
            condition: Dict of {field: value} to match, or complex condition

        Returns:
            True if all conditions match
        """
        # Handle "and" conditions
        if "and" in condition:
            and_conditions = condition["and"]
            for sub_cond in and_conditions:
                if not self._matches_single_condition(extraction, sub_cond):
                    return False
            return True

        # Handle "or" conditions
        if "or" in condition:
            or_conditions = condition["or"]
            for sub_cond in or_conditions:
                if self._matches_single_condition(extraction, sub_cond):
                    return True
            return False

        # Simple key-value conditions
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

    def _matches_single_condition(self, extraction: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Check if extraction matches a single condition object.

        Handles conditions like: {"field": "SEVERITY", "equals": "severe"}

        Args:
            extraction: Single extraction result
            condition: Single condition dict

        Returns:
            True if condition matches
        """
        if "field" in condition and "equals" in condition:
            field = condition["field"]
            expected = condition["equals"]
            actual = extraction.get(field)
            if isinstance(expected, list):
                return actual in expected
            return actual == expected

        # Fall back to simple matching if not in field/equals format
        return self._matches_condition(extraction, condition)

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
        """Apply case rules - supports threshold conditions, expressions, and value matching.

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

            # Try to evaluate 'when' as a full Python expression first
            # This handles cases like "SCORE >= 8" or "SCORE >= 4 and SCORE < 8"
            try:
                # Build context with namespace values (both uppercase and lowercase)
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
                    # Expression evaluated but was False, continue to next rule
                    continue
            except Exception:
                pass

            # Fallback: Try simple numeric threshold comparison (e.g., "< 4.0", ">= 8.0")
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
            cost_usd, latency_ms, llm_calls, and strategy metrics
        """
        if not self._usage_records:
            metrics = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms": 0.0,
                "llm_calls": 0,
            }
        else:
            prompt_tokens = sum(u.get("prompt_tokens", 0) for u in self._usage_records)
            completion_tokens = sum(u.get("completion_tokens", 0) for u in self._usage_records)
            cost_usd = sum(u.get("cost_usd", 0.0) for u in self._usage_records)
            latency_ms = sum(u.get("latency_ms", 0.0) for u in self._usage_records)

            metrics = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "llm_calls": len(self._usage_records),
            }

        # Add strategy metrics
        metrics["strategy_metrics"] = {
            "adaptive_mode": self.config.adaptive,
            "stopped_early": self._stopped_early,
            "early_verdict": self._early_verdict,
            "reviews_skipped": self._reviews_skipped,
            "has_search_strategy": bool(self.strategy),
        }

        # Add hybrid retrieval metrics if available
        if self._retriever:
            retriever_metrics = self._retriever.get_metrics()
            metrics["embedding_tokens"] = retriever_metrics.get("embedding_tokens", 0)
            metrics["embedding_cost_usd"] = retriever_metrics.get("embedding_cost_usd", 0.0)
            metrics["cost_usd"] += retriever_metrics.get("embedding_cost_usd", 0.0)
            metrics["strategy_metrics"]["hybrid_cache_hit"] = retriever_metrics.get("cache_hit", False)

        return metrics

    async def execute(
        self,
        reviews: List[Dict[str, Any]],
        business: Dict[str, Any],
        query: str = "",
        sample_id: str = "",
    ) -> Dict[str, Any]:
        """Execute Formula Seed against restaurant data.

        Supports two execution modes:
        - Parallel (default): Process all filtered reviews in parallel
        - Adaptive: Batch processing with early stopping to save tokens

        Args:
            reviews: List of review dicts with 'text' and 'review_id' fields
            business: Restaurant business info dict
            query: The agenda/query text (for hybrid embedding retrieval)
            sample_id: Sample ID for caching (for hybrid embedding retrieval)

        Returns:
            Dict with output values and observability data
        """
        self._total_reviews = len(reviews)

        # Step 1: Filter reviews by keywords (Python, no LLM)
        filtered = self._filter_reviews(reviews)

        # Step 1.5: Hybrid embedding retrieval (if enabled and strategy says so)
        if self.config.hybrid and self.strategy.get("use_embeddings_when"):
            from pathlib import Path

            cache_path = None
            if self.config.embedding_cache_path:
                cache_path = Path(self.config.embedding_cache_path)

            self._retriever = HybridRetriever(
                cache_path=cache_path,
                embedding_model=self.config.embedding_model,
            )

            filtered = await self._retriever.retrieve_if_needed(
                strategy=self.strategy,
                keyword_matched=filtered,
                all_reviews=reviews,
                query=query,
                executor=self._executor,
                sample_id=sample_id,
            )

        # Step 2: Extract signals via LLM
        if self.config.adaptive:
            # Adaptive mode: batch processing with early stopping
            await self._extract_adaptive(filtered)
        else:
            # Parallel mode: process all at once
            await self._extract_signals(filtered)

        # Step 3 & 4: Compute aggregations and execute calculation DAG
        self._execute_compute(business)

        # Step 4.5: Override verdict with early verdict if stopped early
        if self._stopped_early and self._early_verdict:
            # Only override if verdict not already computed or differs
            current_verdict = self._namespace.get("VERDICT")
            if current_verdict is None or current_verdict != self._early_verdict:
                logger.debug(
                    f"Using early verdict: {self._early_verdict} "
                    f"(computed would be: {current_verdict})"
                )
                self._namespace["VERDICT"] = self._early_verdict

        # Step 5: Return output values
        result = self._get_output()

        # Add filtering stats
        result["_filter_stats"] = {
            "total_reviews": len(reviews),
            "filtered_reviews": len(filtered),
            "relevant_extractions": len(self._extractions),
        }

        # Add strategy execution stats
        result["_strategy_stats"] = {
            "adaptive_mode": self.config.adaptive,
            "stopped_early": self._stopped_early,
            "early_verdict": self._early_verdict,
            "reviews_skipped": self._reviews_skipped,
            "has_search_strategy": bool(self.strategy),
        }

        return result
