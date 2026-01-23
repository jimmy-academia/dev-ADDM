"""Phase 2: Formula Seed Interpreter.

Executes the Formula Seed against restaurant data:
- Process ALL reviews via per-review LLM calls (guaranteed coverage)
- Extract signals via LLM
- Compute verdict from extractions
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from addm.llm import LLMService
from addm.methods.amos.config import AMOSConfig
from addm.methods.amos.search.executor import SafeExpressionExecutor
from addm.methods.amos.seed_transform import transform_formula_seed
from addm.methods.amos.phase2_prompts import (
    SEVERITY_NORMALIZATION_MAP as _SEVERITY_NORMALIZATION_MAP,
    normalize_enum_value as _normalize_enum_value,
    validate_enum_strict as _validate_enum_strict,
    build_extraction_prompt as _build_extraction_prompt,
    parse_extraction_response as _parse_extraction_response,
    build_batch_extraction_prompt as _build_batch_extraction_prompt,
    parse_batch_extraction_response as _parse_batch_extraction_response,
)
from addm.utils.async_utils import gather_with_concurrency
from addm.utils.debug_logger import get_debug_logger
from addm.utils.output import output
from addm.utils.text_validation import validate_multi_span_snippet

logger = logging.getLogger(__name__)


class FormulaSeedInterpreter:
    """Executes Formula Seed against restaurant data.

    Processes ALL reviews via per-review LLM calls for extraction,
    then computes verdict from aggregated extractions.

    Uses per-review extraction (not batched) to guarantee 100% review coverage
    and eliminate silent dropouts from batch response parsing.
    """

    def __init__(
        self,
        seed: Dict[str, Any],
        llm: LLMService,
        max_concurrent: int = 256,
        config: Optional[AMOSConfig] = None,
    ):
        """Initialize interpreter.

        Args:
            seed: Formula Seed specification
            llm: LLM service for extraction calls
            max_concurrent: Max concurrent LLM calls for extraction
            config: AMOS configuration
        """
        # Transform seed to ensure phase2-compatible format
        # (handles LLM-generated SQL expressions → op/expr/where format)
        self.seed = transform_formula_seed(seed)
        self.llm = llm
        self.max_concurrent = max_concurrent
        self.config = config or AMOSConfig()

        # Expression executor for LLM-generated expressions
        self._executor = SafeExpressionExecutor()

        # Usage tracking
        self._usage_records: List[Dict[str, Any]] = []
        self._extractions: List[Dict[str, Any]] = []

        # Computed values namespace
        self._namespace: Dict[str, Any] = {}

        # Total reviews count (set during execute)
        self._total_reviews: int = 0

        # Snippet validation stats
        self._snippet_validation_stats: Dict[str, int] = {
            "total_relevant": 0,
            "valid_quotes": 0,
            "rejected_no_quote": 0,
            "rejected_quote_not_found": 0,
        }

        # Enum validation stats (specification enforcement)
        self._enum_validation_stats: Dict[str, Any] = {
            "total_validated": 0,
            "valid": 0,
            "rejected": 0,
            "rejection_details": [],  # List of {field, value, expected, error}
        }

        # Current sample ID for debug logging
        self._current_sample_id: str = ""

        # Wall-clock timing (set during execute)
        self._wall_clock_ms: float = 0.0

    def _validate_and_store_snippet(
        self,
        extraction: Dict[str, Any],
        quote: str,
        review_text: str,
    ) -> Dict[str, Any]:
        """Validate that the supporting quote exists in the review text.

        Uses multi-span validation to handle LLM quotes that combine non-adjacent
        sentences. Requires 80% of segments to exist in monotonic order.

        Args:
            extraction: Extraction result dict
            quote: The supporting_quote from extraction
            review_text: The original review text

        Returns:
            Updated extraction dict (may have is_relevant set to False if validation fails)
        """
        self._snippet_validation_stats["total_relevant"] += 1

        # Check if quote is empty or missing
        if not quote or not quote.strip():
            extraction["is_relevant"] = False
            extraction["_rejected_reason"] = "no_quote_provided"
            self._snippet_validation_stats["rejected_no_quote"] += 1
            logger.debug(f"Rejection: no quote provided for {extraction.get('review_id')}")
            return extraction

        # Use multi-span validation (handles combined non-adjacent sentences)
        validation = validate_multi_span_snippet(quote, review_text)

        if validation["valid"]:
            extraction["_snippet"] = quote
            extraction["_snippet_validated"] = True
            extraction["_snippet_match_type"] = validation["match_type"]
            if validation["match_type"] in ("multi_span", "word_overlap"):
                extraction["_snippet_match_quality"] = f"{validation['match_ratio']:.0%}"
            self._snippet_validation_stats["valid_quotes"] += 1
            return extraction

        # Quote not found - reject extraction
        extraction["is_relevant"] = False
        extraction["_rejected_reason"] = "quote_not_found"
        extraction["_attempted_quote"] = quote[:100]  # Store for debugging
        self._snippet_validation_stats["rejected_quote_not_found"] += 1
        logger.debug(
            f"Rejection: quote not found in review {extraction.get('review_id')}: "
            f"'{quote[:50]}...' (match_type={validation['match_type']}, "
            f"ratio={validation['match_ratio']:.0%})"
        )
        return extraction

    def _validate_enum_fields(self, extraction: Dict[str, Any]) -> Dict[str, Any]:
        """Validate enum fields in extraction against expected values.

        Rejects extractions with invalid enum values rather than silently normalizing.
        This enforces the specification: outputs must use defined enum values.

        Args:
            extraction: Extraction dict with field values

        Returns:
            Extraction with validation errors added if any fields invalid
        """
        if not extraction.get("is_relevant", False):
            return extraction

        fields = self.seed.get("extract", {}).get("fields", [])
        validation_errors = extraction.get("_validation_errors", [])

        for field in fields:
            field_name = field.get("name", "")
            field_type = field.get("type", "")
            values = field.get("values", {})

            if field_type != "enum" or not values:
                continue

            # Get the actual value from extraction (case-insensitive lookup)
            actual = self._get_field_value(extraction, field_name)
            if actual is None:
                continue

            expected_values = list(values.keys())
            is_valid, error_msg, normalized = _validate_enum_strict(
                actual, expected_values, field_name
            )

            self._enum_validation_stats["total_validated"] += 1

            if is_valid:
                self._enum_validation_stats["valid"] += 1
                # Store the normalized value back
                extraction[field_name.lower()] = normalized
            else:
                self._enum_validation_stats["rejected"] += 1
                self._enum_validation_stats["rejection_details"].append({
                    "review_id": extraction.get("review_id", "unknown"),
                    "field": field_name,
                    "value": actual,
                    "expected": expected_values,
                    "error": error_msg,
                })
                validation_errors.append(error_msg)
                logger.debug(f"Enum validation failed: {error_msg}")

        if validation_errors:
            extraction["_validation_errors"] = validation_errors
            # Don't mark as non-relevant, just track the errors
            # The extraction can still be used but with known issues

        return extraction

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

        # Get task context from seed for dynamic prompt generation
        task_name = self.seed.get("task_name", "relevant information")
        extraction_guidelines = self.seed.get("extraction_guidelines")

        prompt = _build_extraction_prompt(
            llm_fields, review_text, review_id,
            task_name=task_name,
            extraction_guidelines=extraction_guidelines,
        )
        messages = [{"role": "user", "content": prompt}]

        response, usage = await self.llm.call_async_with_usage(
            messages,
            context={"phase": "phase2_extract", "review_id": review_id},
        )

        self._usage_records.append(usage)

        # Debug log the LLM call
        if debug_logger := get_debug_logger():
            debug_logger.log_llm_call(
                sample_id=self._current_sample_id,
                phase="phase2_extract_single",
                prompt=prompt,
                response=response,
                model=self.llm._config.get("model", "unknown"),
                latency_ms=usage.get("latency_ms", 0.0),
                metadata={"review_id": review_id},
            )

        extraction = _parse_extraction_response(response)
        extraction["review_id"] = review_id

        # Validate supporting_quote exists in source text
        if extraction.get("is_relevant", False):
            quote = extraction.get("supporting_quote", "")
            extraction = self._validate_and_store_snippet(extraction, quote, review_text)

        # Validate enum fields (specification enforcement)
        if extraction.get("is_relevant", False):
            extraction = self._validate_enum_fields(extraction)

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

    async def _extract_batch_reviews(
        self,
        reviews: List[Dict[str, Any]],
        fields: List[Dict[str, Any]],
        batch_size: int = 10,
    ) -> List[Dict[str, Any]]:
        """Extract signals from multiple reviews in batched LLM calls.

        Much more efficient than single-review extraction:
        - 50 reviews with batch_size=10 = 5 LLM calls (vs 50 calls)

        Args:
            reviews: List of reviews to process
            fields: Field definitions from Formula Seed
            batch_size: Reviews per LLM call (default: 10)

        Returns:
            List of extraction results
        """
        if not reviews or not fields:
            return []

        # Filter out temporal fields
        temporal_field_names = {"REVIEW_DATE", "AGE_YEARS"}
        llm_fields = [f for f in fields if f.get("name") not in temporal_field_names]

        task_name = self.seed.get("task_name", "relevant information")
        extraction_guidelines = self.seed.get("extraction_guidelines")

        # Create review lookup for post-processing
        review_lookup = {r.get("review_id"): r for r in reviews}

        # Process in batches
        all_results = []
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            review_ids = [r.get("review_id", f"unknown_{j}") for j, r in enumerate(batch)]

            prompt = _build_batch_extraction_prompt(
                llm_fields, batch, task_name, extraction_guidelines
            )
            messages = [{"role": "user", "content": prompt}]

            response, usage = await self.llm.call_async_with_usage(
                messages,
                context={"phase": "phase2_batch_extract", "batch_idx": i // batch_size},
            )
            self._usage_records.append(usage)

            # Debug log the LLM call
            if debug_logger := get_debug_logger():
                debug_logger.log_llm_call(
                    sample_id=self._current_sample_id,
                    phase="phase2_batch_extract",
                    prompt=prompt,
                    response=response,
                    model=self.llm._config.get("model", "unknown"),
                    latency_ms=usage.get("latency_ms", 0.0),
                    metadata={"batch_idx": i // batch_size, "batch_size": len(batch)},
                )

            # Parse batch response
            batch_results = _parse_batch_extraction_response(response, review_ids)

            # Post-process each extraction
            for extraction in batch_results:
                rid = extraction.get("review_id", "")
                review = review_lookup.get(rid, {})
                review_text = review.get("text", "")

                # Validate quote
                if extraction.get("is_relevant", False):
                    quote = extraction.get("supporting_quote", "")
                    extraction = self._validate_and_store_snippet(extraction, quote, review_text)

                # Validate enum fields (specification enforcement)
                if extraction.get("is_relevant", False):
                    extraction = self._validate_enum_fields(extraction)

                all_results.append(extraction)

        return all_results

    async def _extract_signals(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract signals from all reviews via per-review LLM calls.

        Uses per-review extraction for reliability (no silent dropouts from batch
        parsing). Each review gets its own LLM call, guaranteeing 100% coverage.

        Args:
            reviews: List of filtered reviews

        Returns:
            List of extraction results (only relevant ones)
        """
        fields = self.seed.get("extract", {}).get("fields", [])
        if not fields or not reviews:
            return []

        # Use PER-REVIEW extraction for reliability (no silent dropouts)
        # Trade-off: More LLM calls, but guaranteed coverage
        tasks = [self._extract_single_review(r, fields) for r in reviews]
        results = await gather_with_concurrency(self.max_concurrent, tasks)

        # Filter to only relevant extractions
        relevant = [r for r in results if r.get("is_relevant", False)]
        self._extractions = relevant

        return relevant

    def _filter_to_incidents_only(self) -> int:
        """Filter extractions to only include actual incidents (outcome != "none").

        SPECIFICATION ENFORCEMENT: Only extractions representing actual incidents
        should contribute to verdict-affecting counts. This prevents scenarios where
        e.g., a review mentioning staff knowledge (but no incident) triggers High Risk.

        Updates self._extractions in place and returns the count of filtered-out
        extractions for observability.

        Returns:
            Number of extractions filtered out (non-incidents)
        """
        outcome_field, none_values = self._get_outcome_field_info()
        if not outcome_field:
            # No outcome field defined - can't filter, keep all
            logger.debug("No outcome field defined in seed - skipping incident filter")
            return 0

        original_count = len(self._extractions)
        filtered = []

        for ext in self._extractions:
            outcome_value = self._get_field_value(ext, outcome_field)
            if self._is_none_value(outcome_value, none_values):
                # This extraction has "no incident" outcome - filter it out
                logger.debug(
                    f"Filtering out non-incident extraction: review_id={ext.get('review_id')}, "
                    f"{outcome_field}={outcome_value}"
                )
                continue
            filtered.append(ext)

        self._extractions = filtered
        filtered_count = original_count - len(filtered)

        if filtered_count > 0:
            logger.info(
                f"Incident filter: removed {filtered_count}/{original_count} "
                f"non-incident extractions (outcome_field={outcome_field}, "
                f"none_values={none_values})"
            )

        return filtered_count

    def _get_enum_values_for_field(self, field_name: str) -> Optional[List[str]]:
        """Get expected enum values for a field from the Formula Seed.

        Args:
            field_name: Name of the field (e.g., "INCIDENT_SEVERITY")

        Returns:
            List of valid enum values if field is an enum, None otherwise
        """
        fields = self.seed.get("extract", {}).get("fields", [])
        for field in fields:
            if field.get("name") == field_name and field.get("type") == "enum":
                values = field.get("values", {})
                if values:
                    return list(values.keys())
        return None

    def _get_outcome_field_info(self) -> Tuple[Optional[str], List[str]]:
        """Get outcome field name and none values from seed metadata.

        This is the generalizable approach: Phase 1 declares which field
        represents the outcome and what values mean "no incident".

        Returns:
            Tuple of (outcome_field_name, list_of_none_values)
            - outcome_field_name: e.g., "INCIDENT_SEVERITY", "QUALITY_LEVEL"
            - list_of_none_values: e.g., ["none", "n/a", "no incident"]
        """
        extract = self.seed.get("extract", {})

        # First try seed metadata (generalizable)
        outcome_field = extract.get("outcome_field")
        none_values = extract.get("none_values", [])

        if outcome_field:
            return outcome_field, none_values

        # Fallback: detect outcome field from field names (backwards compatibility)
        fields = extract.get("fields", [])
        for field in fields:
            field_name = field.get("name", "")
            field_lower = field_name.lower()

            # Look for severity/outcome/quality/level fields
            if any(x in field_lower for x in ["severity", "outcome", "quality", "level", "intensity"]):
                values = field.get("values", {})
                if isinstance(values, dict):
                    # Detect none values from field values
                    detected_none = []
                    for v in values.keys():
                        v_lower = v.lower()
                        if any(ind in v_lower for ind in ["none", "no ", "n/a", "not ", "absent", "nothing"]):
                            detected_none.append(v)
                    return field_name, detected_none

        return None, []

    def _is_none_value(self, value: Any, none_values: List[str]) -> bool:
        """Check if a value represents "no incident" using seed's none_values.

        Args:
            value: The value to check
            none_values: List of values that mean "no incident" from seed

        Returns:
            True if value represents "no incident"
        """
        if value is None:
            return True

        value_str = str(value).strip().lower()

        # Check against seed's none_values (case-insensitive)
        for nv in none_values:
            if value_str == nv.lower():
                return True

        # Also check normalization map as fallback for common variations
        if value_str in _SEVERITY_NORMALIZATION_MAP:
            normalized = _SEVERITY_NORMALIZATION_MAP[value_str]
            if normalized == "none":
                return True

        # Keyword-based detection for unlisted variations
        none_indicators = ["no ", "none", "n/a", "not ", "absent", "nothing", "irrelevant"]
        has_none_indicator = any(ind in value_str for ind in none_indicators)

        # Only return True if it has none indicator AND no severity keywords
        if has_none_indicator:
            severity_keywords = ["mild", "moderate", "severe", "serious", "critical", "emergency", "life"]
            has_severity = any(kw in value_str for kw in severity_keywords)
            if not has_severity:
                return True

        return False

    def _normalize_actual_value(self, field_name: str, actual: Any) -> Any:
        """Normalize an actual value from extraction for comparison.

        Args:
            field_name: Name of the field
            actual: The actual value from extraction

        Returns:
            Normalized value if field is an enum, otherwise original value
        """
        expected_values = self._get_enum_values_for_field(field_name)
        if expected_values:
            return _normalize_enum_value(actual, expected_values)
        return actual

    def _get_field_value(self, extraction: Dict[str, Any], field_name: str) -> Any:
        """Get field value from extraction, handling case-insensitive lookup.

        Formula Seeds use uppercase field names (e.g., INCIDENT_SEVERITY) while
        LLM extractions often return lowercase (e.g., incident_severity). This
        helper tries multiple case variants.

        Args:
            extraction: Extraction dict with field values
            field_name: Field name to look up

        Returns:
            Field value if found, None otherwise
        """
        # Try exact match first
        if field_name in extraction:
            return extraction[field_name]
        # Try lowercase
        lower_name = field_name.lower()
        if lower_name in extraction:
            return extraction[lower_name]
        # Try uppercase
        upper_name = field_name.upper()
        if upper_name in extraction:
            return extraction[upper_name]
        # Try case-insensitive search through all keys (handles mixed case like Account_Type)
        for key in extraction:
            if key.lower() == lower_name:
                return extraction[key]
        return None

    def _fuzzy_enum_match(self, actual: str, expected: str) -> bool:
        """Fuzzy match enum values to handle label inconsistencies.

        Handles cases like:
        - "moderate incident" vs "Moderate"
        - "Severe incident" vs "Severe"
        - "no reaction" vs "No reaction"

        Args:
            actual: The actual value from extraction
            expected: The expected value from compute.where

        Returns:
            True if values match (fuzzy)
        """
        if actual is None or expected is None:
            return actual == expected

        a, e = str(actual).lower().strip(), str(expected).lower().strip()

        # Exact match
        if a == e:
            return True

        # Partial containment (either direction)
        if e in a or a in e:
            return True

        # Strip common suffixes and compare
        suffixes = [' incident', ' reaction', ' risk', ' level', ' quality']
        a_stripped = a
        e_stripped = e
        for suffix in suffixes:
            a_stripped = a_stripped.replace(suffix, '')
            e_stripped = e_stripped.replace(suffix, '')

        if a_stripped == e_stripped:
            return True

        return False

    def _matches_condition(self, extraction: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Check if an extraction matches a where condition.

        Supports:
        - Simple conditions: {"field": "value"}
        - List matching: {"field": ["value1", "value2"]}
        - AND conditions: {"and": [{"field": "FIELD", "equals": "value"}, ...]}
        - OR conditions: {"or": [{"field": "FIELD", "equals": "value"}, ...]}
        - Field/equals with nested: {"field": "X", "equals": "Y", "and": [...]}
        - not_equals: {"field": "X", "not_equals": "Y"}

        Applies enum value normalization for enum fields (e.g., "no reaction" -> "none").
        Uses fuzzy matching for string comparisons to handle label inconsistencies.

        Args:
            extraction: Single extraction result
            condition: Dict of {field: value} to match, or complex condition

        Returns:
            True if all conditions match
        """
        # Check field/equals at top level first (if present)
        if "field" in condition and ("equals" in condition or "not_equals" in condition):
            field = condition["field"]
            actual = self._get_field_value(extraction, field)
            # Normalize actual value for enum fields
            actual = self._normalize_actual_value(field, actual)

            if "equals" in condition:
                expected = condition["equals"]
                if isinstance(expected, list):
                    # Fuzzy list matching for strings
                    if isinstance(actual, str):
                        if not any(
                            self._fuzzy_enum_match(actual, e) if isinstance(e, str) else actual == e
                            for e in expected
                        ):
                            return False
                    elif actual not in expected:
                        return False
                else:
                    # Fuzzy comparison for strings
                    if isinstance(actual, str) and isinstance(expected, str):
                        if not self._fuzzy_enum_match(actual, expected):
                            return False
                    elif actual != expected:
                        return False

            if "not_equals" in condition:
                not_expected = condition["not_equals"]
                if isinstance(not_expected, list):
                    # Fuzzy list matching for strings (negated)
                    if isinstance(actual, str):
                        if any(
                            self._fuzzy_enum_match(actual, e) if isinstance(e, str) else actual == e
                            for e in not_expected
                        ):
                            return False
                    elif actual in not_expected:
                        return False
                else:
                    # Fuzzy comparison for strings (negated)
                    if isinstance(actual, str) and isinstance(not_expected, str):
                        if self._fuzzy_enum_match(actual, not_expected):
                            return False
                    elif actual == not_expected:
                        return False

        # Handle "and" conditions (check all must match)
        if "and" in condition:
            and_conditions = condition["and"]
            for sub_cond in and_conditions:
                if not self._matches_single_condition(extraction, sub_cond):
                    return False

        # Handle "or" conditions (check at least one must match)
        if "or" in condition:
            or_conditions = condition["or"]
            found_match = False
            for sub_cond in or_conditions:
                if self._matches_single_condition(extraction, sub_cond):
                    found_match = True
                    break
            if not found_match:
                return False

        # If no special keys, use simple key-value matching
        special_keys = {"field", "equals", "not_equals", "and", "or"}
        simple_conditions = {k: v for k, v in condition.items() if k not in special_keys}

        for field, expected in simple_conditions.items():
            actual = self._get_field_value(extraction, field)
            # Normalize actual value for enum fields
            actual = self._normalize_actual_value(field, actual)

            # Fuzzy comparison for string values (handles enum label mismatches)
            if isinstance(actual, str) and isinstance(expected, str):
                matches = self._fuzzy_enum_match(actual, expected)
            elif isinstance(expected, list):
                # Match any in list (fuzzy for strings)
                if isinstance(actual, str):
                    matches = any(
                        self._fuzzy_enum_match(actual, e) if isinstance(e, str) else actual == e
                        for e in expected
                    )
                else:
                    matches = actual in expected
            else:
                matches = actual == expected

            logger.debug(
                f"[WHERE DEBUG] field={field}, expected={expected!r}, actual={actual!r}, "
                f"match={matches}"
            )

            if not matches:
                return False

        return True

    def _matches_single_condition(self, extraction: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Check if extraction matches a single condition object.

        Handles conditions like: {"field": "SEVERITY", "equals": "severe"}
        Applies enum value normalization for enum fields.
        Uses fuzzy matching for string values to handle label inconsistencies.

        Args:
            extraction: Single extraction result
            condition: Single condition dict

        Returns:
            True if condition matches
        """
        if "field" in condition and "equals" in condition:
            field = condition["field"]
            expected = condition["equals"]
            actual = self._get_field_value(extraction, field)
            # Normalize actual value for enum fields
            actual = self._normalize_actual_value(field, actual)

            # Fuzzy comparison for strings (handles enum label mismatches)
            if isinstance(expected, list):
                if isinstance(actual, str):
                    return any(
                        self._fuzzy_enum_match(actual, e) if isinstance(e, str) else actual == e
                        for e in expected
                    )
                return actual in expected

            if isinstance(actual, str) and isinstance(expected, str):
                return self._fuzzy_enum_match(actual, expected)
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
        # Support both canonical 'where' and legacy 'condition' keys
        where = op_def.get("where") or op_def.get("condition", {})
        if not where:
            return len(self._extractions)

        return sum(1 for e in self._extractions if self._matches_condition(e, where))

    def _eval_sql_case_expr(self, expr: str, extraction: Dict[str, Any]) -> float:
        """Evaluate SQL-style CASE expression against an extraction.

        Handles single CASE expressions and multiple CASE blocks connected with +.
        Examples:
            "CASE WHEN SEVERITY = 'Mild incident' THEN 2 WHEN SEVERITY = 'moderate' THEN 5 ELSE 0 END"
            "CASE WHEN X = 'a' THEN 5 ELSE 0 END + CASE WHEN Y = 'b' THEN 3 ELSE 0 END"

        Args:
            expr: SQL CASE expression (single or multiple with +)
            extraction: Extraction dict with field values

        Returns:
            Numeric result from CASE evaluation (summed if multiple CASE blocks)
        """
        # Split by + to handle multiple CASE blocks
        # Pattern: CASE ... END
        case_pattern = r"CASE\s+(.+?)\s+END"
        case_blocks = re.findall(case_pattern, expr, re.IGNORECASE | re.DOTALL)

        if not case_blocks:
            return 0.0

        total = 0.0
        for case_body in case_blocks:
            total += self._eval_single_case(case_body, extraction)

        return total

    def _eval_single_case(self, case_body: str, extraction: Dict[str, Any]) -> float:
        """Evaluate a single CASE body (content between CASE and END).

        Args:
            case_body: The content between CASE and END
            extraction: Extraction dict with field values

        Returns:
            Numeric result from the first matching WHEN, or ELSE value
        """
        # Extract WHEN/THEN pairs
        # Pattern: WHEN field = 'value with spaces' THEN number
        # Handle both single and double quotes, capturing multi-word values
        when_pattern = r"WHEN\s+(\w+)\s*=\s*['\"]([^'\"]+)['\"]\s+THEN\s+(\d+(?:\.\d+)?)"
        matches = re.findall(when_pattern, case_body, re.IGNORECASE)

        # Also try unquoted single-word values
        when_pattern_unquoted = r"WHEN\s+(\w+)\s*=\s*(\w+)\s+THEN\s+(\d+(?:\.\d+)?)"
        matches.extend(re.findall(when_pattern_unquoted, case_body, re.IGNORECASE))

        # Also handle IN (...) clauses: WHEN field IN ('val1','val2') THEN number
        in_pattern = r"WHEN\s+(\w+)\s+IN\s*\(([^)]+)\)\s+THEN\s+(\d+(?:\.\d+)?)"
        in_matches = re.findall(in_pattern, case_body, re.IGNORECASE)

        # DEBUG: Log CASE evaluation details
        logger.debug(f"[CASE DEBUG] case_body: {case_body[:100]}...")
        logger.debug(f"[CASE DEBUG] matches found: {matches}")

        for field, value, then_value in matches:
            actual_value = self._get_field_value(extraction, field)
            # Normalize enum values before comparison (handles "no reaction" -> "none", etc.)
            actual_value = self._normalize_actual_value(field, actual_value)
            logger.debug(
                f"[CASE DEBUG] checking: field={field}, expected='{value}', "
                f"actual='{actual_value}', actual_type={type(actual_value).__name__}"
            )
            if actual_value is None:
                actual_value = ""

            # Use fuzzy matching to handle label inconsistencies (e.g., "Severe incident" vs "Severe")
            if self._fuzzy_enum_match(str(actual_value), value):
                logger.debug(f"[CASE DEBUG] MATCH! returning {then_value}")
                return float(then_value)

        # Check IN clauses
        for field, values_str, then_value in in_matches:
            actual_value = self._get_field_value(extraction, field)
            # Normalize enum values before comparison
            actual_value = self._normalize_actual_value(field, actual_value)
            if actual_value is None:
                actual_value = ""

            # Parse the values list (e.g., "'Thai','Vietnamese','Chinese'")
            # Remove quotes and split by comma
            values = [v.strip().strip("'\"") for v in values_str.split(",")]
            if str(actual_value).lower().strip() in [v.lower().strip() for v in values]:
                return float(then_value)

        # Check for ELSE clause
        else_pattern = r"ELSE\s+(\d+(?:\.\d+)?)"
        else_match = re.search(else_pattern, case_body, re.IGNORECASE)
        if else_match:
            logger.debug(f"[CASE DEBUG] no match, returning ELSE value: {else_match.group(1)}")
            return float(else_match.group(1))

        logger.debug("[CASE DEBUG] no match and no ELSE, returning 0.0")
        return 0.0

    def _compute_sum(self, op_def: Dict[str, Any]) -> float:
        """Compute sum aggregation.

        Args:
            op_def: Operation definition with 'expr' and optional 'where'

        Returns:
            Sum of expression values across matching extractions
        """
        expr = op_def.get("expr", "1")
        name = op_def.get("name", "UNNAMED_SUM")
        # Support both canonical 'where' and legacy 'condition' keys
        where = op_def.get("where") or op_def.get("condition", {})

        # Check if this is a SQL-style CASE expression
        is_sql_case = expr.strip().upper().startswith("CASE")

        total = 0.0
        for extraction in self._extractions:
            if where and not self._matches_condition(extraction, where):
                logger.debug(
                    f"[SUM DEBUG] {name}: extraction {extraction.get('review_id')} "
                    f"skipped - where condition not met"
                )
                continue

            if is_sql_case:
                # Handle SQL-style CASE WHEN ... THEN ... END
                value = self._eval_sql_case_expr(expr, extraction)
                logger.debug(
                    f"[SUM DEBUG] {name}: extraction {extraction.get('review_id')} "
                    f"CASE eval = {value}, fields = {list(extraction.keys())}"
                )
                total += value
            else:
                # Evaluate as Python expression in extraction context
                try:
                    # Create safe namespace with extraction values
                    safe_namespace = {
                        **{k: v for k, v in extraction.items() if not k.startswith("_")},
                        **self._namespace,
                    }
                    value = eval(expr, {"__builtins__": {}}, safe_namespace)
                    logger.debug(
                        f"[SUM DEBUG] {name}: extraction {extraction.get('review_id')} "
                        f"expr eval = {value}"
                    )
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
        source_value = self._get_field_value(extraction, source)
        if source_value is None:
            source_value = "none"

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
                # Direct value matching (case-insensitive for strings)
                if str(source_value).lower() == str(when).lower():
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

        # No rule matched - use default_verdict or default as fallback
        default_result = op_def.get("default_verdict") or op_def.get("default")
        if default_result is not None:
            logger.debug(
                f"No rule matched, using default: {default_result} "
                f"(source={source}, value={source_value})"
            )
            return default_result

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
            # Support both canonical 'op' and legacy 'operation' keys
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

    def _build_standard_output(self) -> Dict[str, Any]:
        """Transform internal _extractions + _namespace to standard output format.

        Converts AMOS internal format to the format defined in output_schema.txt:
        {
            "verdict": str,
            "evidences": [{evidence_id, review_id, field, judgement, snippet}],
            "justification": {triggered_rule, direct_evidence, scoring_trace, reasoning}
        }

        Uses seed's outcome_field and none_values for generalizable incident detection
        across all policy types (G1-G6).

        Returns:
            Dict in standard output format
        """
        evidences = []
        evidence_idx = 1

        # Get outcome field info from seed metadata (generalizable approach)
        outcome_field, none_values = self._get_outcome_field_info()

        for ext in self._extractions:
            review_id = ext.get("review_id", ext.get("_review_id", "unknown"))
            snippet = ext.get("_snippet", "")

            # If no snippet stored, try supporting_quote field or extraction metadata
            if not snippet:
                snippet = ext.get("supporting_quote", "")
            if not snippet and "_source_text" in ext:
                snippet = ext["_source_text"][:200]  # Limit snippet length

            # Check if this extraction represents an actual incident
            # Use seed's outcome_field if available, fall back to common names
            if outcome_field:
                outcome_value = self._get_field_value(ext, outcome_field)
            else:
                # Fallback: try common outcome field names (backwards compatibility)
                outcome_value = (
                    self._get_field_value(ext, "incident_severity") or
                    self._get_field_value(ext, "severity") or
                    self._get_field_value(ext, "outcome") or
                    self._get_field_value(ext, "quality_level") or
                    "none"
                )

            # Skip extractions that don't represent actual incidents
            if self._is_none_value(outcome_value, none_values):
                continue

            for field, value in ext.items():
                # Skip internal fields, metadata, and supporting_quote (stored as _snippet)
                if field.startswith("_") or field in ("review_id", "is_relevant", "supporting_quote"):
                    continue

                evidences.append({
                    "evidence_id": f"E{evidence_idx}",
                    "review_id": review_id,
                    "field": field.lower(),
                    "judgement": str(value).lower() if value else "none",
                    "snippet": snippet,
                })
                evidence_idx += 1

        # Build justification from computed values
        justification = self._build_justification(evidences)

        # Export computed values for specification verification
        # This makes the deterministic computation transparent
        computed_values = self._get_computed_values_export()

        return {
            "verdict": self._namespace.get("VERDICT"),
            "evidences": evidences,
            "justification": justification,
            "computed_values": computed_values,
            "validation_stats": {
                "enum_validation": self._enum_validation_stats,
                "snippet_validation": self._snippet_validation_stats,
            },
            "other_notes": None,
        }

    def _get_computed_values_export(self) -> Dict[str, Any]:
        """Export computed namespace values for transparency.

        Returns all computed values (N_*, SCORE, VERDICT, etc.) so that
        the deterministic computation can be verified.

        Returns:
            Dict of computed values from namespace
        """
        export = {}
        for key, value in self._namespace.items():
            # Export all computed values (N_*, SCORE, VERDICT, etc.)
            # Skip internal/temporary values starting with underscore
            if not key.startswith("_"):
                export[key] = value
        return export

    def _build_justification(self, evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build justification section from computed values.

        Args:
            evidences: List of evidence dicts from _build_standard_output

        Returns:
            Justification dict with triggered_rule, direct_evidence, scoring_trace, reasoning
        """
        # Get score from namespace (try common names)
        score = self._namespace.get("SCORE", 0)
        if score == 0:
            score = self._namespace.get("FINAL_RISK_SCORE", 0)
        if score == 0:
            score = self._namespace.get("RISK_SCORE", 0)

        verdict = self._namespace.get("VERDICT", "Unknown")

        # Determine if this is a scoring-based policy (V2/V3) or count-based (V0/V1)
        is_scoring_based = "SCORE" in self._namespace or "INCIDENT_POINTS" in self._namespace

        # Get outcome field from seed (generalizable across G1-G6)
        outcome_field, _ = self._get_outcome_field_info()
        # Build list of field names to check for outcome (lowercase for comparison)
        outcome_field_names = {"incident_severity", "severity", "outcome", "quality_level"}
        if outcome_field:
            outcome_field_names.add(outcome_field.lower())

        # Build scoring trace from incident data
        breakdown = []
        if is_scoring_based:
            # Scoring-based policies (V2/V3): Show point breakdown
            for ev in evidences:
                if ev["field"] in outcome_field_names:
                    points = self._get_points_for_severity(ev["judgement"])
                    if points > 0:
                        modifiers = []
                        # Check for V3 recency modifiers
                        age_years = self._namespace.get("AGE_YEARS")
                        if age_years is not None and age_years > 2.0:
                            modifiers.append("recency_decay")

                        breakdown.append({
                            "evidence_id": ev["evidence_id"],
                            "base_points": str(points),
                            "modifiers": modifiers,
                            "subtotal": str(points),
                        })
        else:
            # Count-based policies (V0/V1): Show count as breakdown
            incident_count = self._namespace.get("N_INCIDENTS", 0)
            if incident_count > 0:
                # Add one breakdown entry per counted incident
                for ev in evidences:
                    if ev["field"] in outcome_field_names:
                        breakdown.append({
                            "evidence_id": ev["evidence_id"],
                            "base_points": "1",
                            "modifiers": [],
                            "subtotal": "1",
                        })

        # Find direct evidence that triggered verdict
        direct = [b["evidence_id"] for b in breakdown if int(b.get("subtotal", 0)) > 0][:5]

        # Build triggered rule description
        if is_scoring_based:
            triggered_rule = self._get_triggered_rule_scoring(score, verdict)
        else:
            incident_count = self._namespace.get("N_INCIDENTS", 0)
            triggered_rule = self._get_triggered_rule_count(incident_count, verdict)

        # Generate reasoning
        reasoning = self._generate_reasoning(score, verdict, len(breakdown), is_scoring_based)

        return {
            "triggered_rule": triggered_rule,
            "direct_evidence": direct if direct else [],
            "scoring_trace": {
                "total_score": str(score) if is_scoring_based else str(len(breakdown)),
                "breakdown": breakdown,
            },
            "reasoning": reasoning,
        }

    def _get_points_for_severity(self, severity: str) -> int:
        """Get point value for a severity/outcome level.

        Uses seed's compute operations to find point values dynamically.
        Falls back to default mapping for common severity levels.

        Args:
            severity: Severity/outcome string (e.g., "severe", "moderate", "mild")

        Returns:
            Point value for the severity/outcome
        """
        # Default point mapping (can be overridden by seed)
        severity_points = {
            "severe": 15,
            "moderate": 8,
            "mild": 3,
            "none": 0,
        }

        # Get outcome field from seed for dynamic lookup
        outcome_field, _ = self._get_outcome_field_info()

        # Try to get points from seed compute operations
        for op in self.seed.get("compute", []):
            if op.get("op") == "case":
                source = op.get("source", "")
                # Check if this operation uses the outcome field (case-insensitive)
                if outcome_field and source.lower() == outcome_field.lower():
                    for rule in op.get("rules", []):
                        if rule.get("when", "").lower() == severity.lower():
                            try:
                                return int(rule.get("then", 0))
                            except (ValueError, TypeError):
                                pass
                # Fallback: check common outcome field names
                elif source.lower() in ("incident_severity", "severity", "outcome", "quality_level"):
                    for rule in op.get("rules", []):
                        if rule.get("when", "").lower() == severity.lower():
                            try:
                                return int(rule.get("then", 0))
                            except (ValueError, TypeError):
                                pass

        return severity_points.get(severity.lower(), 0)

    def _get_triggered_rule_scoring(self, score: float, verdict: str) -> str:
        """Get triggered rule description for scoring-based policies.

        SPECIFICATION ENFORCEMENT: Actually evaluates each rule condition
        against computed namespace values, not just text-matching the verdict.

        Args:
            score: The computed score
            verdict: The verdict

        Returns:
            Rule description string with actual computed values
        """
        # Find the case operation that produces VERDICT
        for op in self.seed.get("compute", []):
            if op.get("name") == "VERDICT" and op.get("op") == "case":
                for rule in op.get("rules", []):
                    when = rule.get("when", "")
                    then = rule.get("then", "")
                    else_rule = rule.get("else")

                    # Handle else clause (no condition to evaluate)
                    # else_rule IS the verdict value (e.g., {"else": "Low Risk"})
                    if else_rule is not None:
                        if str(else_rule).lower() == verdict.lower():
                            return f"else → {verdict}"
                        continue  # else clause doesn't match this verdict

                    # Case-insensitive verdict comparison
                    if str(then).lower() != verdict.lower() or not when:
                        continue

                    # Actually evaluate the condition to verify it's true
                    condition_true = self._evaluate_rule_condition(when)
                    if condition_true:
                        actual_values = self._extract_variables_from_condition(when)
                        actual_str = ", ".join(f"{k}={v}" for k, v in actual_values.items())
                        return f"{when} (actual: {actual_str}) → {verdict}"

                # No rule verified - return default (may happen if seed format differs)
                logger.debug(
                    f"No scoring rule matched for verdict '{verdict}'. "
                    f"Score: {score}, Namespace: {self._namespace}"
                )

        # Default description
        return f"SCORE = {score} → {verdict}"

    def _get_triggered_rule_count(self, count: int, verdict: str) -> str:
        """Get triggered rule description for count-based policies.

        SPECIFICATION ENFORCEMENT: Actually evaluates each rule condition
        against computed namespace values, reporting which rule ACTUALLY
        triggered (not just text-matching the verdict).

        Args:
            count: The incident count
            verdict: The verdict

        Returns:
            Rule description string with actual computed values
        """
        # Find the case operation that produces VERDICT
        for op in self.seed.get("compute", []):
            if op.get("name") == "VERDICT" and op.get("op") == "case":
                for rule in op.get("rules", []):
                    when = rule.get("when", "")
                    then = rule.get("then", "")
                    else_rule = rule.get("else")

                    # Handle else clause (no condition to evaluate)
                    # else_rule IS the verdict value (e.g., {"else": "Low Risk"})
                    if else_rule is not None:
                        if str(else_rule).lower() == verdict.lower():
                            return f"else → {verdict}"
                        continue  # else clause doesn't match this verdict

                    # Case-insensitive verdict comparison
                    if str(then).lower() != verdict.lower() or not when:
                        continue

                    # Actually evaluate the condition to verify it's true
                    condition_true = self._evaluate_rule_condition(when)
                    if condition_true:
                        # Include actual computed value in the output
                        actual_values = self._extract_variables_from_condition(when)
                        actual_str = ", ".join(f"{k}={v}" for k, v in actual_values.items())
                        return f"{when} (actual: {actual_str}) → {verdict}"

                # No rule matched - return default (may happen if seed format differs)
                logger.debug(
                    f"No count rule matched for verdict '{verdict}'. "
                    f"Namespace: {self._namespace}"
                )

        # Default description
        return f"N_INCIDENTS = {count} → {verdict}"

    def _evaluate_rule_condition(self, condition: str) -> bool:
        """Evaluate a rule condition string against the namespace.

        Handles conditions like:
        - "N_SEVERE_FIRSTHAND >= 1"
        - "N_MODERATE_FIRSTHAND >= 2"
        - "SCORE >= 15"
        - "N_SEVERE >= 1 or N_MODERATE_FIRSTHAND >= 2" (compound)

        Args:
            condition: Condition string from rule's "when" field

        Returns:
            True if condition evaluates to True
        """
        if not condition:
            return False

        # Use SafeExpressionExecutor for robust evaluation of compound expressions
        try:
            result = self._executor.execute_bool(condition, self._namespace, default=None)
            if result is not None:
                return result
        except Exception:
            pass

        # Fallback: Parse simple condition VAR OP VALUE
        pattern = r"(\w+)\s*(>=|<=|>|<|==|=)\s*(\d+(?:\.\d+)?)"
        match = re.match(pattern, condition.strip())
        if not match:
            return False

        var_name, op, threshold_str = match.groups()
        threshold = float(threshold_str)
        actual = self._namespace.get(var_name, 0)

        if op in (">=",):
            return actual >= threshold
        elif op in ("<=",):
            return actual <= threshold
        elif op in (">",):
            return actual > threshold
        elif op in ("<",):
            return actual < threshold
        elif op in ("==", "="):
            return actual == threshold

        return False

    def _extract_variables_from_condition(self, condition: str) -> Dict[str, Any]:
        """Extract variable names from condition and return their actual values.

        Args:
            condition: Condition string like "N_SEVERE_FIRSTHAND >= 1"

        Returns:
            Dict of {variable_name: actual_value}
        """
        result = {}
        # Find all variable references (word characters that look like variable names)
        var_pattern = r"\b([A-Z][A-Z0-9_]*)\b"
        matches = re.findall(var_pattern, condition)
        for var_name in matches:
            if var_name in self._namespace:
                result[var_name] = self._namespace[var_name]
        return result

    def _generate_reasoning(
        self, score: float, verdict: str, n_incidents: int, is_scoring_based: bool
    ) -> str:
        """Generate reasoning text for the justification.

        Args:
            score: The computed score or count
            verdict: The verdict
            n_incidents: Number of incidents found
            is_scoring_based: Whether this is a scoring-based policy

        Returns:
            Reasoning string (1-3 sentences)
        """
        if n_incidents == 0:
            return f"No relevant incidents found in the reviews. Verdict: {verdict}."

        if is_scoring_based:
            return (
                f"Found {n_incidents} incident(s) totaling {score} points. "
                f"Based on the scoring thresholds, this results in verdict: {verdict}."
            )
        else:
            return (
                f"Found {n_incidents} incident(s). "
                f"Based on the count thresholds, this results in verdict: {verdict}."
            )

    def get_standard_output(self) -> Dict[str, Any]:
        """Get output in standard format (matching output_schema.txt).

        Should be called after execute() to get the standardized output.
        This returns ONLY schema-compliant fields (no metadata).

        Returns:
            Dict with verdict, evidences, justification, other_notes
        """
        return self._build_standard_output()

    def get_debug_metadata(self) -> Dict[str, Any]:
        """Get AMOS-specific debug metadata (separate from standard output).

        Returns:
            Dict with extractions_count, namespace
        """
        return {
            "extractions_count": len(self._extractions),
            "namespace": self._namespace,
        }

    def get_usage_metrics(self) -> Dict[str, Any]:
        """Get aggregated usage metrics from all LLM calls.

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens,
            cost_usd, wall_clock_ms, llm_calls
        """
        if not self._usage_records:
            metrics = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "wall_clock_ms": 0.0,
                "llm_calls": 0,
            }
        else:
            prompt_tokens = sum(u.get("prompt_tokens", 0) for u in self._usage_records)
            completion_tokens = sum(u.get("completion_tokens", 0) for u in self._usage_records)
            cost_usd = sum(u.get("cost_usd", 0.0) for u in self._usage_records)

            metrics = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": cost_usd,
                "wall_clock_ms": self._wall_clock_ms,
                "llm_calls": len(self._usage_records),
            }

        return metrics

    async def execute(
        self,
        reviews: List[Dict[str, Any]],
        business: Dict[str, Any],
        query: str = "",
        sample_id: str = "",
    ) -> Dict[str, Any]:
        """Execute Formula Seed against restaurant data.

        Processes ALL reviews via per-review LLM calls, extracts signals,
        then computes verdict from aggregated extractions.

        Args:
            reviews: List of review dicts with 'text' and 'review_id' fields
            business: Restaurant business info dict
            query: The agenda/query text (unused, kept for API compatibility)
            sample_id: Sample ID for caching

        Returns:
            Dict with output values and observability data
        """
        start_time = time.perf_counter()

        self._total_reviews = len(reviews)
        self._current_sample_id = sample_id

        # Verbose output: Starting Phase 2
        biz_name = business.get("name", sample_id[:12])
        output.status(f"  [Phase 2] {biz_name} | {len(reviews)} reviews")

        # ===== CHECK FOR ABSTAIN (validation failed in Phase 1) =====
        if self.seed.get("_abstain"):
            output.warn(f"  [Phase 2] ABSTAIN: {self.seed.get('_abstain_reason', 'unknown')[:50]}")
            logger.warning(
                f"Seed marked for ABSTAIN due to validation failure: "
                f"{self.seed.get('_abstain_reason', 'unknown')}"
            )
            self._wall_clock_ms = (time.perf_counter() - start_time) * 1000
            return {
                "VERDICT": "ABSTAIN",
                "_abstain": True,
                "_abstain_reason": self.seed.get("_abstain_reason", []),
                "_abstain_warnings": self.seed.get("_abstain_warnings", []),
                "_extractions": [],
                "_namespace": {"VERDICT": "ABSTAIN"},
                "_stats": {
                    "total_reviews": len(reviews),
                    "abstained": True,
                    "abstain_reason": self.seed.get("_abstain_reason", []),
                    "wall_clock_ms": self._wall_clock_ms,
                },
            }

        # ===== EXTRACT SIGNALS FROM ALL REVIEWS =====
        # Limit to max_reviews if configured
        reviews_to_process = reviews
        if len(reviews) > self.config.max_reviews:
            # Prioritize by recency (sort by date descending)
            reviews_to_process = sorted(
                reviews,
                key=lambda r: r.get("date", ""),
                reverse=True
            )[:self.config.max_reviews]

        output.status(f"  [Phase 2] Extracting signals from {len(reviews_to_process)} reviews...")
        await self._extract_signals(reviews_to_process)

        # ===== FILTER TO INCIDENTS ONLY =====
        # SPECIFICATION ENFORCEMENT: Only extractions with actual incidents
        # (outcome != "none") should contribute to verdict-affecting counts.
        # This prevents non-incident extractions from triggering verdicts.
        n_filtered = self._filter_to_incidents_only()
        if n_filtered > 0:
            output.status(
                f"  [Phase 2] Filtered {n_filtered} non-incident extractions "
                f"({len(self._extractions)} remaining)"
            )

        # Compute verdict from extractions
        self._execute_compute(business)

        # Capture results
        verdict = self._namespace.get("VERDICT")
        extractions_count = len(self._extractions)
        output.status(f"  [Phase 2] {extractions_count} relevant extractions -> verdict={verdict}")

        # Track wall-clock time
        self._wall_clock_ms = (time.perf_counter() - start_time) * 1000

        # Return final output
        result = self._get_output()

        result["_stats"] = {
            "total_reviews": len(reviews),
            "reviews_processed": len(reviews_to_process),
            "extractions_count": extractions_count,
            "extractions_filtered_non_incident": n_filtered,
            "final_verdict": verdict,
            "wall_clock_ms": self._wall_clock_ms,
            "snippet_validation": self._snippet_validation_stats,
        }

        return result
