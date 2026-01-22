"""Phase 2: Formula Seed Interpreter.

Executes the Formula Seed against restaurant data with two-stage retrieval:

Stage 1 (Quick Scan):
- Filter reviews using filter_mode (keyword, embedding, or hybrid)
- Extract signals via LLM
- Compute verdict and check for early exit (severe evidence found)

Stage 2 (Thorough Sweep):
- Process ALL remaining reviews not matched in Stage 1
- Recompute final verdict with complete evidence
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from addm.llm import LLMService
from addm.methods.amos.config import AMOSConfig, FilterMode
from addm.methods.amos.search.executor import SafeExpressionExecutor
from addm.methods.amos.search.embeddings import HybridRetriever
from addm.methods.amos.seed_transform import transform_formula_seed
from addm.utils.async_utils import gather_with_concurrency

logger = logging.getLogger(__name__)


# =============================================================================
# Enum Value Normalization
# =============================================================================

# Common variations that map to canonical enum values
_SEVERITY_NORMALIZATION_MAP = {
    "none": "none", "no": "none", "n/a": "none", "na": "none",
    "not applicable": "none", "no incident": "none", "no reaction": "none",
    "no allergy": "none", "no allergy incident": "none", "no allergic reaction": "none",
    "no relevant incident": "none", "not relevant": "none", "irrelevant": "none",
    "no issue": "none", "no issues": "none", "no problem": "none", "no problems": "none",
    "nothing": "none", "absent": "none", "no symptoms": "none",
    "mild": "mild", "minor": "mild", "slight": "mild", "minimal": "mild", "low": "mild",
    "mild incident": "mild", "mild reaction": "mild", "minor incident": "mild",
    "minor reaction": "mild", "discomfort": "mild",
    "moderate": "moderate", "medium": "moderate", "moderate incident": "moderate",
    "moderate reaction": "moderate", "some symptoms": "moderate", "noticeable": "moderate",
    "severe": "severe", "serious": "severe", "critical": "severe", "high": "severe",
    "extreme": "severe", "severe incident": "severe", "severe reaction": "severe",
    "life-threatening": "severe", "life threatening": "severe", "emergency": "severe",
    "anaphylaxis": "severe", "anaphylactic": "severe", "hospitalization": "severe",
    "hospital": "severe",
}


def _normalize_enum_value(value: Any, expected_values: List[str]) -> Any:
    """Normalize an enum value to match expected canonical values.

    Handles common variations in LLM outputs:
    - "no reaction" -> "none"
    - "no allergy incident described" -> "none"
    - Keyword-based fuzzy matching for severity fields

    Args:
        value: The value to normalize (from LLM extraction)
        expected_values: List of valid enum values (e.g., ["none", "mild", "moderate", "severe"])

    Returns:
        Normalized value if a match is found, otherwise original value
    """
    if value is None:
        return value

    value_str = str(value).strip().lower()

    # 1. Exact match
    for expected in expected_values:
        if value_str == expected.lower():
            return expected

    # 2. Check normalization map
    if value_str in _SEVERITY_NORMALIZATION_MAP:
        normalized = _SEVERITY_NORMALIZATION_MAP[value_str]
        for expected in expected_values:
            if normalized == expected.lower():
                return expected

    # 3. Keyword-based matching for severity enums
    expected_lower = {e.lower() for e in expected_values}
    is_severity_enum = expected_lower & {"none", "mild", "moderate", "severe"}

    if is_severity_enum:
        none_indicators = ["no ", "none", "n/a", "not ", "absent", "nothing", "irrelevant"]
        severity_keywords = ["mild", "moderate", "severe", "serious", "critical", "emergency", "life"]

        has_none_indicator = any(ind in value_str for ind in none_indicators)
        has_severity_keyword = any(kw in value_str for kw in severity_keywords)

        if has_none_indicator and not has_severity_keyword:
            if "none" in expected_lower:
                return "none"

        if "life-threatening" in value_str or "life threatening" in value_str:
            if "severe" in expected_lower:
                return "severe"
        if "anaphyla" in value_str:
            if "severe" in expected_lower:
                return "severe"
        if "emergency" in value_str or "hospital" in value_str:
            if "severe" in expected_lower:
                return "severe"
        if "severe" in value_str or "serious" in value_str or "critical" in value_str:
            if "severe" in expected_lower:
                return "severe"
        if "moderate" in value_str or "medium" in value_str:
            if "moderate" in expected_lower:
                return "moderate"
        if "mild" in value_str or "minor" in value_str or "slight" in value_str:
            if "mild" in expected_lower:
                return "mild"

    return value


def _build_extraction_prompt(
    fields: List[Dict[str, Any]],
    review_text: str,
    review_id: str,
    task_name: str = "relevant information",
    extraction_guidelines: Optional[str] = None,
) -> str:
    """Build extraction prompt from Formula Seed field definitions.

    Args:
        fields: List of field definitions from Formula Seed
        review_text: The review text to analyze
        review_id: Review identifier
        task_name: Human-readable task description (e.g., "allergy safety", "romantic dining")
        extraction_guidelines: Optional task-specific extraction guidelines

    Returns:
        Extraction prompt string
    """
    field_defs = []
    field_names = []
    for field in fields:
        name = field["name"]
        field_names.append(name)
        ftype = field.get("type", "enum")
        values = field.get("values", {})

        if ftype == "enum" and values:
            value_list = ", ".join(values.keys())
            field_defs.append(f"{name.upper()} // MUST be exactly one of: {{{value_list}}}")
            field_defs.append("  IMPORTANT: Use ONLY these exact values. Do NOT paraphrase or use synonyms.")
            for value, description in values.items():
                field_defs.append(f'  - "{value}": {description}')
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

    # Build explicit field list for output constraint
    allowed_fields_list = ", ".join(f'"{name}"' for name in field_names)

    # Build output format with example values from enum definitions
    output_field_examples = []
    for field in fields:
        name = field["name"]
        ftype = field.get("type", "enum")
        values = field.get("values", {})
        if ftype == "enum" and values:
            # Use first enum value as example
            example_value = list(values.keys())[0]
            output_field_examples.append(f'"{name}": "{example_value}"')
        else:
            output_field_examples.append(f'"{name}": <value>')
    output_fields = ", ".join(output_field_examples)

    # Use task-specific guidelines if provided, otherwise generate generic ones
    if extraction_guidelines:
        guidelines_section = f"EXTRACTION GUIDELINES:\n{extraction_guidelines}"
    else:
        guidelines_section = """EXTRACTION GUIDELINES:
- Only extract fields that are explicitly mentioned or strongly implied in the review
- If a field is not discussed or cannot be determined, use the default/none value
- Focus on factual content, not inferences"""

    prompt = f"""Extract the following fields from this review for {task_name}.

{guidelines_section}

STRICT OUTPUT RULES:
1. ONLY output the fields listed below. Do NOT add any additional fields (no "staff_response", "cuisine_risk", "additional_context", etc.).
2. For enum fields, you MUST use EXACTLY one of the listed values. Do NOT paraphrase or use synonyms (e.g., use "none" not "no reaction" or "no incident").
3. The ONLY allowed fields in your output are: "review_id", "is_relevant", "supporting_quote", {allowed_fields_list}

FIELD DEFINITIONS:
{field_definitions}

REVIEW TEXT:
{review_text}

WHEN TO SET is_relevant: false
- If the review does NOT contain any ACTUAL evidence related to {task_name}
- If the review only mentions ingredients, menu items, or general topics without describing an actual incident or relevant experience
- If you cannot find a direct quote that demonstrates evidence for the task
- When in doubt, set is_relevant: false

WHEN TO SET is_relevant: true
- ONLY if the review contains ACTUAL evidence (a real incident, experience, or observation directly related to {task_name})
- You must be able to quote specific text that demonstrates the evidence

CRITICAL - EVIDENCE REQUIREMENT:
You MUST include a "supporting_quote" field with the EXACT text from the review that supports your extraction.
- Copy the relevant sentence(s) verbatim from the review
- The quote must exist in the review text exactly as written
- If you cannot quote specific evidence, set is_relevant: false

Output JSON only. If the review does not contain relevant evidence for this task, output:
{{"is_relevant": false}}

Otherwise output ONLY these fields (no extra fields):
{{
  "review_id": "{review_id}",
  "is_relevant": true,
  "supporting_quote": "<EXACT text from review>",
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
        max_concurrent: int = 256,
        config: Optional[AMOSConfig] = None,
    ):
        """Initialize interpreter.

        Args:
            seed: Formula Seed specification
            llm: LLM service for extraction calls
            max_concurrent: Max concurrent LLM calls for extraction
            config: AMOS configuration (controls adaptive mode, hybrid retrieval, etc.)
        """
        # Transform seed to ensure phase2-compatible format
        # (handles LLM-generated SQL expressions → op/expr/where format)
        self.seed = transform_formula_seed(seed)
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

        # Thorough sweep state
        self._sweep_early_stopped: bool = False

        # Snippet validation stats
        self._snippet_validation_stats: Dict[str, int] = {
            "total_relevant": 0,
            "valid_quotes": 0,
            "rejected_no_quote": 0,
            "rejected_quote_not_found": 0,
        }

    def _validate_and_store_snippet(
        self,
        extraction: Dict[str, Any],
        quote: str,
        review_text: str,
    ) -> Dict[str, Any]:
        """Validate that the supporting quote exists in the review text.

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

        # Normalize both texts for comparison (lowercase, collapse whitespace)
        quote_normalized = " ".join(quote.lower().split())
        review_normalized = " ".join(review_text.lower().split())

        # Check if normalized quote exists in normalized review
        # Use fuzzy matching: allow for minor differences (punctuation, etc.)
        if quote_normalized in review_normalized:
            # Quote found - valid extraction
            extraction["_snippet"] = quote
            extraction["_snippet_validated"] = True
            self._snippet_validation_stats["valid_quotes"] += 1
            return extraction

        # Try partial matching (at least 80% of quote words must be in review)
        quote_words = set(quote_normalized.split())
        if len(quote_words) >= 3:
            review_words = set(review_normalized.split())
            overlap = len(quote_words & review_words) / len(quote_words)
            if overlap >= 0.8:
                # Close enough match
                extraction["_snippet"] = quote
                extraction["_snippet_validated"] = True
                extraction["_snippet_match_quality"] = f"{overlap:.0%}"
                self._snippet_validation_stats["valid_quotes"] += 1
                return extraction

        # Quote not found - reject extraction
        extraction["is_relevant"] = False
        extraction["_rejected_reason"] = "quote_not_found"
        extraction["_attempted_quote"] = quote[:100]  # Store for debugging
        self._snippet_validation_stats["rejected_quote_not_found"] += 1
        logger.debug(
            f"Rejection: quote not found in review {extraction.get('review_id')}: "
            f"'{quote[:50]}...'"
        )
        return extraction

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

        extraction = _parse_extraction_response(response)
        extraction["review_id"] = review_id

        # Validate supporting_quote exists in source text
        if extraction.get("is_relevant", False):
            quote = extraction.get("supporting_quote", "")
            extraction = self._validate_and_store_snippet(extraction, quote, review_text)

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
            # Support both canonical 'op' and legacy 'operation' keys
            op = op_def.get("op") or op_def.get("operation", "")

            # Only run simple aggregations, not verdict computation
            if op == "count":
                self._namespace[name] = self._compute_count(op_def)
            elif op == "sum":
                self._namespace[name] = self._compute_sum(op_def)
            elif op == "expr":
                self._namespace[name] = self._compute_expr(op_def)

    def _should_early_exit(self) -> bool:
        """Check if Quick Scan found sufficient severe evidence for early exit.

        Uses Formula Seed's verdict_metadata to determine severity if available.
        Falls back to stopping_condition from search_strategy.
        Policy-agnostic: works for G1-G6 with different verdict labels.

        Returns:
            True if early exit is warranted (severe evidence found)
        """
        # Get verdict metadata from Formula Seed (generated in Phase 1)
        metadata = self.seed.get("verdict_metadata", {})
        severe_verdicts = metadata.get("severe_verdicts", [])

        # If severe_verdicts defined, check if current verdict is severe
        if severe_verdicts:
            verdict = self._namespace.get("VERDICT")
            return verdict in severe_verdicts

        # Fall back to existing stopping_condition logic from search_strategy
        stopping_expr = self.strategy.get("stopping_condition", "False")
        context = {
            "namespace": self._namespace,
            **self._namespace,
        }
        return self._executor.execute_bool(stopping_expr, context, default=False)

    async def _sweep_remaining_reviews(
        self,
        unprocessed: List[Dict[str, Any]],
    ) -> None:
        """Thorough sweep of remaining reviews not matched by keywords.

        Processes ALL unprocessed reviews in parallel, adding any relevant extractions
        to self._extractions.

        Args:
            unprocessed: List of reviews that weren't matched by keyword filter
        """
        if not unprocessed:
            return

        fields = self.seed.get("extract", {}).get("fields", [])
        if not fields:
            return

        # Limit sweep size
        max_reviews = self.config.max_sweep_reviews
        if len(unprocessed) > max_reviews:
            # Prioritize by recency (sort by date descending)
            unprocessed = sorted(
                unprocessed,
                key=lambda r: r.get("date", ""),
                reverse=True
            )[:max_reviews]

        # Run ALL reviews in parallel (no batching for maximum speed)
        tasks = [self._extract_single_review(r, fields) for r in unprocessed]
        results = await gather_with_concurrency(self.max_concurrent, tasks)

        # Add relevant extractions
        for r in results:
            if r.get("is_relevant", False):
                self._extractions.append(r)

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
                    # Case-insensitive list matching for strings
                    if isinstance(actual, str):
                        if not any(
                            actual.lower() == e.lower() if isinstance(e, str) else actual == e
                            for e in expected
                        ):
                            return False
                    elif actual not in expected:
                        return False
                else:
                    # Case-insensitive comparison for strings
                    if isinstance(actual, str) and isinstance(expected, str):
                        if actual.lower() != expected.lower():
                            return False
                    elif actual != expected:
                        return False

            if "not_equals" in condition:
                not_expected = condition["not_equals"]
                if isinstance(not_expected, list):
                    # Case-insensitive list matching for strings
                    if isinstance(actual, str):
                        if any(
                            actual.lower() == e.lower() if isinstance(e, str) else actual == e
                            for e in not_expected
                        ):
                            return False
                    elif actual in not_expected:
                        return False
                else:
                    # Case-insensitive comparison for strings
                    if isinstance(actual, str) and isinstance(not_expected, str):
                        if actual.lower() == not_expected.lower():
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

            # Case-insensitive comparison for string values
            if isinstance(actual, str) and isinstance(expected, str):
                matches = actual.lower() == expected.lower()
            elif isinstance(expected, list):
                # Match any in list (case-insensitive for strings)
                if isinstance(actual, str):
                    matches = any(
                        actual.lower() == e.lower() if isinstance(e, str) else actual == e
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
        Uses case-insensitive comparison for string values.

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

            # Case-insensitive comparison for strings
            if isinstance(expected, list):
                if isinstance(actual, str):
                    return any(
                        actual.lower() == e.lower() if isinstance(e, str) else actual == e
                        for e in expected
                    )
                return actual in expected

            if isinstance(actual, str) and isinstance(expected, str):
                return actual.lower() == expected.lower()
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

            # Compare (case-insensitive, exact match for strings)
            # Use exact matching to avoid "Mild incident" matching when looking for "mild"
            if str(actual_value).lower().strip() == value.lower().strip():
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

        return {
            "verdict": self._namespace.get("VERDICT"),
            "evidences": evidences,
            "justification": justification,
            "other_notes": None,
        }

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

        Args:
            score: The computed score
            verdict: The verdict

        Returns:
            Rule description string
        """
        # Find the case operation that produces VERDICT
        for op in self.seed.get("compute", []):
            if op.get("name") == "VERDICT" and op.get("op") == "case":
                for rule in op.get("rules", []):
                    when = rule.get("when", "")
                    then = rule.get("then", "")
                    if then == verdict:
                        return f"{when} → {verdict}"

        # Default description
        return f"score = {score} → {verdict}"

    def _get_triggered_rule_count(self, count: int, verdict: str) -> str:
        """Get triggered rule description for count-based policies.

        Args:
            count: The incident count
            verdict: The verdict

        Returns:
            Rule description string
        """
        # Find the case operation that produces VERDICT
        for op in self.seed.get("compute", []):
            if op.get("name") == "VERDICT" and op.get("op") == "case":
                for rule in op.get("rules", []):
                    when = rule.get("when", "")
                    then = rule.get("then", "")
                    if then == verdict:
                        return f"{when} → {verdict}"

        # Default description
        return f"count = {count} → {verdict}"

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

        Returns:
            Dict with verdict, evidences, justification, other_notes
        """
        standard = self._build_standard_output()

        # Add AMOS-specific metadata (for debugging, not part of standard)
        standard["_amos_metadata"] = {
            "extractions_count": len(self._extractions),
            "stopped_early": self._stopped_early,
            "reviews_skipped": self._reviews_skipped,
            "namespace": self._namespace,
        }

        return standard

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
            "filter_mode": self.config.filter_mode.value,
            "early_exit": self._stopped_early,
            "sweep_early_stopped": self._sweep_early_stopped,
        }

        # Add embedding metrics if retriever was used
        if self._retriever:
            retriever_metrics = self._retriever.get_metrics()
            metrics["embedding_tokens"] = retriever_metrics.get("embedding_tokens", 0)
            metrics["embedding_cost_usd"] = retriever_metrics.get("embedding_cost_usd", 0.0)
            metrics["cost_usd"] += retriever_metrics.get("embedding_cost_usd", 0.0)

        return metrics

    async def _filter_by_mode(
        self,
        reviews: List[Dict[str, Any]],
        query: str,
        sample_id: str,
    ) -> List[Dict[str, Any]]:
        """Filter reviews for Stage 1 based on filter_mode.

        Args:
            reviews: All reviews to filter
            query: The agenda/query text (for embedding-based filtering)
            sample_id: Sample ID for embedding cache

        Returns:
            List of filtered reviews for Stage 1 extraction
        """
        from pathlib import Path

        if self.config.filter_mode == FilterMode.KEYWORD:
            # Keyword-only filtering
            return self._filter_reviews(reviews)

        elif self.config.filter_mode == FilterMode.EMBEDDING:
            # Embedding-only filtering (no keyword pre-filter)
            cache_path = None
            if self.config.embedding_cache_path:
                cache_path = Path(self.config.embedding_cache_path)

            self._retriever = HybridRetriever(
                cache_path=cache_path,
                embedding_model=self.config.embedding_model,
            )

            # Get top-k by embedding similarity (no keyword filter)
            return await self._retriever.retrieve_by_embedding(
                all_reviews=reviews,
                query=query,
                sample_id=sample_id,
                top_k=min(20, len(reviews)),  # Top 20 or all if fewer
            )

        elif self.config.filter_mode == FilterMode.HYBRID:
            # Keyword + embedding: union of both
            keyword_filtered = self._filter_reviews(reviews)

            cache_path = None
            if self.config.embedding_cache_path:
                cache_path = Path(self.config.embedding_cache_path)

            self._retriever = HybridRetriever(
                cache_path=cache_path,
                embedding_model=self.config.embedding_model,
            )

            # Get embedding matches not already in keyword set
            keyword_ids = {r.get("review_id") for r in keyword_filtered}
            remaining = [r for r in reviews if r.get("review_id") not in keyword_ids]

            if remaining:
                embedding_filtered = await self._retriever.retrieve_by_embedding(
                    all_reviews=remaining,
                    query=query,
                    sample_id=sample_id,
                    top_k=min(10, len(remaining)),  # Add top 10 from embedding
                )
                return keyword_filtered + embedding_filtered

            return keyword_filtered

        else:
            # Fallback to keyword
            return self._filter_reviews(reviews)

    async def execute(
        self,
        reviews: List[Dict[str, Any]],
        business: Dict[str, Any],
        query: str = "",
        sample_id: str = "",
    ) -> Dict[str, Any]:
        """Execute Formula Seed against restaurant data with two-stage retrieval.

        Stage 1 (Quick Scan): Filter using filter_mode, extract, check early exit
        Stage 2 (Thorough Sweep): Process ALL remaining reviews (always on)

        Args:
            reviews: List of review dicts with 'text' and 'review_id' fields
            business: Restaurant business info dict
            query: The agenda/query text (for embedding-based filtering)
            sample_id: Sample ID for caching (for embedding cache)

        Returns:
            Dict with output values and observability data
        """
        self._total_reviews = len(reviews)

        # ===== CHECK FOR ABSTAIN (validation failed in Phase 1) =====
        if self.seed.get("_abstain"):
            logger.warning(
                f"Seed marked for ABSTAIN due to validation failure: "
                f"{self.seed.get('_abstain_reason', 'unknown')}"
            )
            return {
                "VERDICT": "ABSTAIN",
                "_abstain": True,
                "_abstain_reason": self.seed.get("_abstain_reason", []),
                "_abstain_warnings": self.seed.get("_abstain_warnings", []),
                "_extractions": [],
                "_namespace": {"VERDICT": "ABSTAIN"},
                "_filter_stats": {
                    "filter_mode": self.config.filter_mode.value,
                    "total_reviews": len(reviews),
                    "abstained": True,
                    "abstain_reason": self.seed.get("_abstain_reason", []),
                },
            }

        # ===== STAGE 1: QUICK SCAN =====
        # Filter reviews based on filter_mode
        filtered = await self._filter_by_mode(reviews, query, sample_id)

        # Extract signals from filtered reviews
        await self._extract_signals(filtered)

        # Compute verdict from Stage 1 extractions
        self._execute_compute(business)

        # Capture Stage 1 results for stats
        stage1_verdict = self._namespace.get("VERDICT")
        stage1_extractions = len(self._extractions)

        # Check early exit (severe evidence found in Stage 1)
        if self._should_early_exit():
            logger.debug(f"Early exit: Stage 1 verdict '{stage1_verdict}' is severe")
            result = self._get_output()
            result["_filter_stats"] = {
                "filter_mode": self.config.filter_mode.value,
                "total_reviews": len(reviews),
                "stage1_filtered": len(filtered),
                "stage1_extractions": stage1_extractions,
                "stage1_verdict": stage1_verdict,
                "early_exit": True,
                "sweep_performed": False,
                "final_extractions": len(self._extractions),
                "final_verdict": stage1_verdict,
                "verdict_flipped": False,
                "snippet_validation": self._snippet_validation_stats,
            }
            return result

        # ===== STAGE 2: THOROUGH SWEEP (always on) =====
        filtered_ids = {r.get("review_id") for r in filtered}
        unprocessed = [r for r in reviews if r.get("review_id") not in filtered_ids]

        sweep_performed = False
        sweep_reviews_processed = 0
        stage2_extractions_added = 0

        if unprocessed:
            sweep_performed = True
            sweep_reviews_processed = min(len(unprocessed), self.config.max_sweep_reviews)

            logger.debug(
                f"Stage 2 sweep: {len(unprocessed)} unprocessed reviews, "
                f"processing up to {self.config.max_sweep_reviews}"
            )

            # Track Stage 2 extractions separately for verdict stabilization
            stage1_extraction_count = len(self._extractions)
            await self._sweep_remaining_reviews(unprocessed)
            stage2_extractions_added = len(self._extractions) - stage1_extraction_count

            # Recompute verdict with all extractions
            self._execute_compute(business)

            # Verdict stabilization: if Stage 2 would flip verdict, require strong evidence
            new_verdict = self._namespace.get("VERDICT")
            if new_verdict != stage1_verdict and stage2_extractions_added > 0:
                # Count Stage 2 validated extractions (those with _snippet_validated=True)
                stage2_validated = sum(
                    1 for e in self._extractions[stage1_extraction_count:]
                    if e.get("_snippet_validated", False)
                )

                # Require at least 2 validated Stage 2 extractions to flip verdict
                min_required = 2
                if stage2_validated < min_required:
                    logger.debug(
                        f"Verdict stabilization: Stage 2 would flip {stage1_verdict} → {new_verdict} "
                        f"but only {stage2_validated} validated extractions (need {min_required}). "
                        f"Reverting to Stage 1 extractions only."
                    )
                    # Revert to Stage 1 extractions only
                    self._extractions = self._extractions[:stage1_extraction_count]
                    self._execute_compute(business)
                    stage2_extractions_added = 0

        # Return final output
        result = self._get_output()

        result["_filter_stats"] = {
            "filter_mode": self.config.filter_mode.value,
            "total_reviews": len(reviews),
            "stage1_filtered": len(filtered),
            "stage1_extractions": stage1_extractions,
            "stage1_verdict": stage1_verdict,
            "early_exit": False,
            "sweep_performed": sweep_performed,
            "sweep_reviews_processed": sweep_reviews_processed,
            "sweep_early_stopped": self._sweep_early_stopped,
            "stage2_extractions_added": stage2_extractions_added,
            "final_extractions": len(self._extractions),
            "final_verdict": self._namespace.get("VERDICT"),
            "verdict_flipped": stage1_verdict != self._namespace.get("VERDICT"),
            "snippet_validation": self._snippet_validation_stats,
        }

        return result
