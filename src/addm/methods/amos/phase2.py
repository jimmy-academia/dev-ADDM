"""Phase 2: Formula Seed Interpreter.

Executes the Formula Seed against restaurant data:
- Process ALL reviews via per-review LLM calls (guaranteed coverage)
- Extract signals via LLM
- Compute verdict from extractions

Code organization:
- phase2_helpers.py: Validation and field helper functions
- phase2_compute.py: ComputeOperationsMixin (count, sum, case, expr, lookup)
- phase2_prompts.py: Prompt building and response parsing
"""

import logging
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# Type alias for progress callback
ProgressCallback = Callable[[int, str, float, str], None]

from addm.llm import LLMService
from addm.methods.amos.config import AMOSConfig
from addm.methods.amos.search.executor import SafeExpressionExecutor
from addm.methods.amos.phase2_prompts import (
    normalize_verdict_label as _normalize_verdict_label,
    build_extraction_prompt as _build_extraction_prompt,
    parse_extraction_response as _parse_extraction_response,
)
from addm.methods.amos.phase2_helpers import (
    get_field_value,
    validate_snippet,
    validate_enum_fields,
    get_outcome_field_info,
    is_none_value,
    matches_condition,
)
from addm.methods.amos.phase2_compute import ComputeOperationsMixin
from addm.utils.async_utils import gather_with_concurrency
from addm.utils.debug_logger import get_debug_logger
from addm.utils.output import output

logger = logging.getLogger(__name__)


class FormulaSeedInterpreter(ComputeOperationsMixin):
    """Executes Formula Seed against restaurant data.

    Formula Seed Structure (from Phase 1):
    ```
    {
      "task_name": "Allergy Risk Assessment",
      "extraction_guidelines": "...",
      "extract": {
        "fields": [
          {"name": "INCIDENT_SEVERITY", "type": "enum", "values": {...}},
          {"name": "ACCOUNT_TYPE", "type": "enum", "values": {...}},
        ],
        "outcome_field": "INCIDENT_SEVERITY",
        "none_values": ["none", "n/a"]
      },
      "compute": [
        {"name": "N_SEVERE", "op": "count", "where": {"INCIDENT_SEVERITY": "severe"}},
        {"name": "SCORE", "op": "sum", "expr": "CASE WHEN ... END"},
        {"name": "VERDICT", "op": "case", "source": "SCORE", "rules": [...]}
      ],
      "scoring": {
        "recency_rules": {"reference_date": "2022-01-01", "rules": [...]}
      },
      "output": ["VERDICT", "N_SEVERE", "SCORE"]
    }
    ```

    How Phase 2 uses the Formula Seed:
    1. extract.fields → Defines what to extract from each review (LLM prompt)
    2. extract.outcome_field → Which field indicates incident severity
    3. extract.none_values → Values meaning "no incident" (filtered out)
    4. compute → Operations to aggregate extractions into counts/scores/verdict
    5. scoring.recency_rules → V3 age-based weighting
    6. output → Which computed values to include in result
    """

    def __init__(
        self,
        seed: Dict[str, Any],
        llm: LLMService,
        max_concurrent: int = 256,
        config: Optional[AMOSConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """Initialize interpreter.

        Args:
            seed: Formula Seed specification (from Phase 1)
            llm: LLM service for extraction calls
            max_concurrent: Max concurrent LLM calls for extraction
            config: AMOS configuration
            progress_callback: Optional callback for progress updates
        """
        self.seed = seed
        self.llm = llm
        self.max_concurrent = max_concurrent
        self.config = config or AMOSConfig()
        self.progress_callback = progress_callback

        # Expression executor for evaluating rule conditions
        self._executor = SafeExpressionExecutor()

        # Cache reference date for V3 recency weighting
        self._reference_date: Optional[datetime] = self._parse_reference_date()

        # Runtime state
        self._usage_records: List[Dict[str, Any]] = []
        self._extractions: List[Dict[str, Any]] = []
        self._namespace: Dict[str, Any] = {}  # Computed values (N_*, SCORE, VERDICT)
        self._total_reviews: int = 0
        self._current_sample_id: str = ""
        self._wall_clock_ms: float = 0.0

        # Validation stats
        self._snippet_validation_stats: Dict[str, int] = {
            "total_relevant": 0,
            "valid_quotes": 0,
            "rejected_no_quote": 0,
            "rejected_quote_not_found": 0,
        }
        self._enum_validation_stats: Dict[str, Any] = {
            "total_validated": 0,
            "valid": 0,
            "rejected": 0,
            "rejection_details": [],
        }

    # =========================================================================
    # ENTRY POINT
    # =========================================================================

    async def execute(
        self,
        reviews: List[Dict[str, Any]],
        business: Dict[str, Any],
        query: str = "",
        sample_id: str = "",
    ) -> Dict[str, Any]:
        """Execute Formula Seed against restaurant data.

        This is the main entry point. The flow is:
        1. Check for ABSTAIN (if Phase 1 validation failed)
        2. Extract signals from all reviews via LLM
        3. Filter to incidents only (remove "none" outcomes)
        4. Execute compute operations (count, sum, case) → _namespace
        5. Return output with verdict and metadata

        Args:
            reviews: List of review dicts with 'text', 'review_id', 'date'
            business: Restaurant info dict
            query: Unused (kept for API compatibility)
            sample_id: Sample ID for logging

        Returns:
            Dict with VERDICT, _extractions, _namespace, _stats
        """
        start_time = time.perf_counter()
        self._total_reviews = len(reviews)
        self._current_sample_id = sample_id

        # Set debug logger context
        if debug_logger := get_debug_logger():
            debug_logger.set_context(sample_id)

        biz_name = business.get("name", sample_id[:12])
        output.status(f"  [Phase 2] {biz_name} | {len(reviews)} reviews")

        # ===== CHECK FOR ABSTAIN =====
        if self.seed.get("_abstain"):
            output.warn(f"  [Phase 2] ABSTAIN: {self.seed.get('_abstain_reason', 'unknown')[:50]}")
            self._wall_clock_ms = (time.perf_counter() - start_time) * 1000
            return {
                "VERDICT": "ABSTAIN",
                "_abstain": True,
                "_abstain_reason": self.seed.get("_abstain_reason", []),
                "_extractions": [],
                "_namespace": {"VERDICT": "ABSTAIN"},
                "_stats": {"total_reviews": len(reviews), "abstained": True},
            }

        # ===== EXTRACT SIGNALS =====
        # Uses seed.extract.fields to build LLM prompt
        reviews_to_process = reviews
        if len(reviews) > self.config.max_reviews:
            reviews_to_process = sorted(
                reviews, key=lambda r: r.get("date", ""), reverse=True
            )[:self.config.max_reviews]

        self._report_progress("extracting", 50, f"0/{len(reviews_to_process)} reviews")
        output.status(f"  [Phase 2] Extracting signals from {len(reviews_to_process)} reviews...")
        await self._extract_signals(reviews_to_process)

        # ===== FILTER TO INCIDENTS ONLY =====
        # Uses seed.extract.outcome_field and seed.extract.none_values
        n_filtered = self._filter_to_incidents_only()
        if n_filtered > 0:
            output.status(f"  [Phase 2] Filtered {n_filtered} non-incidents")

        # ===== COMPUTE VERDICT =====
        # Uses seed.compute operations (count, sum, case, expr, lookup)
        self._report_progress("computing", 96, "aggregating")
        self._execute_compute(business)  # From ComputeOperationsMixin

        verdict = self._namespace.get("VERDICT")
        output.status(f"  [Phase 2] {len(self._extractions)} extractions -> verdict={verdict}")
        self._report_progress("done", 100, f"verdict={verdict}")

        self._wall_clock_ms = (time.perf_counter() - start_time) * 1000

        # ===== BUILD OUTPUT =====
        # Uses seed.output to select which computed values to include
        result = self._get_output()
        result["_stats"] = {
            "total_reviews": len(reviews),
            "reviews_processed": len(reviews_to_process),
            "extractions_count": len(self._extractions),
            "extractions_filtered_non_incident": n_filtered,
            "final_verdict": verdict,
            "wall_clock_ms": self._wall_clock_ms,
            "snippet_validation": self._snippet_validation_stats,
        }
        return result

    # =========================================================================
    # EXTRACTION (uses seed.extract.fields)
    # =========================================================================

    async def _extract_signals(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract signals from all reviews via per-review LLM calls."""
        fields = self.seed.get("extract", {}).get("fields", [])
        if not fields or not reviews:
            return []

        total = len(reviews)
        completed = 0

        async def extract_with_progress(review: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal completed
            result = await self._extract_single_review(review, fields)
            completed += 1
            progress = 50 + (completed / total) * 45
            self._report_progress("extracting", progress, f"{completed}/{total} reviews")
            return result

        tasks = [extract_with_progress(r) for r in reviews]
        results = await gather_with_concurrency(self.max_concurrent, tasks)

        relevant = [r for r in results if r.get("is_relevant", False)]
        self._extractions = relevant
        return relevant

    async def _extract_single_review(
        self,
        review: Dict[str, Any],
        fields: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract signals from a single review via LLM.

        Uses:
        - seed.extract.fields → LLM prompt for what to extract
        - seed.task_name → Context for extraction
        - seed.extraction_guidelines → Instructions for LLM
        """
        review_id = review.get("review_id", "unknown")
        review_text = review.get("text", "")

        # Filter out temporal fields (populated from metadata, not LLM)
        temporal_fields = {"REVIEW_DATE", "AGE_YEARS"}
        llm_fields = [f for f in fields if f.get("name") not in temporal_fields]

        # Build prompt using seed context
        prompt = _build_extraction_prompt(
            llm_fields,
            review_text,
            review_id,
            task_name=self.seed.get("task_name", "relevant information"),
            extraction_guidelines=self.seed.get("extraction_guidelines"),
        )

        response, usage = await self.llm.call_async_with_usage(
            [{"role": "user", "content": prompt}],
            context={"sample_id": self._current_sample_id, "phase": "phase2_extract"},
        )
        self._usage_records.append(usage)

        extraction = _parse_extraction_response(response)
        extraction["review_id"] = review_id

        # Validate quote exists in source text
        if extraction.get("is_relevant", False):
            quote = extraction.get("supporting_quote", "")
            extraction = validate_snippet(
                extraction, quote, review_text, self._snippet_validation_stats
            )

        # Validate enum fields match seed.extract.fields.values
        if extraction.get("is_relevant", False):
            fields_def = self.seed.get("extract", {}).get("fields", [])
            extraction = validate_enum_fields(
                extraction, fields_def, self._enum_validation_stats
            )

        # Populate AGE_YEARS for V3 recency weighting
        self._populate_temporal_fields(extraction, review, fields)

        return extraction

    def _populate_temporal_fields(
        self,
        extraction: Dict[str, Any],
        review: Dict[str, Any],
        fields: List[Dict[str, Any]],
    ) -> None:
        """Populate REVIEW_DATE and AGE_YEARS from review metadata."""
        has_recency = bool(self.seed.get("scoring", {}).get("recency_rules", {}).get("rules"))
        needs_age = any(f.get("name") == "AGE_YEARS" for f in fields) or has_recency
        needs_date = any(f.get("name") == "REVIEW_DATE" for f in fields)

        if not (needs_age or needs_date):
            return

        review_date = review.get("date")
        if not review_date:
            return

        if isinstance(review_date, str):
            try:
                review_date_dt = datetime.fromisoformat(review_date.replace("Z", "+00:00"))
            except ValueError:
                return
        else:
            review_date_dt = review_date

        if needs_date:
            extraction["REVIEW_DATE"] = review_date_dt.strftime("%Y-%m-%d")

        if needs_age:
            ref_date = self._reference_date or datetime.now()
            if review_date_dt.tzinfo:
                review_date_dt = review_date_dt.replace(tzinfo=None)
            age_days = (ref_date - review_date_dt).days
            extraction["AGE_YEARS"] = age_days / 365.25

    def _filter_to_incidents_only(self) -> int:
        """Filter extractions to only include actual incidents.

        Uses seed.extract.outcome_field and seed.extract.none_values
        to determine which extractions represent "no incident".
        """
        outcome_field, none_values = get_outcome_field_info(self.seed)
        if not outcome_field:
            return 0

        original_count = len(self._extractions)
        self._extractions = [
            ext for ext in self._extractions
            if not is_none_value(get_field_value(ext, outcome_field), none_values)
        ]
        return original_count - len(self._extractions)

    # =========================================================================
    # COMPUTE (provided by ComputeOperationsMixin, uses seed.compute)
    # =========================================================================
    # _execute_compute() iterates through seed.compute operations:
    # - count: Count extractions matching where condition
    # - sum: Sum expression over extractions (with V3 recency weighting)
    # - case: Apply conditional rules to determine value
    # - expr: Evaluate mathematical expression
    # - lookup: Lookup value from business attributes

    # =========================================================================
    # OUTPUT BUILDING (uses seed.output)
    # =========================================================================

    def _get_output(self) -> Dict[str, Any]:
        """Get output values as specified in seed.output."""
        output_fields = self.seed.get("output", [])
        result = {}
        for field in output_fields:
            if field in self._namespace:
                result[field] = self._namespace[field]
        result["_extractions"] = self._extractions
        result["_namespace"] = self._namespace
        return result

    def _build_standard_output(self) -> Dict[str, Any]:
        """Transform to standard output format for evaluation."""
        evidences = []
        evidence_idx = 1
        outcome_field, none_values = get_outcome_field_info(self.seed)

        for ext in self._extractions:
            review_id = ext.get("review_id", "unknown")
            snippet = ext.get("_snippet") or ext.get("supporting_quote", "")

            if outcome_field:
                outcome_value = get_field_value(ext, outcome_field)
            else:
                outcome_value = (
                    get_field_value(ext, "incident_severity") or
                    get_field_value(ext, "severity") or "none"
                )

            if is_none_value(outcome_value, none_values):
                continue

            for field, value in ext.items():
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

        raw_verdict = self._namespace.get("VERDICT")
        return {
            "verdict": _normalize_verdict_label(raw_verdict),
            "evidences": evidences,
            "justification": self._build_justification(evidences),
            "computed_values": {k: v for k, v in self._namespace.items() if not k.startswith("_")},
            "validation_stats": {
                "enum_validation": self._enum_validation_stats,
                "snippet_validation": self._snippet_validation_stats,
            },
            "other_notes": None,
        }

    def _build_justification(self, evidences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build justification section from computed values."""
        score = self._namespace.get("SCORE") or self._namespace.get("RISK_SCORE", 0)
        verdict = self._namespace.get("VERDICT", "Unknown")
        is_scoring = "SCORE" in self._namespace or "INCIDENT_POINTS" in self._namespace

        outcome_field, _ = get_outcome_field_info(self.seed)
        outcome_names = {"incident_severity", "severity", "outcome"}
        if outcome_field:
            outcome_names.add(outcome_field.lower())

        breakdown = []
        for ev in evidences:
            if ev["field"] in outcome_names:
                points = self._get_points_for_severity(ev["judgement"])
                if points > 0:
                    breakdown.append({
                        "evidence_id": ev["evidence_id"],
                        "base_points": str(points),
                        "modifiers": [],
                        "subtotal": str(points),
                    })

        direct = [b["evidence_id"] for b in breakdown if int(b.get("subtotal", 0)) > 0][:5]
        triggered_rule = self._get_triggered_rule(verdict, score if is_scoring else len(breakdown), is_scoring)

        return {
            "triggered_rule": triggered_rule,
            "direct_evidence": direct,
            "scoring_trace": {"total_score": str(score), "breakdown": breakdown},
            "reasoning": self._generate_reasoning(score, verdict, len(breakdown), is_scoring),
        }

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_standard_output(self) -> Dict[str, Any]:
        """Get output in standard format (for evaluation)."""
        return self._build_standard_output()

    def get_debug_metadata(self) -> Dict[str, Any]:
        """Get debug metadata."""
        return {"extractions_count": len(self._extractions), "namespace": self._namespace}

    def get_usage_metrics(self) -> Dict[str, Any]:
        """Get aggregated usage metrics from all LLM calls."""
        if not self._usage_records:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    "cost_usd": 0.0, "wall_clock_ms": 0.0, "llm_calls": 0}

        prompt_tokens = sum(u.get("prompt_tokens", 0) for u in self._usage_records)
        completion_tokens = sum(u.get("completion_tokens", 0) for u in self._usage_records)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": sum(u.get("cost_usd", 0.0) for u in self._usage_records),
            "wall_clock_ms": self._wall_clock_ms,
            "llm_calls": len(self._usage_records),
        }

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _parse_reference_date(self) -> Optional[datetime]:
        """Parse reference date from seed.scoring.recency_rules."""
        recency_config = self.seed.get("scoring", {}).get("recency_rules", {})
        ref_date_str = recency_config.get("reference_date")
        if ref_date_str:
            try:
                return datetime.fromisoformat(ref_date_str)
            except ValueError:
                logger.warning(f"Invalid reference_date: {ref_date_str}")
        return None

    def _get_recency_weight(self, age_years: float) -> float:
        """Get recency weight multiplier for V3 policies.

        Uses seed.scoring.recency_rules.rules to determine weight.
        """
        rules = self.seed.get("scoring", {}).get("recency_rules", {}).get("rules", [])
        if not rules:
            return 1.0

        for rule in sorted(rules, key=lambda r: r.get("max_age_years", 999)):
            if age_years <= rule.get("max_age_years", 999):
                return rule.get("weight", 1.0)
        return 0.25

    def _report_progress(self, step: str, progress: float, detail: str = "") -> None:
        """Report progress via callback if configured."""
        if self.progress_callback:
            self.progress_callback(2, step, progress, detail)

    def _get_points_for_severity(self, severity: str) -> int:
        """Get point value from seed.compute case operations."""
        outcome_field, _ = get_outcome_field_info(self.seed)
        severity_lower = severity.lower()

        for op in self.seed.get("compute", []):
            if op.get("op") == "case":
                source = op.get("source", "").lower()
                if outcome_field and source == outcome_field.lower():
                    for rule in op.get("rules", []):
                        if rule.get("when", "").lower() == severity_lower:
                            try:
                                return int(rule.get("then", 0))
                            except (ValueError, TypeError):
                                pass
        return 0

    def _get_triggered_rule(self, verdict: str, value: float, is_scoring: bool) -> str:
        """Get triggered rule description from seed.compute VERDICT operation."""
        for op in self.seed.get("compute", []):
            if op.get("name") == "VERDICT" and op.get("op") == "case":
                for rule in op.get("rules", []):
                    when = rule.get("when", "")
                    then = rule.get("then", "")
                    else_rule = rule.get("else")

                    if else_rule is not None:
                        if str(else_rule).lower() == verdict.lower():
                            return f"else → {verdict}"
                        continue

                    if str(then).lower() != verdict.lower() or not when:
                        continue

                    if self._evaluate_rule_condition(when):
                        values = self._extract_variables_from_condition(when)
                        actual_str = ", ".join(f"{k}={v}" for k, v in values.items())
                        return f"{when} (actual: {actual_str}) → {verdict}"

        label = "SCORE" if is_scoring else "N_INCIDENTS"
        return f"{label} = {value} → {verdict}"

    def _evaluate_rule_condition(self, condition: str) -> bool:
        """Evaluate a rule condition against _namespace."""
        if not condition:
            return False

        try:
            result = self._executor.execute_bool(condition, self._namespace, default=None)
            if result is not None:
                return result
        except Exception:
            pass

        # Fallback: parse simple VAR OP VALUE
        match = re.match(r"(\w+)\s*(>=|<=|>|<|==|=)\s*(\d+(?:\.\d+)?)", condition.strip())
        if not match:
            return False

        var_name, op, threshold_str = match.groups()
        threshold = float(threshold_str)
        actual = self._namespace.get(var_name, 0)

        ops = {">=": lambda a, t: a >= t, "<=": lambda a, t: a <= t,
               ">": lambda a, t: a > t, "<": lambda a, t: a < t,
               "==": lambda a, t: a == t, "=": lambda a, t: a == t}
        return ops.get(op, lambda a, t: False)(actual, threshold)

    def _extract_variables_from_condition(self, condition: str) -> Dict[str, Any]:
        """Extract variable values from condition string."""
        result = {}
        for var_name in re.findall(r"\b([A-Z][A-Z0-9_]*)\b", condition):
            if var_name in self._namespace:
                result[var_name] = self._namespace[var_name]
        return result

    def _generate_reasoning(self, score: float, verdict: str, n_incidents: int, is_scoring: bool) -> str:
        """Generate reasoning text."""
        if n_incidents == 0:
            return f"No relevant incidents found. Verdict: {verdict}."
        if is_scoring:
            return f"Found {n_incidents} incident(s) totaling {score} points → {verdict}."
        return f"Found {n_incidents} incident(s) → {verdict}."
