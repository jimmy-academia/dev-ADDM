"""Recursive LLM (RLM) method for restaurant assessment.

RLM stores context as a Python variable (not in the prompt) and allows
the LLM to write Python code to adaptively explore/search the context.
This helps with "needle in haystack" problems where too much context
dilutes the signal.

Key difference from direct baseline:
- Direct: Sends all K reviews in prompt → context rot
- RLM: Sends only instructions → LLM decides what to examine

Reference: https://github.com/ysz/recursive-llm
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rlm import RLM

from addm.data.types import Sample
from addm.llm import LLMService
from addm.methods.base import Method
from addm.utils.debug_logger import get_debug_logger


def _ensure_api_key():
    """Ensure API key is set for litellm (used by RLM)."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    # Look for .openaiapi at Station directory level
    key_file = Path(__file__).parent.parent.parent.parent / ".openaiapi"
    if key_file.exists():
        key = key_file.read_text().strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key


# Load API key on module import
_ensure_api_key()


def _format_reviews_context(restaurant: Dict[str, Any]) -> str:
    """Format restaurant reviews as searchable indexed text.

    Args:
        restaurant: Restaurant dict with 'reviews' list

    Returns:
        Formatted string with indexed reviews for RLM to search
    """
    reviews = restaurant.get("reviews", [])
    lines = []
    for i, r in enumerate(reviews):
        review_id = r.get("review_id", f"R{i}")
        text = r.get("text", "").strip()
        stars = r.get("stars", "?")
        date = r.get("date", "unknown")
        lines.append(f"REVIEW_{i} [id={review_id}, stars={stars}, date={date}]:\n{text}")
    return "\n\n" + "=" * 40 + "\n\n".join(lines)


def _format_sample_context(sample: Sample) -> str:
    """Format sample context for RLM.

    Args:
        sample: Sample with context (str or structured data)

    Returns:
        Formatted string for RLM to search
    """
    context = sample.context or ""
    if isinstance(context, str):
        return context
    return str(context)


RLM_QUERY_TEMPLATE = """{agenda}

## Your Task

Analyze the reviews stored in the `context` variable to determine the verdict.

Use Python code to:
1. Search for relevant keywords (e.g., `re.search(r'allerg|reaction|epipen', review, re.I)`)
2. Extract and examine relevant review snippets
3. Apply the scoring rules from the agenda above
4. Call `FINAL(result)` with your JSON assessment when done

## Output Format (REQUIRED)

When you reach a conclusion, call FINAL with a JSON object matching this exact structure:

```python
FINAL({{
    "verdict": "<one of the verdict options from the agenda>",
    "evidences": [
        {{
            "evidence_id": "E1",
            "review_id": "<id of the source review>",
            "field": "<the term/field this evidence relates to>",
            "judgement": "<your classification for this field>",
            "snippet": "<verbatim quote from the review>"
        }}
    ],
    "justification": {{
        "triggered_rule": "<which verdict rule was triggered>",
        "direct_evidence": ["E1", "E2"],
        "scoring_trace": {{
            "total_score": "<numeric total score>",
            "breakdown": [
                {{
                    "evidence_id": "E1",
                    "base_points": "<points from severity>",
                    "modifiers": ["<modifier name if applicable>"],
                    "subtotal": "<points for this incident>"
                }}
            ]
        }},
        "reasoning": "<1-3 sentences explaining how the evidence leads to the verdict>"
    }},
    "other_notes": null
}})
```

## Available Context

The `context` variable contains all {num_reviews} reviews. You can:
- Slice: `context[:1000]` to see first 1000 chars
- Search: `re.findall(r'REVIEW_\\d+.*?(?=REVIEW_|$)', context, re.DOTALL)` to get individual reviews
- Filter: Loop through reviews and check for keywords

Start by exploring the context to understand its structure, then search for relevant evidence.
"""


class RLMMethod(Method):
    """Recursive LLM method - uses code execution to explore context."""

    name = "rlm"

    def __init__(
        self,
        model: str = "gpt-5-nano",
        recursive_model: Optional[str] = None,
        max_depth: int = 3,
        max_iterations: int = 15,
    ):
        """Initialize RLM method.

        Args:
            model: Primary LLM model for decisions
            recursive_model: Model for recursive calls (defaults to model)
            max_depth: Maximum recursion depth
            max_iterations: Maximum REPL iterations
        """
        self.model = model
        self.recursive_model = recursive_model or model
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self._rlm: Optional[RLM] = None

    def _get_rlm(self) -> RLM:
        """Lazy initialization of RLM instance."""
        if self._rlm is None:
            self._rlm = RLM(
                model=self.model,
                recursive_model=self.recursive_model,
                max_depth=self.max_depth,
                max_iterations=self.max_iterations,
            )
        return self._rlm

    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        """Run RLM evaluation on a sample.

        Args:
            sample: Input sample with query and context
            llm: LLM service (not used directly - RLM has its own LLM calls)

        Returns:
            Dict with sample_id, output, and usage metrics
        """
        # Set debug logger context - all LLM calls go to debug/{sample_id}.jsonl
        if debug_logger := get_debug_logger():
            debug_logger.set_context(sample.sample_id)

        import time

        start_time = time.perf_counter()

        # Format context
        context = _format_sample_context(sample)

        # Count reviews for template
        num_reviews = context.count("REVIEW_") if "REVIEW_" in context else "N"

        # Build query with agenda
        query = RLM_QUERY_TEMPLATE.format(
            agenda=sample.query,
            num_reviews=num_reviews,
        )

        # Run RLM
        rlm = self._get_rlm()
        try:
            result = await rlm.acompletion(query=query, context=context)
        except Exception as e:
            return self._make_result(
                sample.sample_id,
                "",
                {},
                llm_calls=0,
                error=str(e),
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Parse FINAL output
        parsed = self._parse_final_output(result)

        # RLM doesn't expose detailed token usage
        usage = {"latency_ms": latency_ms}

        return self._make_result(
            sample.sample_id,
            result,
            usage,
            llm_calls=self.max_iterations,  # Upper bound estimate
            parsed=parsed,
        )

    def _parse_final_output(self, result: str) -> Dict[str, Any]:
        """Parse FINAL() output from RLM result.

        Args:
            result: Raw result string from RLM

        Returns:
            Parsed dict or error dict
        """
        if not result:
            return {"error": "Empty result"}

        # Look for FINAL(...) pattern
        # Handle both FINAL({...}) and FINAL("...") formats
        final_patterns = [
            r"FINAL\s*\(\s*(\{.*?\})\s*\)",  # FINAL({...})
            r"FINAL\s*\(\s*'([^']+)'\s*\)",  # FINAL('...')
            r"FINAL\s*\(\s*\"([^\"]+)\"\s*\)",  # FINAL("...")
        ]

        for pattern in final_patterns:
            match = re.search(pattern, result, re.DOTALL)
            if match:
                content = match.group(1)
                try:
                    # Try parsing as JSON
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If not JSON, return as string value
                    return {"verdict": content, "raw": True}

        # If no FINAL found, try to extract verdict from text
        verdict_match = re.search(
            r"(?:verdict|VERDICT)[:\s]*[\"']?([A-Za-z\s]+Risk)[\"']?",
            result,
            re.IGNORECASE,
        )
        if verdict_match:
            verdict = verdict_match.group(1).strip()
            # Normalize
            v_lower = verdict.lower()
            if "low" in v_lower:
                verdict = "Low Risk"
            elif "critical" in v_lower:
                verdict = "Critical Risk"
            elif "high" in v_lower:
                verdict = "High Risk"
            return {"verdict": verdict, "extracted": True}

        return {"error": "Could not parse FINAL output", "raw_result": result[:500]}


# Empirical: ~3000 tokens per RLM iteration (500 prompt + 2500 completion)
TOKENS_PER_ITERATION = 3000


async def eval_restaurant_rlm(
    restaurant: Dict[str, Any],
    agenda: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-5-nano",
    max_iterations: int = 30,
    token_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Standalone function to evaluate a restaurant using RLM.

    This is for use in run_baseline.py alongside the existing eval_restaurant.

    Args:
        restaurant: Restaurant data dict with 'business' and 'reviews'
        agenda: The policy agenda/prompt text
        system_prompt: Output schema instructions (integrated into RLM query)
        model: LLM model to use
        max_iterations: Maximum RLM iterations (ignored if token_limit set)
        token_limit: Token budget - converted to iterations via TOKENS_PER_ITERATION

    Returns:
        Evaluation result dict
    """
    import time

    business = restaurant.get("business", {})
    name = business.get("name", "Unknown")
    business_id = business.get("business_id", "")

    # Format context
    context = _format_reviews_context(restaurant)
    num_reviews = len(restaurant.get("reviews", []))

    # Build query using the proper template with agenda
    # This uses the standard output format defined in RLM_QUERY_TEMPLATE
    query = RLM_QUERY_TEMPLATE.format(
        agenda=agenda,
        num_reviews=num_reviews,
    )

    start_time = time.perf_counter()

    # Convert token_limit to iterations if specified
    if token_limit is not None:
        effective_iterations = max(1, token_limit // TOKENS_PER_ITERATION)
    else:
        effective_iterations = max_iterations

    # Initialize RLM
    rlm = RLM(
        model=model,
        max_iterations=effective_iterations,
    )

    try:
        result = await rlm.acompletion(query=query, context=context)
    except Exception as e:
        return {
            "business_id": business_id,
            "name": name,
            "error": str(e),
        }

    latency_ms = (time.perf_counter() - start_time) * 1000

    # Parse result
    parsed = _parse_rlm_result(result)
    verdict = parsed.get("verdict")

    # Normalize verdict
    if verdict:
        v_lower = verdict.lower()
        if "low" in v_lower:
            verdict = "Low Risk"
        elif "critical" in v_lower:
            verdict = "Critical Risk"
        elif "high" in v_lower:
            verdict = "High Risk"

    return {
        "business_id": business_id,
        "name": name,
        "response": result,
        "parsed": parsed,
        "verdict": verdict,
        "risk_score": None,
        "prompt_chars": len(query) + len(context),
        "latency_ms": latency_ms,
    }


def _parse_rlm_result(result: str) -> Dict[str, Any]:
    """Parse RLM result - may be direct verdict or JSON."""
    if not result:
        return {"parse_error": "Empty result"}

    result = result.strip()

    # RLM extracts FINAL() content directly - check if it's a verdict string
    if result in ("Low Risk", "High Risk", "Critical Risk"):
        return {"verdict": result}

    # Check for verdict-like strings (case insensitive)
    result_lower = result.lower()
    if "low risk" in result_lower:
        return {"verdict": "Low Risk"}
    if "critical risk" in result_lower:
        return {"verdict": "Critical Risk"}
    if "high risk" in result_lower:
        return {"verdict": "High Risk"}

    # Try parsing as JSON
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        pass

    # Look for FINAL(...) pattern with JSON (fallback)
    match = re.search(r"FINAL\s*\(\s*(\{.*?\})\s*\)", result, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as e:
            return {"parse_error": f"JSON decode error: {e}"}

    # Try to find verdict in text
    verdict_match = re.search(
        r"(?:verdict|VERDICT)[:\s]*[\"']?([A-Za-z\s]+Risk)[\"']?",
        result,
        re.IGNORECASE,
    )
    if verdict_match:
        return {"verdict": verdict_match.group(1).strip(), "extracted": True}

    return {"parse_error": "Could not parse result"}
