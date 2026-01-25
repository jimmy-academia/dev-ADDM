"""ReACT (Reasoning and Acting) baseline method.

Reference: ReAct: Synergizing Reasoning and Acting in Language Models
Yao et al., ICLR 2023
https://arxiv.org/abs/2210.03629

Adapted from anot/methods/react.py for ADDM benchmark.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from addm.data.types import Sample
from addm.llm import LLMService
from addm.methods.base import Method
from addm.utils.debug_logger import get_debug_logger


MAX_STEPS = 5

SYSTEM_PROMPT = """You are analyzing restaurant reviews to answer a query. Use tools to explore the data.

Available actions:
- read("reviews") → get all review texts
- read("reviews.0") → get first review
- read("reviews.0.text") → get text of first review
- read("reviews.0.stars") → get star rating of first review
- read("business") → get business info (name, categories, attributes)
- search("keyword") → find reviews containing keyword (returns indices)
- count() → get total number of reviews
- finish("your answer") → submit your final answer

Format each step as:
Thought: [your reasoning about what to do next]
Action: [one action from above]

Be efficient - aim to finish in 3-4 steps. Start by understanding the query, then gather relevant evidence, then answer."""


def tool_read(path: str, data: dict) -> str:
    """Read value at path in nested data structure.

    Supports paths like:
    - "reviews" → all reviews
    - "reviews.0" → first review
    - "reviews.0.text" → text of first review
    - "business.name" → business name
    """
    if not path:
        return json.dumps(data, ensure_ascii=False)[:2000]

    parts = re.split(r'\.|\[|\]', path)
    parts = [x for x in parts if x]

    val = data
    for part in parts:
        try:
            if isinstance(val, list) and part.isdigit():
                val = val[int(part)]
            elif isinstance(val, dict):
                val = val.get(part)
            else:
                return f"Error: cannot access '{part}' in {type(val).__name__}"
        except (IndexError, KeyError, TypeError):
            return f"Error: path '{path}' not found"

    if val is None:
        return f"Error: path '{path}' not found"
    if isinstance(val, str):
        return val
    result = json.dumps(val, ensure_ascii=False)
    if len(result) > 3000:
        return result[:3000] + "\n... (truncated, use more specific path)"
    return result


def tool_search(keyword: str, data: dict) -> str:
    """Search for reviews containing keyword.

    Returns indices of matching reviews.
    """
    reviews = data.get("reviews", [])
    matches = []
    keyword_lower = keyword.lower()

    for i, review in enumerate(reviews):
        text = review.get("text", "")
        if keyword_lower in text.lower():
            matches.append(i)

    if not matches:
        return f"No reviews found containing '{keyword}'"
    return f"Found {len(matches)} reviews containing '{keyword}': indices {matches[:20]}"


def tool_count(data: dict) -> str:
    """Count total number of reviews."""
    reviews = data.get("reviews", [])
    return f"Total reviews: {len(reviews)}"


class ReACTMethod(Method):
    """ReACT baseline - agentic loop with thought/action/observation."""

    name = "react"

    def __init__(self, max_steps: int = MAX_STEPS):
        """Initialize ReACT method.

        Args:
            max_steps: Maximum number of reasoning steps (default: 5)
        """
        self.max_steps = max_steps

    def _parse_context(self, context: str) -> dict:
        """Parse context JSON string into dict."""
        try:
            return json.loads(context) if context else {}
        except json.JSONDecodeError:
            return {}

    def _build_prompt(
        self,
        query: str,
        data: dict,
        history: List[dict],
    ) -> str:
        """Build prompt for ReACT step."""
        reviews = data.get("reviews", [])
        n_reviews = len(reviews)

        parts = []
        parts.append(f"[QUERY]")
        parts.append(query)
        parts.append("")
        parts.append(f"[DATA INFO]")
        parts.append(f"Restaurant with {n_reviews} reviews available.")
        parts.append("Use read(path), search(keyword), count(), or finish(answer).")
        parts.append("")

        if history:
            parts.append("[PREVIOUS STEPS]")
            for i, step in enumerate(history, 1):
                parts.append(f"Step {i}:")
                parts.append(f"Thought: {step['thought']}")
                parts.append(f"Action: {step['action']}")
                parts.append(f"Observation: {step['observation']}")
                parts.append("")

        parts.append("[YOUR TURN]")
        parts.append("Thought: ")

        return "\n".join(parts)

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse Thought and Action from response."""
        thought = ""
        action = ""

        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\nAction:|\Z)", response, re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(
            r"Action:\s*(.+?)(?=\nThought:|\nObservation:|\Z)",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if action_match:
            action = action_match.group(1).strip()

        return thought, action

    def _execute_action(self, action: str, data: dict) -> Tuple[str, bool, str]:
        """Execute an action and return observation.

        Returns:
            Tuple of (observation, is_finished, final_answer)
        """
        action = action.strip()

        # Check for finish action
        finish_match = re.search(
            r'finish\s*\(\s*["\']?(.+?)["\']?\s*\)', action, re.IGNORECASE | re.DOTALL
        )
        if finish_match:
            answer = finish_match.group(1).strip()
            return f"Final answer submitted.", True, answer

        # Check for read action
        read_match = re.search(r'read\s*\(\s*["\']([^"\']*)["\']?\s*\)', action, re.IGNORECASE)
        if read_match:
            path = read_match.group(1)
            result = tool_read(path, data)
            return result, False, ""

        # Check for search action
        search_match = re.search(r'search\s*\(\s*["\']([^"\']+)["\']?\s*\)', action, re.IGNORECASE)
        if search_match:
            keyword = search_match.group(1)
            result = tool_search(keyword, data)
            return result, False, ""

        # Check for count action
        if re.search(r'count\s*\(\s*\)', action, re.IGNORECASE):
            result = tool_count(data)
            return result, False, ""

        # Unknown action
        return (
            f"Unknown action: {action}. Use read(path), search(keyword), count(), or finish(answer).",
            False,
            "",
        )

    async def run_sample(self, sample: Sample, llm: LLMService) -> Dict[str, Any]:
        """Run ReACT evaluation on a sample.

        Args:
            sample: Input sample with query and context
            llm: LLM service

        Returns:
            Dict with sample_id, output, and usage metrics
        """
        # Set debug logger context - all LLM calls go to debug/{sample_id}.jsonl
        if debug_logger := get_debug_logger():
            debug_logger.set_context(sample.sample_id)

        start_time = time.time()
        usage_records = []

        # Parse context
        data = self._parse_context(sample.context or "")
        history = []
        final_answer = ""

        for step in range(self.max_steps):
            prompt = self._build_prompt(sample.query, data, history)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            response, usage = await llm.call_async_with_usage(
                messages,
                context={
                    "sample_id": sample.sample_id,
                    "method": self.name,
                    "step": step + 1,
                },
            )
            usage_records.append(usage)

            thought, action = self._parse_response(response)

            if not action:
                # No action found, try to extract from full response
                action = 'finish("Unable to determine answer")'

            observation, is_finished, answer = self._execute_action(action, data)

            if is_finished:
                final_answer = answer
                break

            history.append({
                "thought": thought,
                "action": action,
                "observation": observation,
            })

        # If we hit max steps without finishing, force a conclusion
        if not final_answer:
            prompt = self._build_prompt(sample.query, data, history)
            prompt += f"\nYou have reached the maximum steps. Submit your answer now."
            prompt += f"\nThought: Based on my exploration, I will provide my final answer."
            prompt += f'\nAction: finish("'

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            response, usage = await llm.call_async_with_usage(
                messages,
                context={
                    "sample_id": sample.sample_id,
                    "method": self.name,
                    "step": "force_finish",
                },
            )
            usage_records.append(usage)
            final_answer = response

        # Aggregate usage
        total_usage = self._accumulate_usage(usage_records)
        total_usage["latency_ms"] = (time.time() - start_time) * 1000

        return self._make_result(
            sample.sample_id,
            final_answer,
            total_usage,
            llm_calls=len(usage_records),
            steps_taken=len(history) + 1,
            max_steps=self.max_steps,
        )
