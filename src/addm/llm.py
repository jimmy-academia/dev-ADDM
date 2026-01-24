"""Unified LLM service with async batching and provider abstraction."""

import asyncio
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency
    openai = None
try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None

from addm.utils.async_utils import gather_with_concurrency
from addm.utils.debug_logger import get_debug_logger
from addm.utils.usage import compute_cost, usage_tracker


def _load_api_key():
    """Load API key from file if not in environment."""
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        return
    # Look for .openaiapi at Station directory level (../../../.openaiapi from src/addm/llm.py)
    key_file = Path(__file__).parent.parent.parent.parent / ".openaiapi"
    if key_file.exists():
        key = key_file.read_text().strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key


_load_api_key()


class LLMServiceError(RuntimeError):
    pass


class LLMService:
    def __init__(self) -> None:
        self._config: Dict[str, Any] = {
            "provider": "openai",
            "model": "gpt-5-nano",
            # Note: temperature not set - use model defaults (1.0)
            # Many models (gpt-5-mini, gpt-5-nano) only support default temperature
            "max_tokens": None,
            "base_url": "",
            "request_timeout": 300.0,  # 5 minutes for large K values
            "max_retries": 0,  # No retries - single attempt with long timeout
            "max_concurrent": 32,
            # Rate limit mitigation: small delay between requests (seconds)
            # Set to 0.0 to disable, 0.03-0.1 recommended for high-volume runs
            "request_delay": 0.03,
        }
        self._clients: Dict[str, Any] = {}
        self._mock_responder: Optional[Callable[[List[Dict[str, str]]], str]] = None

    def configure(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in {"request_timeout", "request_delay"}:
                self._config[key] = float(value)
            elif key in {"max_tokens", "max_retries", "max_concurrent"}:
                self._config[key] = int(value)
            else:
                self._config[key] = value

    def set_mock_responder(self, responder: Callable[[List[Dict[str, str]]], str]) -> None:
        self._mock_responder = responder

    def _get_openai_client(self, async_mode: bool) -> Any:
        if openai is None:
            raise LLMServiceError("openai package is not installed")
        key = "openai_async" if async_mode else "openai"
        if key in self._clients:
            return self._clients[key]
        base_url = self._config.get("base_url") or None
        if async_mode:
            client = openai.AsyncOpenAI(base_url=base_url) if hasattr(openai, "AsyncOpenAI") else openai.AsyncClient(base_url=base_url)
        else:
            client = openai.OpenAI(base_url=base_url) if hasattr(openai, "OpenAI") else openai.Client(base_url=base_url)
        self._clients[key] = client
        return client

    def _get_anthropic_client(self, async_mode: bool) -> Any:
        if anthropic is None:
            raise LLMServiceError("anthropic package is not installed")
        key = "anthropic_async" if async_mode else "anthropic"
        if key in self._clients:
            return self._clients[key]
        client = anthropic.AsyncAnthropic() if async_mode else anthropic.Anthropic()
        self._clients[key] = client
        return client

    def _build_anthropic_payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        system = "\n\n".join(system_msgs) if system_msgs else None
        user = "\n\n".join(user_msgs)
        return {
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

    async def call_async(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Call LLM asynchronously, returning just the response text.

        Args:
            messages: List of message dicts with 'role' and 'content'
            context: Optional context for usage tracking (sample_id, run_id, etc.)
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            Response text from the LLM
        """
        response, _ = await self._call_async_with_usage(messages, context, response_format)
        return response

    async def call_async_with_usage(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Call LLM asynchronously, returning response and usage info.

        Args:
            messages: List of message dicts with 'role' and 'content'
            context: Optional context for usage tracking (sample_id, run_id, etc.)
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            Tuple of (response_text, usage_dict)
            usage_dict contains: prompt_tokens, completion_tokens, latency_ms, cost_usd
        """
        return await self._call_async_with_usage(messages, context, response_format)

    async def _call_async_with_usage(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Internal method that performs the LLM call and tracks usage."""
        provider = self._config["provider"]
        model = self._config["model"]
        max_tokens = self._config["max_tokens"]
        timeout = self._config["request_timeout"]
        max_retries = self._config["max_retries"]
        request_delay = self._config.get("request_delay", 0.0)

        # Hard timeout margin beyond SDK timeout (safety net for hung connections)
        hard_timeout = timeout + 30

        for attempt in range(max_retries + 1):
            # Rate limit mitigation: small delay before first attempt (not retries)
            # Retries already have exponential backoff
            if attempt == 0 and request_delay > 0:
                await asyncio.sleep(request_delay)

            attempt_start = time.perf_counter()  # Reset each attempt (only count successful)
            try:
                if provider == "mock":
                    latency_ms = (time.perf_counter() - attempt_start) * 1000
                    if self._mock_responder:
                        response = self._mock_responder(messages)
                    else:
                        response = "mock-response"
                    # Mock usage - estimate based on message length
                    prompt_chars = sum(len(m.get("content", "")) for m in messages)
                    usage = {
                        "prompt_tokens": prompt_chars // 4,  # rough estimate
                        "completion_tokens": len(response) // 4,
                        "latency_ms": latency_ms,
                        "cost_usd": 0.0,
                    }
                    return response, usage

                if provider == "openai":
                    client = self._get_openai_client(async_mode=True)
                    kwargs: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "timeout": timeout,
                    }
                    # Don't set temperature - use model defaults (1.0)
                    # Many models only support default temperature
                    if max_tokens:
                        kwargs["max_tokens"] = max_tokens
                    if response_format:
                        kwargs["response_format"] = response_format
                    # Hard timeout wrapper - catches hung connections SDK timeout misses
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(**kwargs),
                        timeout=hard_timeout,
                    )
                    latency_ms = (time.perf_counter() - attempt_start) * 1000
                    response = resp.choices[0].message.content

                    # Extract usage from response
                    prompt_tokens = getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0
                    completion_tokens = getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0
                    cost_usd = compute_cost(model, prompt_tokens, completion_tokens)

                    # Record to global tracker
                    prompt_preview = str(messages[-1].get("content", ""))[:200] if messages else ""
                    usage_tracker.record(
                        model=model,
                        provider=provider,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        context=context,
                        prompt_preview=prompt_preview,
                        response_preview=str(response)[:200] if response else "",
                    )

                    # Log to debug logger if enabled
                    debug_logger = get_debug_logger()
                    if debug_logger and debug_logger.enabled:
                        prompt_text = "\n\n".join(
                            f"{m['role']}: {m['content']}" for m in messages
                        )
                        debug_logger.log_llm_call(
                            sample_id=context.get("sample_id", "unknown") if context else "unknown",
                            phase=context.get("phase", "main") if context else "main",
                            prompt=prompt_text,
                            response=response or "",
                            model=model,
                            latency_ms=latency_ms,
                            metadata=context,
                        )

                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "latency_ms": latency_ms,
                        "cost_usd": cost_usd,
                    }
                    return response, usage

                if provider == "anthropic":
                    client = self._get_anthropic_client(async_mode=True)
                    payload = self._build_anthropic_payload(messages)
                    kwargs = {
                        "model": model,
                        "messages": payload["messages"],
                        "max_tokens": max_tokens or 1024,
                        "timeout": timeout,
                    }
                    if payload["system"]:
                        kwargs["system"] = payload["system"]
                    # Hard timeout wrapper - catches hung connections SDK timeout misses
                    resp = await asyncio.wait_for(
                        client.messages.create(**kwargs),
                        timeout=hard_timeout,
                    )
                    latency_ms = (time.perf_counter() - attempt_start) * 1000
                    response = resp.content[0].text

                    # Extract usage from response
                    prompt_tokens = getattr(resp.usage, "input_tokens", 0) if resp.usage else 0
                    completion_tokens = getattr(resp.usage, "output_tokens", 0) if resp.usage else 0
                    cost_usd = compute_cost(model, prompt_tokens, completion_tokens)

                    # Record to global tracker
                    prompt_preview = str(messages[-1].get("content", ""))[:200] if messages else ""
                    usage_tracker.record(
                        model=model,
                        provider=provider,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        context=context,
                        prompt_preview=prompt_preview,
                        response_preview=str(response)[:200] if response else "",
                    )

                    # Log to debug logger if enabled
                    debug_logger = get_debug_logger()
                    if debug_logger and debug_logger.enabled:
                        prompt_text = "\n\n".join(
                            f"{m['role']}: {m['content']}" for m in messages
                        )
                        debug_logger.log_llm_call(
                            sample_id=context.get("sample_id", "unknown") if context else "unknown",
                            phase=context.get("phase", "main") if context else "main",
                            prompt=prompt_text,
                            response=response or "",
                            model=model,
                            latency_ms=latency_ms,
                            metadata=context,
                        )

                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "latency_ms": latency_ms,
                        "cost_usd": cost_usd,
                    }
                    return response, usage

                raise LLMServiceError(f"Unsupported provider: {provider}")
            except asyncio.TimeoutError:  # Hard timeout fired
                if attempt >= max_retries:
                    raise LLMServiceError(
                        f"Hard timeout after {max_retries + 1} attempts "
                        f"({hard_timeout}s each)"
                    )
                # Fall through to backoff and retry
                last_exc = asyncio.TimeoutError(f"Attempt {attempt + 1} timed out")
            except Exception as exc:  # pragma: no cover - network or SDK errors
                if attempt >= max_retries:
                    raise
                last_exc = exc
            # Exponential backoff before retry
            jitter = random.random() * 0.2
            sleep_for = (2 ** attempt) + jitter
            await asyncio.sleep(sleep_for)
        raise LLMServiceError(f"LLM request failed: {last_exc}")

    def call(self, messages: List[Dict[str, str]]) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.call_async(messages))
        raise LLMServiceError("call() cannot be used from an active event loop; use call_async().")

    async def batch_call(
        self,
        batch: List[List[Dict[str, str]]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Call LLM for a batch of messages, returning just responses.

        Args:
            batch: List of message lists
            context: Optional base context (will be augmented with batch_idx)

        Returns:
            List of response texts
        """
        max_concurrent = self._config["max_concurrent"]

        async def call_with_idx(idx: int, messages: List[Dict[str, str]]) -> str:
            ctx = {**(context or {}), "batch_idx": idx}
            return await self.call_async(messages, ctx)

        tasks = [call_with_idx(i, messages) for i, messages in enumerate(batch)]
        return await gather_with_concurrency(max_concurrent, tasks)

    async def batch_call_with_usage(
        self,
        batch: List[List[Dict[str, str]]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Call LLM for a batch of messages, returning responses and usage.

        Args:
            batch: List of message lists
            context: Optional base context (will be augmented with batch_idx)

        Returns:
            Tuple of (list of response texts, list of usage dicts)
        """
        max_concurrent = self._config["max_concurrent"]

        async def call_with_idx(
            idx: int, messages: List[Dict[str, str]]
        ) -> Tuple[str, Dict[str, Any]]:
            ctx = {**(context or {}), "batch_idx": idx}
            return await self.call_async_with_usage(messages, ctx)

        tasks = [call_with_idx(i, messages) for i, messages in enumerate(batch)]
        results = await gather_with_concurrency(max_concurrent, tasks)
        responses = [r[0] for r in results]
        usages = [r[1] for r in results]
        return responses, usages
