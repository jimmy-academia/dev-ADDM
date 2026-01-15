"""Unified LLM service with async batching and provider abstraction."""

import asyncio
import random
import time
from typing import Any, Callable, Dict, List, Optional

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency
    openai = None
try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None

from addm.utils.async_utils import gather_with_concurrency


class LLMServiceError(RuntimeError):
    pass


class LLMService:
    def __init__(self) -> None:
        self._config: Dict[str, Any] = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": None,
            "base_url": "",
            "request_timeout": 90.0,
            "max_retries": 4,
            "max_concurrent": 32,
        }
        self._clients: Dict[str, Any] = {}
        self._mock_responder: Optional[Callable[[List[Dict[str, str]]], str]] = None

    def configure(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in {"temperature", "request_timeout"}:
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

    async def call_async(self, messages: List[Dict[str, str]]) -> str:
        provider = self._config["provider"]
        model = self._config["model"]
        temperature = self._config["temperature"]
        max_tokens = self._config["max_tokens"]
        timeout = self._config["request_timeout"]
        max_retries = self._config["max_retries"]

        for attempt in range(max_retries + 1):
            try:
                if provider == "mock":
                    if self._mock_responder:
                        return self._mock_responder(messages)
                    return "mock-response"
                if provider == "openai":
                    client = self._get_openai_client(async_mode=True)
                    kwargs: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "timeout": timeout,
                    }
                    if max_tokens:
                        kwargs["max_tokens"] = max_tokens
                    resp = await client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content
                if provider == "anthropic":
                    client = self._get_anthropic_client(async_mode=True)
                    payload = self._build_anthropic_payload(messages)
                    kwargs = {
                        "model": model,
                        "messages": payload["messages"],
                        "temperature": temperature,
                        "max_tokens": max_tokens or 1024,
                        "timeout": timeout,
                    }
                    if payload["system"]:
                        kwargs["system"] = payload["system"]
                    resp = await client.messages.create(**kwargs)
                    return resp.content[0].text
                raise LLMServiceError(f"Unsupported provider: {provider}")
            except Exception as exc:  # pragma: no cover - network or SDK errors
                if attempt >= max_retries:
                    raise
                jitter = random.random() * 0.2
                sleep_for = (2 ** attempt) + jitter
                await asyncio.sleep(sleep_for)
                last_exc = exc
        raise LLMServiceError(f"LLM request failed: {last_exc}")

    def call(self, messages: List[Dict[str, str]]) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.call_async(messages))
        raise LLMServiceError(\"call() cannot be used from an active event loop; use call_async().\")

    async def batch_call(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        max_concurrent = self._config["max_concurrent"]
        tasks = [self.call_async(messages) for messages in batch]
        return await gather_with_concurrency(max_concurrent, tasks)
