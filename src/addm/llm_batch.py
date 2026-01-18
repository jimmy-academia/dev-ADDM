"""OpenAI Batch API helpers for 24hrbatch mode."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency
    openai = None

from addm.llm import LLMServiceError, _load_api_key


class BatchClient:
    """Minimal OpenAI Batch client for chat completions."""

    def __init__(self, base_url: str = "") -> None:
        if openai is None:
            raise LLMServiceError("openai package is not installed")
        _load_api_key()
        base_url = base_url or None
        self._client = openai.OpenAI(base_url=base_url)

    def upload_batch_file(self, request_items: Iterable[Dict[str, Any]]) -> str:
        """Upload JSONL batch file and return file_id."""
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            for item in request_items:
                handle.write(json.dumps(item, ensure_ascii=True) + "\n")
            temp_path = Path(handle.name)

        try:
            with temp_path.open("rb") as handle:
                file_obj = self._client.files.create(file=handle, purpose="batch")
        finally:
            temp_path.unlink(missing_ok=True)

        return file_obj.id

    def submit_batch(
        self,
        input_file_id: str,
        completion_window: str = "24h",
        endpoint: str = "/v1/chat/completions",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a batch job and return batch_id."""
        batch = self._client.batches.create(
            input_file_id=input_file_id,
            completion_window=completion_window,
            endpoint=endpoint,
            metadata=metadata or {},
        )
        return batch.id

    def get_batch(self, batch_id: str) -> Dict[str, Any]:
        """Retrieve batch status."""
        batch = self._client.batches.retrieve(batch_id)
        return batch

    def download_file(self, file_id: str) -> bytes:
        """Download file content by file_id."""
        content = self._client.files.content(file_id)
        if hasattr(content, "read"):
            return content.read()
        if isinstance(content, (bytes, bytearray)):
            return bytes(content)
        if hasattr(content, "text"):
            return content.text.encode("utf-8")
        raise LLMServiceError("Unsupported file content response type")


def build_chat_batch_item(
    custom_id: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a single chat completion batch request item."""
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None and model != "gpt-5-nano":
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
