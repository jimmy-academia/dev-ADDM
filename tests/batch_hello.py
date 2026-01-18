#!/usr/bin/env python3
"""Submit a tiny batch job and optionally poll for completion."""

import argparse
import json
import time
from typing import Any, Optional

from addm.llm_batch import BatchClient, build_chat_batch_item


def _get_batch_field(batch: Any, key: str) -> Optional[Any]:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hello-world OpenAI Batch test")
    parser.add_argument("--model", default="gpt-5-nano", help="Model to use")
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=0,
        help="Seconds to poll for completion (0 = submit only)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Polling interval in seconds",
    )
    args = parser.parse_args()

    client = BatchClient()
    messages = [{"role": "user", "content": "Say exactly: hello batch"}]
    item = build_chat_batch_item(
        custom_id="hello::1",
        model=args.model,
        messages=messages,
        temperature=0.0,
    )

    input_file_id = client.upload_batch_file([item])
    batch_id = client.submit_batch(input_file_id)
    print(f"Submitted batch: {batch_id}")

    if args.wait_seconds <= 0:
        return

    deadline = time.time() + args.wait_seconds
    while time.time() < deadline:
        batch = client.get_batch(batch_id)
        status = _get_batch_field(batch, "status")
        print(f"Status: {status}")
        if status in {"completed", "failed", "expired", "cancelled"}:
            break
        time.sleep(args.poll_interval)

    if status != "completed":
        return

    output_file_id = _get_batch_field(batch, "output_file_id")
    if not output_file_id:
        print("No output file available yet.")
        return

    output_bytes = client.download_file(output_file_id)
    for line in output_bytes.splitlines():
        if not line:
            continue
        item = json.loads(line)
        content = (
            item.get("response", {})
            .get("body", {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        print(f"Response: {content}")


if __name__ == "__main__":
    main()
