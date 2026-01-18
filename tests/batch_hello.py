#!/usr/bin/env python3
"""Submit a tiny batch job and install a cron fetcher."""

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Optional

from addm.llm_batch import BatchClient, build_chat_batch_item
from addm.utils.cron import install_cron_job, remove_cron_job


def _get_batch_field(batch: Any, key: str) -> Optional[Any]:
    if isinstance(batch, dict):
        return batch.get(key)
    return getattr(batch, key, None)


def _build_cron_command(batch_id: str) -> str:
    repo_root = Path.cwd().resolve()
    cmd = [
        sys.executable,
        "tests/batch_hello.py",
        "--batch-id",
        batch_id,
    ]
    command = " ".join(shlex.quote(c) for c in cmd)
    return f"cd {shlex.quote(str(repo_root))} && {command}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Hello-world OpenAI Batch test")
    parser.add_argument("--model", default="gpt-5-nano", help="Model to use")
    parser.add_argument("--batch-id", default=None, help="Batch ID for fetch-only runs")
    args = parser.parse_args()

    client = BatchClient()
    if args.batch_id:
        batch = client.get_batch(args.batch_id)
        status = _get_batch_field(batch, "status")
        print(f"Status: {status}")
        if status not in {"completed", "failed", "expired", "cancelled"}:
            return

        output_file_id = _get_batch_field(batch, "output_file_id")
        if not output_file_id:
            print("No output file available.")
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

        marker = f"ADDM_BATCH_{args.batch_id}"
        try:
            remove_cron_job(marker)
            print(f"Removed cron job for {args.batch_id}")
        except Exception as exc:
            print(f"[WARN] Failed to remove cron job: {exc}")
        return

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

    marker = f"ADDM_BATCH_{batch_id}"
    cron_line = f"*/5 * * * * {_build_cron_command(batch_id)} # {marker}"
    try:
        install_cron_job(cron_line, marker)
        print(f"Installed cron job for batch {batch_id}")
    except Exception as exc:
        print(f"[WARN] Failed to install cron job: {exc}")


if __name__ == "__main__":
    main()
