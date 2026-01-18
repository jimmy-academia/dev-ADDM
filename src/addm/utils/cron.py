"""Cron helpers for macOS/Linux batch polling."""

from __future__ import annotations

import os
import subprocess
from typing import List


def _is_windows() -> bool:
    return os.name == "nt"


def read_crontab() -> List[str]:
    if _is_windows():
        raise RuntimeError("cron is not supported on Windows")
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        if "no crontab" in stderr:
            return []
        raise RuntimeError(result.stderr.strip() or "failed to read crontab")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return lines


def write_crontab(lines: List[str]) -> None:
    if _is_windows():
        raise RuntimeError("cron is not supported on Windows")
    content = "\n".join(lines) + "\n"
    subprocess.run(["crontab", "-"], input=content, text=True, check=True)


def install_cron_job(job_line: str, marker: str) -> None:
    """Add or replace a cron job line using marker for dedupe."""
    lines = read_crontab()
    lines = [line for line in lines if marker not in line]
    lines.append(job_line)
    write_crontab(lines)


def remove_cron_job(marker: str) -> None:
    """Remove any cron line containing marker."""
    lines = read_crontab()
    lines = [line for line in lines if marker not in line]
    write_crontab(lines)

