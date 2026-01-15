"""Async utilities for bounded concurrency."""

import asyncio
from typing import Iterable, Awaitable, TypeVar, List

T = TypeVar("T")


async def gather_with_concurrency(limit: int, tasks: Iterable[Awaitable[T]]) -> List[T]:
    if limit <= 0:
        return await asyncio.gather(*tasks)

    semaphore = asyncio.Semaphore(limit)

    async def _run(task: Awaitable[T]) -> T:
        async with semaphore:
            return await task

    return await asyncio.gather(*[_run(t) for t in tasks])
