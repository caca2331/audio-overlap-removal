"""Small bounded-concurrency primitive shared by scan and processing."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor


def _bounded_ordered_map(function, items, workers: int):
    """Run at most ``workers`` jobs concurrently and yield in input order."""
    if workers == 1:
        for item in items:
            yield function(item)
        return

    iterator = iter(items)
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="reference-cancel",
    ) as executor:
        pending = deque()
        for _ in range(workers):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass
