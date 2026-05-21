"""Lightweight observability helpers for pipeline progress + GPU memory.

The LTX pipelines previously relied on ``tqdm`` for progress visibility, which
works well on a TTY but is silent in containerized log aggregators (the
``\\r``-based bar redraws are merged into a single never-flushed line).

This module exposes:

* :func:`gpu_mem_str` — a compact snapshot of current CUDA memory usage,
  suitable for inclusion in log messages.
* :func:`log_section` — a context manager that logs the start and end of a
  named code section together with elapsed time and GPU allocation delta.

Both helpers are intentionally cheap (a few CUDA stat reads) and safe to call
anywhere in the pipeline. Output goes through the standard :mod:`logging`
module under the ``ltx_pipelines.progress`` logger.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch

logger = logging.getLogger("ltx_pipelines.progress")


def _gib(n: int) -> str:
    return f"{n / (1024**3):.2f}G"


def gpu_mem_str() -> str:
    """Return a compact string describing current CUDA memory usage.

    Example: ``"gpu used=27.30G alloc=24.10G peak=27.30G free=51.95G"``.
    Returns ``"gpu=none"`` if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return "gpu=none"
    dev = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(dev)
    used = total - free
    allocated = torch.cuda.memory_allocated(dev)
    peak = torch.cuda.max_memory_allocated(dev)
    return f"gpu used={_gib(used)} alloc={_gib(allocated)} peak={_gib(peak)} free={_gib(free)}"


@contextmanager
def log_section(name: str, *, reset_peak: bool = True) -> Iterator[None]:
    """Log the start and end of a named section with timing + memory delta.

    On exit, also reports the change in ``torch.cuda.memory_allocated`` so it
    is obvious when a section leaks GPU memory (e.g. a model that wasn't
    fully released).
    """
    if torch.cuda.is_available() and reset_peak:
        torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    t0 = time.monotonic()
    logger.info(">>> %s  start  %s", name, gpu_mem_str())
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        delta = after - before
        sign = "+" if delta >= 0 else "-"
        logger.info(
            "<<< %s  done in %.2fs  %s  alloc_delta=%s%s",
            name,
            elapsed,
            gpu_mem_str(),
            sign,
            _gib(abs(delta)),
        )


def logged_steps(total: int, name: str = "step") -> Iterator[int]:
    """Yield ``range(total)`` indices, logging timing + GPU mem after each.

    Designed as a drop-in replacement for ``tqdm(range(total))`` inside
    denoising / decoding loops that otherwise produce no observable progress
    in non-TTY environments.
    """
    for i in range(total):
        t0 = time.monotonic()
        yield i
        logger.info(
            "%s %d/%d done in %.2fs  %s",
            name,
            i + 1,
            total,
            time.monotonic() - t0,
            gpu_mem_str(),
        )
