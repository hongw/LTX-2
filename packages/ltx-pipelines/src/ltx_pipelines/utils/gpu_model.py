import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

import torch

from ltx_pipelines.utils.helpers import cleanup_memory
from ltx_pipelines.utils.runtime_logging import gpu_mem_str

_M = TypeVar("_M", bound=torch.nn.Module)

logger = logging.getLogger("ltx_pipelines.progress")


def _gib(n: int) -> str:
    return f"{n / (1024**3):.2f}G"


@contextmanager
def gpu_model(model: _M) -> Iterator[_M]:
    """Context manager that yields a model and releases its memory on exit.
    Moves all parameters and buffers to ``meta`` device on exit, which
    immediately releases the underlying storage on **both** GPU and CPU,
    then runs ``cleanup_memory()`` to reclaim fragmented CUDA memory.
    Usage::
        with gpu_model(build_encoder()) as encoder:
            ...  # use encoder — typed as the concrete class
        # GPU + CPU memory freed automatically
    """
    cls = type(model).__name__
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    logger.info("gpu_model[%s] enter  %s", cls, gpu_mem_str())
    try:
        yield model
    finally:
        torch.cuda.synchronize()
        # .to("meta") releases storage for all parameters/buffers regardless
        # of their original device (CUDA or CPU).
        model.to("meta")
        cleanup_memory()
        after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        freed = before - after  # positive = released
        logger.info(
            "gpu_model[%s] exit   %s  freed=%s",
            cls,
            gpu_mem_str(),
            _gib(freed) if freed >= 0 else f"-{_gib(-freed)}",
        )
