"""Stub DelegatingBuilder for single-GPU environments."""

from typing import Generic, TypeVar

T = TypeVar("T")


class DelegatingBuilder(Generic[T]):
    """No-op placeholder. Multi-GPU delegation is not used in SoyMedia."""
    pass
