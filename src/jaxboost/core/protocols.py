"""
Core protocols (interfaces) for jaxboost.

All components are defined as Protocols, enabling:
- Duck typing (no inheritance required)
- Easy composition of different strategies
- Type safety with mypy/pyright
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from jax import Array

# Generic type for parameters (must be a valid JAX PyTree)
P = TypeVar("P")


@runtime_checkable
class SplitFn(Protocol[P]):
    """Protocol for split functions.

    A split function computes a scalar "split score" for each sample,
    which is then passed to a routing function to determine left/right.
    """

    def init_params(self, key: Array, num_features: int, **kwargs: Any) -> P:
        """Initialize split parameters."""
        ...

    def compute_score(self, params: P, x: Array) -> Array:
        """Compute split score. Positive → right, Negative → left."""
        ...


@runtime_checkable
class RoutingFn(Protocol):
    """Protocol for routing functions.

    Converts split scores to routing probabilities (soft routing).
    """

    def __call__(self, score: Array, temperature: float = 1.0) -> Array:
        """Convert split score to probability of going right."""
        ...
