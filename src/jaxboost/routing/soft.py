"""Soft (sigmoid) routing function.

Converts split scores to probabilities using sigmoid function.
This enables gradient flow through the tree structure.
"""

from __future__ import annotations

import jax
from jax import Array


def soft_routing(score: Array, temperature: float = 1.0) -> Array:
    """Soft routing using sigmoid function.

    Args:
        score: Split scores, any shape.
        temperature: Sharpness parameter. Higher = softer, Lower = sharper.

    Returns:
        Probability of going right, same shape as score, values in [0, 1].
    """
    return jax.nn.sigmoid(score * temperature)
