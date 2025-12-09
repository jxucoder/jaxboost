"""Classification loss functions."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def sigmoid_binary_cross_entropy(logits: Array, targets: Array) -> Array:
    """Binary cross-entropy loss with logits (numerically stable).

    Args:
        logits: Raw logits (before sigmoid), shape (batch,).
        targets: Binary targets (0 or 1), shape (batch,).

    Returns:
        Scalar binary cross-entropy loss.
    """
    # Numerically stable BCE:
    # max(logits, 0) - logits * targets + log(1 + exp(-|logits|))
    loss = (
        jnp.maximum(logits, 0)
        - logits * targets
        + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )
    return jnp.mean(loss)
