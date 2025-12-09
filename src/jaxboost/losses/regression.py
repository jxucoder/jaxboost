"""Regression loss functions."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def mse_loss(predictions: Array, targets: Array) -> Array:
    """Mean squared error loss.

    Args:
        predictions: Model predictions, shape (batch,).
        targets: Ground truth targets, shape (batch,).

    Returns:
        Scalar mean squared error.
    """
    return jnp.mean((predictions - targets) ** 2)
