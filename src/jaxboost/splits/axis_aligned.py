"""Axis-aligned split function (soft version).

Uses softmax for differentiable feature selection.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class AxisAlignedSplitParams(NamedTuple):
    """Parameters for a soft axis-aligned split.

    Attributes:
        feature_logits: Soft feature selection logits, shape (num_features,).
        threshold: Split threshold, scalar.
    """

    feature_logits: Array  # (num_features,)
    threshold: Array  # scalar


class AxisAlignedSplit:
    """Soft axis-aligned split using softmax feature selection."""

    def init_params(
        self,
        key: Array,
        num_features: int,
        init_threshold_scale: float = 0.1,
    ) -> AxisAlignedSplitParams:
        """Initialize split parameters."""
        keys = jax.random.split(key, 2)
        feature_logits = jax.random.normal(keys[0], (num_features,)) * 0.01
        threshold = jax.random.normal(keys[1], ()) * init_threshold_scale
        return AxisAlignedSplitParams(feature_logits=feature_logits, threshold=threshold)

    def compute_score(self, params: AxisAlignedSplitParams, x: Array) -> Array:
        """Compute split score. Positive → right, Negative → left."""
        feature_probs = jax.nn.softmax(params.feature_logits)
        selected_value = jnp.sum(x * feature_probs, axis=-1)
        return selected_value - params.threshold
