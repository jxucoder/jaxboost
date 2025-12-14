"""
Binary classification objective functions.

All objectives are pre-wrapped with AutoObjective and ready to use
with XGBoost/LightGBM.
"""

import jax
import jax.numpy as jnp

from jaxboost.objective.auto import AutoObjective


@AutoObjective
def focal_loss(
    y_pred: jax.Array,
    y_true: jax.Array,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> jax.Array:
    """
    Focal Loss for imbalanced binary classification.

    Down-weights well-classified examples and focuses on hard examples.
    Particularly useful for highly imbalanced datasets.

    Args:
        y_pred: Raw prediction (logit), will be passed through sigmoid
        y_true: Binary label (0 or 1)
        gamma: Focusing parameter. Higher = more focus on hard examples. Default: 2.0
        alpha: Class balance weight for positive class. Default: 0.25

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002

    Example:
        >>> model = xgb.train(params, dtrain, obj=focal_loss.xgb_objective)
        >>> # With custom parameters:
        >>> model = xgb.train(params, dtrain, obj=focal_loss.get_xgb_objective(gamma=3.0))
    """
    p = jax.nn.sigmoid(y_pred)
    # Clip for numerical stability
    p = jnp.clip(p, 1e-7, 1 - 1e-7)

    # Cross-entropy loss
    ce_loss = -y_true * jnp.log(p) - (1 - y_true) * jnp.log(1 - p)

    # Focal weight: (1 - p_t)^gamma where p_t is prob of true class
    p_t = y_true * p + (1 - y_true) * (1 - p)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    focal_weight = alpha_t * (1 - p_t) ** gamma

    return focal_weight * ce_loss


@AutoObjective
def binary_crossentropy(
    y_pred: jax.Array,
    y_true: jax.Array,
) -> jax.Array:
    """
    Binary Cross-Entropy Loss.

    Standard binary classification loss. Included for testing and as a baseline.

    Args:
        y_pred: Raw prediction (logit), will be passed through sigmoid
        y_true: Binary label (0 or 1)

    Example:
        >>> model = xgb.train(params, dtrain, obj=binary_crossentropy.xgb_objective)
    """
    p = jax.nn.sigmoid(y_pred)
    p = jnp.clip(p, 1e-7, 1 - 1e-7)
    return -y_true * jnp.log(p) - (1 - y_true) * jnp.log(1 - p)


@AutoObjective
def weighted_binary_crossentropy(
    y_pred: jax.Array,
    y_true: jax.Array,
    pos_weight: float = 1.0,
) -> jax.Array:
    """
    Weighted Binary Cross-Entropy Loss.

    Applies a weight to the positive class to handle class imbalance.

    Args:
        y_pred: Raw prediction (logit), will be passed through sigmoid
        y_true: Binary label (0 or 1)
        pos_weight: Weight for positive class. Default: 1.0

    Example:
        >>> # 10x weight for positive class
        >>> obj = weighted_binary_crossentropy.with_params(pos_weight=10.0)
        >>> model = xgb.train(params, dtrain, obj=obj.xgb_objective)
    """
    p = jax.nn.sigmoid(y_pred)
    p = jnp.clip(p, 1e-7, 1 - 1e-7)
    return -pos_weight * y_true * jnp.log(p) - (1 - y_true) * jnp.log(1 - p)


@AutoObjective
def hinge_loss(
    y_pred: jax.Array,
    y_true: jax.Array,
    margin: float = 1.0,
) -> jax.Array:
    """
    Smooth Hinge Loss for binary classification.

    SVM-style loss with smooth approximation for non-zero Hessians.

    Args:
        y_pred: Raw prediction score
        y_true: Binary label (0 or 1), will be converted to {-1, +1}
        margin: Margin parameter. Default: 1.0

    Example:
        >>> model = xgb.train(params, dtrain, obj=hinge_loss.xgb_objective)
    """
    # Convert {0, 1} to {-1, +1}
    y_signed = 2 * y_true - 1

    # Smooth hinge: use softplus for differentiability
    # hinge(z) = max(0, 1 - z) ≈ softplus(1 - z)
    z = y_signed * y_pred
    return jax.nn.softplus(margin - z)

