"""Loss functions for jaxboost."""

from jaxboost.losses.classification import sigmoid_binary_cross_entropy
from jaxboost.losses.regression import mse_loss

__all__ = [
    "mse_loss",
    "sigmoid_binary_cross_entropy",
]
