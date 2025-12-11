"""
jaxboost: Differentiable gradient boosting with JAX.

Features:
- Soft oblivious trees with sigmoid routing
- Hyperplane splits for feature interactions
- Sparse splits for interpretable feature selection
- Neural ODE boosting for continuous-time dynamics
- GPU-efficient vectorized computation
- End-to-end training via optax

Quick Start:
    >>> from jaxboost import GBMTrainer
    >>> trainer = GBMTrainer(task="regression")
    >>> model = trainer.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

from jaxboost._version import __version__

# High-level API (recommended)
from jaxboost.training import GBMTrainer, TrainerConfig

# Low-level components
from jaxboost.aggregation import EulerBoosting, ODEBoosting, boosting_aggregate
from jaxboost.losses import mse_loss, sigmoid_binary_cross_entropy
from jaxboost.routing import soft_routing
from jaxboost.splits import (
    AxisAlignedSplit,
    AxisAlignedSplitParams,
    FactorizedInteractionSplit,
    FactorizedInteractionParams,
    HyperplaneSplit,
    HyperplaneSplitParams,
    InteractionDiscoverySplit,
    InteractionDiscoveryParams,
    SparseHyperplaneSplit,
    SparseHyperplaneSplitParams,
    TopKHyperplaneSplit,
    TopKHyperplaneSplitParams,
)
from jaxboost.structures import ObliviousTree, ObliviousTreeParams

__all__ = [
    "__version__",
    # High-level API
    "GBMTrainer",
    "TrainerConfig",
    # Splits
    "AxisAlignedSplit",
    "AxisAlignedSplitParams",
    "HyperplaneSplit",
    "HyperplaneSplitParams",
    "SparseHyperplaneSplit",
    "SparseHyperplaneSplitParams",
    "TopKHyperplaneSplit",
    "TopKHyperplaneSplitParams",
    "InteractionDiscoverySplit",
    "InteractionDiscoveryParams",
    "FactorizedInteractionSplit",
    "FactorizedInteractionParams",
    # Routing
    "soft_routing",
    # Structures
    "ObliviousTree",
    "ObliviousTreeParams",
    # Aggregation
    "boosting_aggregate",
    "ODEBoosting",
    "EulerBoosting",
    # Losses
    "mse_loss",
    "sigmoid_binary_cross_entropy",
]
