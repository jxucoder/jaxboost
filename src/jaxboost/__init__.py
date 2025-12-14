"""
jaxboost: Differentiable gradient boosting with JAX.

Two main capabilities:
1. Native soft decision trees - end-to-end differentiable training
2. XGBoost/LightGBM objective generator - automatic gradient/Hessian computation

Features:
- Soft oblivious trees with sigmoid routing
- Linear leaf trees for extrapolation beyond training data
- Hyperplane splits for feature interactions
- Sparse splits for interpretable feature selection
- Attention-based splits for input-dependent routing
- Information Bottleneck trees for principled regularization
- Mixture of Experts (MOE) with GBDT experts
- Neural ODE boosting for continuous-time dynamics
- GPU-efficient vectorized computation
- End-to-end training via optax
- Automatic objective functions for XGBoost/LightGBM

Quick Start (Soft Trees):
    >>> from jaxboost import GBMTrainer
    >>> trainer = GBMTrainer(task="regression")
    >>> model = trainer.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)

Quick Start (MOE Ensemble):
    >>> from jaxboost.ensemble import MOEEnsemble
    >>> moe = MOEEnsemble(num_experts=4, gating="tree")
    >>> params = moe.fit(X_train, y_train)
    >>> predictions = moe.predict(params, X_test)

Quick Start (XGBoost Objectives):
    >>> from jaxboost.objective import focal_loss, auto_objective
    >>> model = xgb.train(params, dtrain, obj=focal_loss.xgb_objective)
    >>>
    >>> @auto_objective
    ... def my_loss(y_pred, y_true):
    ...     return (y_pred - y_true) ** 2
    >>> model = xgb.train(params, dtrain, obj=my_loss.xgb_objective)
"""

from jaxboost._version import __version__

# High-level API (recommended)
from jaxboost.training import GBMTrainer, TrainerConfig

# XGBoost/LightGBM objective functions (convenient imports)
from jaxboost.objective import (
    AutoObjective,
    auto_objective,
    focal_loss,
    huber,
    quantile,
)

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
from jaxboost.structures import (
    ObliviousTree,
    ObliviousTreeParams,
    LinearLeafTree,
    LinearLeafParams,
    LinearLeafEnsemble,
)

# Ensemble methods (MOE)
from jaxboost.ensemble import (
    MOEEnsemble,
    MOEParams,
    LinearGating,
    MLPGating,
    TreeGating,
)

__all__ = [
    "__version__",
    # High-level API
    "GBMTrainer",
    "TrainerConfig",
    # XGBoost/LightGBM Objectives (see jaxboost.objective for more)
    "AutoObjective",
    "auto_objective",
    "focal_loss",
    "huber",
    "quantile",
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
    "LinearLeafTree",
    "LinearLeafParams",
    "LinearLeafEnsemble",
    # Aggregation
    "boosting_aggregate",
    "ODEBoosting",
    "EulerBoosting",
    # Losses (for soft trees)
    "mse_loss",
    "sigmoid_binary_cross_entropy",
    # Ensemble (MOE)
    "MOEEnsemble",
    "MOEParams",
    "LinearGating",
    "MLPGating",
    "TreeGating",
]
