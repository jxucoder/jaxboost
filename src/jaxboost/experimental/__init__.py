"""
Experimental components for jaxboost.

This module contains research-stage features that are not yet stable.
APIs may change without notice between versions.

Components:
- Soft decision trees (GBMTrainer, ObliviousTree, LinearLeafTree)
- Information Bottleneck trees (IBTree)
- Mixture of Experts ensembles (MOEEnsemble)
- Neural ODE boosting (ODEBoosting)
- Empirical Prior-Fitted Networks (EmpiricalPFNTrainer)
- Advanced split functions (attention, sparse, interaction discovery)

Example:
    >>> from jaxboost.experimental import GBMTrainer
    >>> trainer = GBMTrainer(task="regression")
    >>> model = trainer.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)

Warning:
    These APIs are experimental and may change without deprecation warnings.
    For production use, prefer the stable `jaxboost.objective` module.
"""

import warnings as _warnings

# Emit warning on import
_warnings.warn(
    "jaxboost.experimental contains research-stage features. "
    "APIs may change without notice. For production, use jaxboost.objective.",
    stacklevel=2,
)

# =============================================================================
# Soft Decision Trees (High-level API)
# =============================================================================
# =============================================================================
# Split Functions
# =============================================================================
from jaxboost.splits import (
    # Basic splits
    AxisAlignedSplit,
    AxisAlignedSplitParams,
    FactorizedInteractionParams,
    FactorizedInteractionSplit,
    HyperplaneSplit,
    HyperplaneSplitParams,
    InteractionDiscoveryParams,
    # Advanced splits
    InteractionDiscoverySplit,
    # Sparse/regularized splits
    SparseHyperplaneSplit,
    SparseHyperplaneSplitParams,
    TopKHyperplaneSplit,
    TopKHyperplaneSplitParams,
)

# =============================================================================
# Tree Structures
# =============================================================================
from jaxboost.structures import (
    LinearLeafEnsemble,
    LinearLeafParams,
    LinearLeafTree,
    ObliviousTree,
    ObliviousTreeParams,
)
from jaxboost.training import GBMTrainer, TrainerConfig

# Attention-based splits (if available)
try:
    from jaxboost.splits.attention import AttentionSplit, AttentionSplitParams
except ImportError:
    pass

# =============================================================================
# Routing Functions
# =============================================================================
# =============================================================================
# Aggregation Methods
# =============================================================================
from jaxboost.aggregation import (
    EulerBoosting,
    ODEBoosting,
    boosting_aggregate,
)

# =============================================================================
# Mixture of Experts
# =============================================================================
from jaxboost.ensemble import (
    LinearGating,
    MLPGating,
    MOEEnsemble,
    MOEParams,
    TreeGating,
)

# =============================================================================
# Losses (for soft tree training)
# =============================================================================
from jaxboost.losses import mse_loss, sigmoid_binary_cross_entropy
from jaxboost.routing import soft_routing

# Hybrid MOE (XGBoost/LightGBM experts)
try:
    from jaxboost.ensemble.hybrid_moe import (
        EMMOE,
        EMConfig,
        create_catboost_expert,
        create_lightgbm_expert,
        create_xgboost_expert,
    )
except ImportError:
    pass

# =============================================================================
# Information Bottleneck Trees
# =============================================================================
from jaxboost.ib import IBTree, IBTreeEnsemble, IBTreeParams

# =============================================================================
# Empirical Prior-Fitted Networks (PFN)
# =============================================================================
from jaxboost.pfn import (
    # DataPFN
    DataPFN,
    DataPFNConfig,
    # Empirical prior
    EmpiricalPrior,
    # Prior generator
    PriorGenerator,
    # Structure discovery
    StructureStats,
    discover_structure,
    discover_structure_fast,
    generate_dataset_from_prior,
    learn_prior_from_datasets,
    learn_prior_from_single_dataset,
    train_data_pfn,
)

__all__ = [
    # High-level API
    "GBMTrainer",
    "TrainerConfig",
    # Structures
    "ObliviousTree",
    "ObliviousTreeParams",
    "LinearLeafTree",
    "LinearLeafParams",
    "LinearLeafEnsemble",
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
    # Aggregation
    "boosting_aggregate",
    "EulerBoosting",
    "ODEBoosting",
    # Losses
    "mse_loss",
    "sigmoid_binary_cross_entropy",
    # MOE
    "MOEEnsemble",
    "MOEParams",
    "LinearGating",
    "MLPGating",
    "TreeGating",
    # IB Trees
    "IBTree",
    "IBTreeParams",
    "IBTreeEnsemble",
    # PFN
    "StructureStats",
    "discover_structure",
    "discover_structure_fast",
    "EmpiricalPrior",
    "learn_prior_from_datasets",
    "learn_prior_from_single_dataset",
    "PriorGenerator",
    "generate_dataset_from_prior",
    "DataPFN",
    "DataPFNConfig",
    "train_data_pfn",
]
