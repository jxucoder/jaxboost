"""Ensemble methods for jaxboost.

This module provides ensemble architectures beyond simple boosting:
- Mixture of Experts (MOE) with GBDT experts
- Hybrid MOE with external GBDT libraries (XGBoost, LightGBM, CatBoost)
- Various gating mechanisms (Linear, MLP, Tree)
"""

from jaxboost.ensemble.gating import (
    GatingFn,
    LinearGating,
    LinearGatingParams,
    MLPGating,
    MLPGatingParams,
    TreeGating,
    TreeGatingParams,
    load_balance_loss,
    router_z_loss,
    sparse_top_k,
)
from jaxboost.ensemble.moe import (
    MOEEnsemble,
    MOEParams,
    TaskAwareMOEEnsemble,
)
from jaxboost.ensemble.hybrid_moe import (
    EMMOE,
    EMConfig,
    HardEMMOE,
    SparseEMMOE,
    # Backward compatibility
    HybridMOE,
    HybridMOEConfig,
    # Expert factories
    create_xgboost_expert,
    create_lightgbm_expert,
    create_catboost_expert,
)

__all__ = [
    # Gating
    "GatingFn",
    "LinearGating",
    "LinearGatingParams",
    "MLPGating",
    "MLPGatingParams",
    "TreeGating",
    "TreeGatingParams",
    # Utilities
    "sparse_top_k",
    "load_balance_loss",
    "router_z_loss",
    # Differentiable MOE
    "MOEEnsemble",
    "MOEParams",
    "TaskAwareMOEEnsemble",
    # EM-MOE (external experts)
    "EMMOE",
    "EMConfig",
    "HardEMMOE",
    "SparseEMMOE",
    "HybridMOE",  # alias for EMMOE
    "HybridMOEConfig",  # alias for EMConfig
    "create_xgboost_expert",
    "create_lightgbm_expert",
    "create_catboost_expert",
]

