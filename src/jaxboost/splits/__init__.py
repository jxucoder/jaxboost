"""Split functions for jaxboost."""

from jaxboost.splits.attention import (
    AttentionSplit,
    AttentionSplitParams,
    MultiHeadAttentionSplit,
    MultiHeadAttentionSplitParams,
)
from jaxboost.splits.axis_aligned import AxisAlignedSplit, AxisAlignedSplitParams
from jaxboost.splits.hyperplane import HyperplaneSplit, HyperplaneSplitParams
from jaxboost.splits.interaction_discovery import (
    FactorizedInteractionSplit,
    FactorizedInteractionParams,
    InteractionDiscoverySplit,
    InteractionDiscoveryParams,
)
from jaxboost.splits.sparse_hyperplane import (
    SparseHyperplaneSplit,
    SparseHyperplaneSplitParams,
    TopKHyperplaneSplit,
    TopKHyperplaneSplitParams,
)

__all__ = [
    "AttentionSplit",
    "AttentionSplitParams",
    "MultiHeadAttentionSplit",
    "MultiHeadAttentionSplitParams",
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
]
