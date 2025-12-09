"""Split functions for jaxboost."""

from jaxboost.splits.attention import (
    AttentionSplit,
    AttentionSplitParams,
    MultiHeadAttentionSplit,
    MultiHeadAttentionSplitParams,
)
from jaxboost.splits.axis_aligned import AxisAlignedSplit, AxisAlignedSplitParams
from jaxboost.splits.hyperplane import HyperplaneSplit, HyperplaneSplitParams

__all__ = [
    "AttentionSplit",
    "AttentionSplitParams",
    "MultiHeadAttentionSplit",
    "MultiHeadAttentionSplitParams",
    "AxisAlignedSplit",
    "AxisAlignedSplitParams",
    "HyperplaneSplit",
    "HyperplaneSplitParams",
]
