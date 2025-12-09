"""Split functions for jaxboost."""

from jaxboost.splits.axis_aligned import AxisAlignedSplit, AxisAlignedSplitParams
from jaxboost.splits.hyperplane import HyperplaneSplit, HyperplaneSplitParams

__all__ = [
    "AxisAlignedSplit",
    "AxisAlignedSplitParams",
    "HyperplaneSplit",
    "HyperplaneSplitParams",
]
