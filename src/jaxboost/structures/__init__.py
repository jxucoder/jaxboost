"""Tree structures for jaxboost."""

from jaxboost.structures.oblivious import ObliviousTree, ObliviousTreeParams
from jaxboost.structures.linear_leaf import (
    LinearLeafTree,
    LinearLeafParams,
    LinearLeafEnsemble,
)

__all__ = [
    "ObliviousTree",
    "ObliviousTreeParams",
    "LinearLeafTree",
    "LinearLeafParams",
    "LinearLeafEnsemble",
]
