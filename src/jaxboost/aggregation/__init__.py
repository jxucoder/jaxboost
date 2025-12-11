"""Aggregation functions for jaxboost."""

from jaxboost.aggregation.boosting import boosting_aggregate
from jaxboost.aggregation.ode_boosting import (
    EulerBoosting,
    ODEBoosting,
    ODEBoostingParams,
)

__all__ = [
    "boosting_aggregate",
    "ODEBoosting",
    "ODEBoostingParams",
    "EulerBoosting",
]
