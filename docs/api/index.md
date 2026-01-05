# API Reference

Welcome to the jaxboost API reference documentation.

## Overview

jaxboost provides automatic objective functions for XGBoost, LightGBM, and CatBoost using JAX automatic differentiation.

### Core

- [`@auto_objective`](losses.md) - Decorator to create custom objectives
- [`AutoObjective`](losses.md) - Class for custom objective functions
- [`MultiClassObjective`](losses.md) - Multi-class classification objectives
- [`MultiOutputObjective`](losses.md) - Multi-output objectives (uncertainty)
- [`MaskedMultiTaskObjective`](losses.md) - Multi-task with missing labels

### Built-in Objectives

- [Binary Classification](losses.md#binary-classification) - `focal_loss`, `binary_crossentropy`, `hinge_loss`
- [Regression](losses.md#regression) - `mse`, `huber`, `quantile`, `tweedie`, `asymmetric`
- [Multi-class](losses.md#multi-class-classification) - `softmax_cross_entropy`, `focal_multiclass`
- [Survival](losses.md#survival-analysis) - `cox_partial_likelihood`, `aft`, `weibull_aft`
- [Multi-task](losses.md#multi-task-learning) - `multi_task_regression`, `multi_task_classification`
- [Uncertainty](losses.md#uncertainty-estimation) - `gaussian_nll`, `laplace_nll`

## Module Structure

```
jaxboost/
└── objective/      # Automatic objective functions
    ├── auto.py         # @auto_objective decorator
    ├── binary.py       # Binary classification
    ├── regression.py   # Regression objectives
    ├── multiclass.py   # Multi-class classification
    ├── multi_output.py # Multi-output (uncertainty)
    ├── multi_task.py   # Multi-task learning
    └── survival.py     # Survival analysis
```

## Quick Example

```python
import xgboost as xgb
from jaxboost import auto_objective, focal_loss

# Use built-in objective
model = xgb.train(params, dtrain, obj=focal_loss.xgb_objective)

# Create custom objective
@auto_objective
def my_loss(y_pred, y_true):
    return (y_pred - y_true) ** 2

model = xgb.train(params, dtrain, obj=my_loss.xgb_objective)
```
