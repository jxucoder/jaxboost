<p align="center">
  <img src="assets/logo.png" alt="JAXBoost" width="300">
</p>

<h2 align="center">JAX Autodiff for XGBoost/LightGBM Objectives</h2>

<p align="center">
  <em>Write a loss function, get gradients and Hessians automatically. No manual derivation needed.</em>
</p>

---

## Features

- **Automatic Gradients** — JAX computes first derivatives for any loss function
- **Automatic Hessians** — Second derivatives computed automatically via autodiff
- **Built-in Objectives** — Focal loss, Huber, quantile, survival, and 20+ more
- **Works Everywhere** — XGBoost and LightGBM compatible

---

## Installation

```bash
pip install jaxboost
```

!!! note "JAX Backend"
    JAXBoost requires JAX. If not installed, it will use the CPU backend by default. For GPU support, install JAX with CUDA following the [JAX installation guide](https://github.com/google/jax#installation).

---

## Quick Example

=== "Built-in Objectives"

    ```python
    import xgboost as xgb
    from jaxboost import focal_loss, huber, quantile

    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {"max_depth": 4, "eta": 0.1}

    # Focal loss for imbalanced classification
    model = xgb.train(params, dtrain, obj=focal_loss.xgb_objective)

    # Huber loss for robust regression
    model = xgb.train(params, dtrain, obj=huber.xgb_objective)

    # Quantile regression (90th percentile)
    model = xgb.train(params, dtrain, obj=quantile(0.9).xgb_objective)
    ```

=== "Custom Objectives"

    ```python
    import jax.numpy as jnp
    from jaxboost import auto_objective

    @auto_objective
    def my_loss(y_pred, y_true):
        """Custom asymmetric loss."""
        error = y_true - y_pred
        return jnp.where(error > 0, 2 * error**2, error**2)

    model = xgb.train(params, dtrain, obj=my_loss.xgb_objective)
    ```

---

## Why JAXBoost?

| Traditional Approach | JAXBoost |
|---------------------|----------|
| Derive gradients by hand | Write loss, get gradients free |
| Derive Hessians by hand | Write loss, get Hessians free |
| Error-prone manual math | JAX autodiff is correct by construction |
| One loss = hours of work | One loss = 5 lines of code |

!!! success "Real Example"
    Focal loss requires computing:
    
    - **Gradient**: \( -(1-p)^\gamma \log(p) - \gamma(1-p)^{\gamma-1}p\log(p) + ... \)
    - **Hessian**: Even more complex...
    
    With JAXBoost, just write `(1-p)**gamma * cross_entropy(p, y)` and autodiff handles the rest.

---

## Available Objectives

| Category | Objectives |
|----------|------------|
| **Regression** | `mse`, `huber`, `quantile`, `tweedie`, `asymmetric`, `log_cosh`, `pseudo_huber`, `mae_smooth`, `poisson`, `gamma` |
| **Binary Classification** | `focal_loss`, `binary_crossentropy`, `weighted_binary_crossentropy`, `hinge_loss` |
| **Multi-class** | `softmax_cross_entropy`, `focal_multiclass`, `label_smoothing`, `class_balanced` |
| **Survival** | `aft`, `weibull_aft` |
| **Multi-task** | `multi_task_regression`, `multi_task_classification`, `multi_task_huber`, `multi_task_quantile` |
| **Uncertainty** | `gaussian_nll`, `laplace_nll` |

---

## Next Steps

- **[Quick Start Guide](getting-started/quickstart.md)** — Get up and running in minutes
- **[API Reference](api/index.md)** — Detailed documentation for all objectives
- **[Research Notes](research.md)** — Archived research on differentiable trees
