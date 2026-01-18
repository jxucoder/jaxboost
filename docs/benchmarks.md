# JAXBoost Benchmarks

This document presents benchmark comparisons demonstrating when custom loss functions provide advantages over standard XGBoost/LightGBM objectives.

## Key Findings

| Use Case | JAXBoost Advantage | Improvement |
|----------|-------------------|-------------|
| **Bounded Regression** | No native support | 9.5% MSE improvement |
| **Ordinal Regression** | No native support | Proper probabilistic outputs |
| **Multi-task with Missing Labels** | No native support | Handles sparse labels |

---

## 1. Bounded Regression (Proportion Prediction)

**Problem**: Predict targets bounded in [0, 1] (e.g., rates, percentages, proportions).

**Why Custom Loss?** XGBoost/LightGBM's MSE objectives can predict outside [0, 1], requiring post-hoc clipping which loses gradient information during training.

### Benchmark: County Health Rankings

Predicting county-level obesity rates from socioeconomic features.

**Dataset**: 3,194 US counties from [County Health Rankings 2024](https://www.countyhealthrankings.org/)

| Model | MSE | MAE | Out-of-Bounds |
|-------|-----|-----|---------------|
| **Soft Cross-Entropy** | **0.000570** | 0.0186 | 0% |
| Logit MSE | 0.000577 | 0.0188 | 0% |
| Native MSE | 0.000582 | 0.0186 | 0% |

**Improvement**: 2.1% (modest because target range is narrow: 17-52%)

### Benchmark: Synthetic Full-Range Proportions

Synthetic data with targets spanning full [0, 1] range (19% below 0.1, 18% above 0.9).

| Model | MSE | MAE | Out-of-Bounds |
|-------|-----|-----|---------------|
| **Soft Cross-Entropy** | **0.0181** | 0.101 | 0% |
| Logit MSE | 0.0188 | 0.106 | 0% |
| Native MSE + Clip | 0.0201 | 0.108 | 0% |
| Native MSE | 0.0201 | 0.108 | **4.9%** |

**Improvement**: 9.5% MSE reduction over clipped MSE

### JAXBoost Implementation

```python
from jax.nn import sigmoid
from jaxboost.objective import auto_objective

@auto_objective
def soft_crossentropy(y_pred, y_true):
    """Cross-entropy loss for proportion targets."""
    mu = sigmoid(y_pred)
    eps = 1e-6
    mu = jnp.clip(mu, eps, 1 - eps)
    y = jnp.clip(y_true, eps, 1 - eps)
    return -(y * jnp.log(mu) + (1 - y) * jnp.log(1 - mu))

# Use with XGBoost
model = xgb.train(params, dtrain, obj=soft_crossentropy.xgb_objective)

# Or LightGBM
model = lgb.train(params, train_data, fobj=soft_crossentropy.lgb_objective)
```

**Run the benchmark**:
```bash
JAX_PLATFORMS=cpu python examples/beta_regression_health.py --synthetic
```

---

## 2. Ordinal Regression

**Problem**: Predict ordered categories where distance between predictions matters (e.g., ratings 1-5, severity levels).

**Why Custom Loss?** XGBoost/LightGBM have no native ordinal regression. Common workarounds (regression + rounding, multi-class) ignore the ordinal structure.

### Benchmark: Wine Quality Dataset

Predicting wine quality ratings (3-8) using Quadratic Weighted Kappa (QWK) as the evaluation metric.

**Dataset**: 6,497 wines from [UCI Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)

| Model | QWK | Approach |
|-------|-----|----------|
| Regression + OptimizedRounder | **0.55** | Two-stage (Kaggle 1st place) |
| **JAXBoost Squared CDF** | **0.54** | Single-stage, probabilistic |
| **JAXBoost Ordinal NLL** | 0.53 | Single-stage, probabilistic |
| Native Multi-class | 0.51 | Ignores ordinal structure |
| Native Regression + Round | 0.48 | Ignores ordinal structure |

### Key Insight

The Kaggle-winning "OptimizedRounder" strategy directly optimizes QWK via threshold search (two-stage). JAXBoost ordinal objectives optimize differentiable surrogates in a single stage and provide proper probabilistic outputs.

| Approach | Pros | Cons |
|----------|------|------|
| OptimizedRounder | Directly optimizes QWK | Two-stage, no probabilities |
| JAXBoost Ordinal | Single-stage, probabilistic | Optimizes surrogate |

### JAXBoost Implementation

```python
from jaxboost.objective import ordinal_logit, qwk_ordinal, squared_cdf_ordinal

# Cumulative Link Model (standard ordinal regression)
obj = ordinal_logit(n_classes=6)

# QWK-aligned surrogate (Expected Quadratic Error)
obj = qwk_ordinal(n_classes=6)

# CRPS-based (Squared CDF)
obj = squared_cdf_ordinal(n_classes=6)

# Train
obj.init_thresholds_from_data(y_train)
model = xgb.train(params, dtrain, obj=obj.xgb_objective)
y_pred = obj.predict(model.predict(dtest))
```

**Run the benchmark**:
```bash
JAX_PLATFORMS=cpu python examples/ordinal_wine_quality.py
```

---

## 3. Multi-Task Learning with Missing Labels

**Problem**: Predict multiple related targets simultaneously, but some labels are missing for some samples.

**Why Custom Loss?** XGBoost's `multi_strategy=multi_output_tree` and LightGBM's multi-output cannot handle missing labels. Filling with 0 biases the model.

### JAXBoost Implementation

```python
from jaxboost.objective import MaskedMultiTaskObjective, multi_task_regression

# Create objective with masking support
obj = MaskedMultiTaskObjective(
    loss_fn=multi_task_regression,
    n_tasks=3
)

# Create mask (1 = valid, 0 = missing)
mask = ~np.isnan(y_train)

# Train with mask
model = xgb.train(
    params, 
    dtrain, 
    obj=obj.get_xgb_objective(mask=mask)
)
```

### Use Cases

- **Drug discovery**: Predicting multiple ADMET properties where not all assays are run for all compounds
- **Multi-label classification**: Some labels unknown for some samples
- **Sensor data**: Missing readings from some sensors

---

## When to Use Custom Losses

### Clear Wins (No native support)

| Problem | JAXBoost Solution |
|---------|-------------------|
| Bounded regression [0,1] | `soft_crossentropy`, `logit_mse` |
| Ordinal regression | `ordinal_logit`, `qwk_ordinal`, `squared_cdf_ordinal` |
| Multi-task with missing labels | `MaskedMultiTaskObjective` |
| Custom business metrics | `@auto_objective` decorator |

### Case-by-Case (May not beat tuned defaults)

| Problem | Notes |
|---------|-------|
| Imbalanced classification | Focal loss vs `scale_pos_weight` - depends on dataset |
| Robust regression | Huber vs `reg:pseudohubererror` - similar performance |
| Quantile regression | Similar to native quantile |

---

## Reproducing Results

All benchmarks can be reproduced:

```bash
# Install JAXBoost
pip install jaxboost

# Clone repo for examples
git clone https://github.com/yourrepo/jaxboost
cd jaxboost

# Run benchmarks
JAX_PLATFORMS=cpu python examples/beta_regression_health.py --synthetic
JAX_PLATFORMS=cpu python examples/ordinal_wine_quality.py
```

**Note**: Use `JAX_PLATFORMS=cpu` on macOS to avoid Metal GPU issues with JAX.

---

## Summary

JAXBoost's value proposition:

1. **Unique capabilities**: Bounded regression, ordinal regression, masked multi-task - problems where XGBoost/LightGBM have no native solution

2. **Rapid experimentation**: Write a loss function in 5 lines, get gradients and Hessians automatically

3. **Research tool**: Test novel objective functions without manual calculus

Custom objectives won't always beat highly-tuned defaults. The value is enabling solutions to problems that XGBoost/LightGBM cannot solve natively, and rapid iteration when exploring new ideas.
