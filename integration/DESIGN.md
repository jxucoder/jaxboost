# JAXBoost Integration Test Design

**Target Version**: v0.5.0 (PyPI)  
**Last Updated**: 2026-01-19

## 1. Goals

### Primary Goal
Validate that **end users** can successfully install and use JAXBoost from PyPI with XGBoost and LightGBM on their systems.

### Success Criteria
- User runs `pip install jaxboost` and it works
- All documented API patterns function correctly
- Tests pass on Mac (ARM + Intel), Linux, and Windows
- Python 3.12+ is fully supported (3.10-3.11 also work)

---

## 2. User Personas

### Persona A: Data Scientist on Mac M1/M2/M3
- Uses Apple Silicon Mac
- Wants GPU acceleration via Metal
- Python 3.12 via homebrew or conda
- Installs: `pip install jaxboost xgboost`

### Persona B: Data Scientist on Intel Mac
- Uses older Intel-based Mac
- CPU-only JAX (no Metal support)
- May need to pin JAX version for compatibility
- Installs: `pip install jaxboost xgboost jax==0.4.26 jaxlib==0.4.26`

### Persona C: ML Engineer on Linux
- Uses Ubuntu/Debian on cloud or local
- May have GPU (CUDA) or CPU-only
- Python 3.10-3.12 via system or conda
- Installs: `pip install jaxboost xgboost lightgbm`

### Persona D: Developer on Windows
- Uses Windows 10/11
- CPU-only JAX
- Python 3.10-3.12 via official installer
- Installs: `pip install jaxboost xgboost`

---

## 3. Test Categories

| ID | Category | Description | Priority | Blocking? |
|----|----------|-------------|----------|-----------|
| T1 | Imports | Package imports without errors | P0 | Yes |
| T2 | Regression | MSE, Huber, Quantile, etc. | P0 | Yes |
| T3 | Binary Classification | Focal loss, BCE, etc. | P0 | Yes |
| T4 | Custom Objectives | `@auto_objective` decorator | P0 | Yes |
| T5 | Ordinal Regression | CLM, QWK, SLACE objectives | P1 | No |
| T6 | Metrics | MSE, MAE, QWK metrics | P1 | No |
| T7 | LightGBM | LightGBM 4.x integration | P1 | No |
| T8 | Sklearn API | XGBRegressor/Classifier | P2 | No |
| T9 | Multi-task | Multi-output objectives | P2 | No |
| T10 | Edge Cases | Empty data, NaN handling | P2 | No |

---

## 4. API Patterns to Test

### 4.1 Simple Objectives (No Parameters)
```python
from jaxboost import huber, mse, focal_loss

# Direct property access
model = xgb.train(params, dtrain, obj=huber.xgb_objective)
```

**Test**: Verify `.xgb_objective` property returns callable that works with XGBoost.

### 4.2 Parameterized Objectives
```python
from jaxboost import quantile, tweedie, asymmetric

# Method 1: with_params() - creates new instance
q90 = quantile.with_params(q=0.9)
model = xgb.train(params, dtrain, obj=q90.xgb_objective)

# Method 2: get_xgb_objective() - inline
model = xgb.train(params, dtrain, obj=quantile.get_xgb_objective(q=0.9))
```

**Test**: Both methods produce equivalent results.

### 4.3 Custom Objectives
```python
from jaxboost import auto_objective
import jax.numpy as jnp

@auto_objective
def my_loss(y_pred, y_true, alpha=0.5):
    return alpha * (y_pred - y_true) ** 2

# Use default params
model = xgb.train(params, dtrain, obj=my_loss.xgb_objective)

# Use custom params
model = xgb.train(params, dtrain, obj=my_loss.get_xgb_objective(alpha=0.7))
```

**Test**: Decorator works, gradients/hessians are computed correctly via JAX autodiff.

### 4.4 Ordinal Objectives (Stateful)
```python
from jaxboost.objective import ordinal_logit, qwk_ordinal

# Create and initialize
ordinal = ordinal_logit(n_classes=6)
ordinal.init_thresholds_from_data(y_train)

# Train
model = xgb.train(params, dtrain, obj=ordinal.xgb_objective)

# Predict
raw_preds = model.predict(dtest)
probs = ordinal.predict_proba(raw_preds)  # (n_samples, n_classes)
classes = ordinal.predict(raw_preds)       # (n_samples,)
```

**Test**: Thresholds initialize correctly, predictions produce valid probabilities.

### 4.5 Multi-class Objectives
```python
from jaxboost import softmax_cross_entropy, focal_multiclass

# Factory function returns MultiClassObjective
loss = softmax_cross_entropy(n_classes=3)

params = {"num_class": 3, ...}
model = xgb.train(params, dtrain, obj=loss.xgb_objective)
```

**Test**: Works with XGBoost multi-class, predictions are correct shape.

### 4.6 Metrics
```python
from jaxboost.metric import mse_metric, qwk_metric

# Metrics are factory functions that return Metric objects
metric = mse_metric()

model = xgb.train(
    {'disable_default_eval_metric': 1, ...},
    dtrain,
    obj=huber.xgb_objective,
    custom_metric=metric.xgb_metric,  # Note: .xgb_metric property
    evals=[(dtest, 'test')]
)
```

**Test**: Metrics compute correctly, integrate with XGBoost eval.

### 4.7 LightGBM Integration (v4.x API)
```python
import lightgbm as lgb
from jaxboost import huber

# LightGBM 4.x: objective goes in params dict
params = {
    "objective": huber.lgb_objective,  # NOT fobj parameter
    "metric": "None",
    ...
}
model = lgb.train(params, train_data, num_boost_round=100)
```

**Test**: Works with LightGBM 4.x API, no `fobj` parameter.

### 4.8 Sklearn API
```python
from xgboost import XGBRegressor, XGBClassifier
from jaxboost import huber, focal_loss

reg = XGBRegressor(objective=huber.sklearn_objective, ...)
reg.fit(X_train, y_train)
```

**Test**: `.sklearn_objective` works with sklearn-style API.

---

## 5. Platform-Specific Considerations

### 5.1 JAX Version Matrix

| Platform | JAX Version | JAXLib Version | Notes |
|----------|-------------|----------------|-------|
| macOS ARM64 | >=0.4.34 | >=0.4.34 | Metal support via jax-metal |
| macOS x86_64 | 0.4.26 | 0.4.26 | Last version with Intel wheels |
| Linux x86_64 | >=0.4.20 | >=0.4.20 | Full support |
| Linux ARM64 | >=0.4.20 | >=0.4.20 | Full support |
| Windows x86_64 | >=0.4.20 | >=0.4.20 | CPU only |

### 5.2 Known Platform Issues

1. **Intel Mac + JAX 0.5+**: No wheels available, must use older JAX
2. **Windows + GPU**: JAX doesn't support CUDA on Windows
3. **jax-metal**: Optional dependency, should not be required

### 5.3 CI Matrix

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ["3.10", "3.11", "3.12"]
    exclude:
      # Skip slow combinations if needed
      - os: windows-latest
        python-version: "3.10"
```

---

## 6. Test Implementation Plan

### Phase 1: Core Tests (P0)
1. `test_t1_imports.py` - All imports work
2. `test_t2_regression.py` - Regression objectives
3. `test_t3_classification.py` - Binary classification
4. `test_t4_custom.py` - Custom objectives

### Phase 2: Feature Tests (P1)
5. `test_t5_ordinal.py` - Ordinal regression
6. `test_t6_metrics.py` - Evaluation metrics
7. `test_t7_lightgbm.py` - LightGBM integration

### Phase 3: Extended Tests (P2)
8. `test_t8_sklearn.py` - Sklearn API
9. `test_t9_multitask.py` - Multi-task learning
10. `test_t10_edge_cases.py` - Edge cases

---

## 7. Test Data Strategy

### Synthetic Data (Primary)
- Use `sklearn.datasets.make_regression`, `make_classification`
- Deterministic with `random_state=42`
- Small datasets (500-1000 samples) for fast tests

### Real Data (Optional, for benchmarks)
- UCI Wine Quality (ordinal)
- UCI Adult (binary classification)
- Boston Housing (regression) - deprecated, use California Housing

---

## 8. Assertions & Validation

### For Regression
- Model trains without error
- Predictions are finite (no NaN/Inf)
- MSE decreases during training
- MSE < reasonable threshold (e.g., 50000 for scaled data)

### For Classification
- Model trains without error
- Predictions are in valid range
- Accuracy > random baseline (e.g., > 0.5 for binary)

### For Ordinal
- Thresholds are strictly increasing
- Probabilities sum to 1.0
- Predictions are valid class indices

### For Metrics
- Metric values are finite
- Metric appears in XGBoost eval output
- Values change during training

---

## 9. Open Questions

1. **PyPI Version**: Should tests install specific version or latest?
   - Recommendation: Test against latest, pin in CI for reproducibility

2. **JAX Version Handling**: How to handle Intel Mac JAX compatibility?
   - Recommendation: Document workaround, don't block tests

3. **LightGBM API Change**: Should we support old `fobj` API?
   - Recommendation: Support both in code, test only v4.x in integration

4. **Multi-class Predictions**: XGBoost returns class labels, not logits
   - Recommendation: Document behavior, tests should handle both

---

## 10. Next Steps

1. [x] Review and approve this design
2. [x] Create `conftest.py` with shared fixtures
3. [x] Implement Phase 1 tests (P0)
4. [x] Set up GitHub Actions workflow
5. [x] Implement Phase 2 tests (P1)
6. [x] Implement Phase 3 tests (P2)
7. [x] Add to CI/CD pipeline
