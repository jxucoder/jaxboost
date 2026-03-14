# JAXBoost Code Review Plan

## Overview
Systematic review of the JAXBoost library — a JAX-powered automatic differentiation gradient boosting library (v0.5.0) supporting XGBoost and LightGBM.

---

## Phase 1: Core Engine Review

### 1.1 AutoObjective (`src/jaxboost/objective/auto.py`)
- [ ] Verify JAX autodiff correctness (grad/hessian computation)
- [ ] Review JIT compilation and vmap caching strategy for performance
- [ ] Check scalar vs array kwargs handling
- [ ] Validate XGBoost and LightGBM output format compatibility
- [ ] Review numerical stability (float32/float64 handling, clipping, epsilon guards)
- [ ] Check edge cases: NaN/Inf inputs, empty arrays, single-element batches

### 1.2 Public API (`src/jaxboost/__init__.py`)
- [ ] Verify all public exports match documented API
- [ ] Check for unintentional internal leakage
- [ ] Confirm `__all__` completeness

---

## Phase 2: Objectives Review

### 2.1 Regression (`src/jaxboost/objective/regression.py`)
- [ ] Review 10 loss functions: mse, huber, quantile, tweedie, asymmetric, log_cosh, pseudo_huber, mae_smooth, poisson, gamma
- [ ] Verify mathematical correctness of each loss
- [ ] Check parameter validation and default values
- [ ] Numerical stability for edge-case inputs (very large/small values)

### 2.2 Binary Classification (`src/jaxboost/objective/binary.py`)
- [ ] Review focal_loss, binary_crossentropy, weighted_binary_crossentropy, hinge_loss
- [ ] Verify sigmoid/logit transformations
- [ ] Check class weight handling

### 2.3 Multi-class Classification (`src/jaxboost/objective/multiclass.py`)
- [ ] Review softmax_cross_entropy, focal_multiclass, label_smoothing, class_balanced
- [ ] Verify MultiClassObjective decorator correctness
- [ ] Check multi-dimensional gradient/Hessian reshaping

### 2.4 Ordinal Regression (`src/jaxboost/objective/ordinal.py`)
- [ ] Review Cumulative Link Models (logit, probit)
- [ ] Verify SLACE paper implementations (sord, oll, slace objectives)
- [ ] Check threshold initialization from data
- [ ] Review hybrid objective combining NLL + EQE
- [ ] Validate QWK-aligned ordinal objective

### 2.5 Survival Analysis (`src/jaxboost/objective/survival.py`)
- [ ] Review AFT and Weibull AFT implementations
- [ ] Verify censoring handling

### 2.6 Multi-task & Multi-output
- [ ] `multi_task.py`: Review masking for missing labels, task-specific losses
- [ ] `multi_output.py`: Review Gaussian/Laplace NLL for uncertainty estimation
- [ ] Verify output shape conventions

---

## Phase 3: Metrics Review

### 3.1 Base Framework (`src/jaxboost/metric/base.py`)
- [ ] Review Metric base class interface
- [ ] Check make_metric factory function

### 3.2 Domain-specific Metrics
- [ ] `classification.py`: AUC, log-loss, accuracy, F1, precision, recall
- [ ] `regression.py`: MSE, RMSE, MAE, R²
- [ ] `ordinal.py`: QWK, ordinal MAE, ordinal accuracy, adjacent accuracy
- [ ] `bounded.py`: Bounded MSE, out-of-bounds metric
- [ ] Verify XGBoost/LightGBM eval metric format compliance

---

## Phase 4: Test Suite Review

### 4.1 Test Coverage Assessment
- [ ] Map each objective/metric to its corresponding test
- [ ] Identify any untested public API surface
- [ ] Review numerical gradient/Hessian verification in conftest.py

### 4.2 Test Quality
- [ ] Check for flaky tests (random seeds, tolerances)
- [ ] Review edge case coverage (`test_edge_cases.py`)
- [ ] Verify integration-level tests exist (end-to-end with XGBoost/LightGBM)
- [ ] Check ordinal test thoroughness (814 LOC — most extensive)

### 4.3 Test Infrastructure
- [ ] Review conftest.py fixtures and data generators
- [ ] Verify JAX CPU-only enforcement for CI reproducibility
- [ ] Check pytest markers and configuration

---

## Phase 5: Experimental Module Review

### 5.1 `src/jaxboost/experimental/__init__.py`
- [ ] Review import-time warnings (appropriate user messaging)
- [ ] Check Soft Decision Trees implementation
- [ ] Review Information Bottleneck Trees
- [ ] Review Mixture of Experts ensemble
- [ ] Check Neural ODE boosting
- [ ] Review Prior-Fitted Networks
- [ ] Assess stability/readiness of each experimental feature

---

## Phase 6: Infrastructure & Configuration

### 6.1 Build & Packaging (`pyproject.toml`)
- [ ] Verify dependency version constraints (JAX ≥0.4.20, optax ≥0.1.7, chex ≥0.1.8)
- [ ] Check optional dependency groups (dev, docs, benchmark, ode, polars, macos)
- [ ] Review ruff linting configuration and rules
- [ ] Verify hatchling build configuration and version sourcing

### 6.2 CI/CD (`.github/workflows/`)
- [ ] `tests.yml`: Python matrix (3.10-3.12), coverage upload
- [ ] `lint.yml`: ruff format + lint checks
- [ ] `docs.yml`: mkdocs build
- [ ] `integration.yml`: PyPI package validation
- [ ] `publish.yml`: Release automation security

### 6.3 Documentation
- [ ] `README.md`: Accuracy of examples and API references
- [ ] `docs/`: mkdocs site completeness
- [ ] `examples/`: Verify examples run correctly
- [ ] `mkdocs.yml`: Navigation structure and plugin config

---

## Phase 7: Cross-cutting Concerns

### 7.1 Code Quality
- [ ] Consistent coding style across modules
- [ ] Appropriate use of type hints
- [ ] Docstring completeness for public API

### 7.2 Performance
- [ ] JIT compilation effectiveness
- [ ] Memory usage for large datasets
- [ ] Benchmark results validation (`docs/benchmarks.md`)

### 7.3 Security & Safety
- [ ] No unsafe deserialization or code execution
- [ ] Dependency supply chain (pinned versions in uv.lock)

### 7.4 API Design
- [ ] Consistency across objective/metric interfaces
- [ ] Sklearn compatibility surface
- [ ] Breaking change risks for v1.0

---

## Review Priority Order
1. **Critical**: Core engine (Phase 1) — correctness of autodiff is foundational
2. **High**: Objectives (Phase 2) — mathematical correctness of loss functions
3. **High**: Tests (Phase 4) — confidence in correctness claims
4. **Medium**: Metrics (Phase 3), Infrastructure (Phase 6)
5. **Low**: Experimental (Phase 5), Cross-cutting (Phase 7)
