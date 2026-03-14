# JAXBoost Code Review Findings

**Version reviewed**: 0.5.0
**Date**: 2026-03-14
**Test suite**: 233/233 tests passing

---

## Executive Summary

JAXBoost is a well-structured library with a clean core abstraction (`AutoObjective`) and impressive breadth of loss functions. The autodiff approach is sound and the API design is consistent. The main areas for improvement are: (1) missing `__all__` synchronization between `__init__.py` files, (2) a performance inefficiency in gradient/Hessian computation, (3) significant code duplication in ordinal SLACE objectives, and (4) test coverage gaps in numerical verification.

**Overall assessment**: Good quality alpha-stage library with solid foundations. The issues identified are mostly about hardening for production use rather than fundamental design flaws.

---

## Phase 1: Core Engine

### 1.1 AutoObjective (`auto.py`) — Generally Sound

**Strengths:**
- Clean separation of scalar vs array kwargs via `_split_kwargs()`
- Smart vmap caching keyed by array kwargs pattern (avoids recompilation)
- Proper dtype handling: computation in float32, output in float64

**Issues:**

**[P1-BUG] Hessian computed via nested `jax.grad` instead of `jax.hessian` or forward-over-reverse**
`auto.py:57` — The Hessian is computed as `jax.grad(lambda: jax.grad(loss)(...))`. This works correctly but is inefficient for JAX — forward-over-reverse (`jax.jacfwd(jax.grad(...))`) is generally faster for scalar-to-scalar second derivatives. For the single-sample scalar case here the difference is small, but it's worth noting.

**[P2-PERF] `grad_hess()` calls `gradient()` and `hessian()` separately**
`auto.py:188-189` — Each call separately converts inputs, splits kwargs, and looks up vmap cache. These two calls could be fused into a single vmap pass using `jax.value_and_grad` or computing both in one `_compute_grad_and_hess_single` function, halving the JIT/vmap overhead.

**[P3-EDGE] `_split_kwargs` array detection is fragile**
`auto.py:93-98` — An array kwarg is detected by checking `isinstance(v, (np.ndarray, jax.Array)) and len(v) == n_samples`. This would incorrectly classify a scalar numpy array or a tuple of length `n_samples` as an array kwarg. A Python list of floats would be treated as scalar. Consider checking `np.ndim(v) >= 1` explicitly.

### 1.2 Public API (`__init__.py`) — Missing Exports

**[P1-BUG] `__all__` in top-level `__init__.py` missing ordinal exports**
The top-level `src/jaxboost/__init__.py:94-136` exports `AutoObjective`, all binary/regression/multiclass/survival/multi-task/multi-output objectives, but **does not export any ordinal objectives** (`ordinal_logit`, `ordinal_probit`, `OrdinalObjective`, `SORDObjective`, etc.). These are available in `jaxboost.objective` but not in the top-level `jaxboost` namespace. This is inconsistent — either all objective categories should be exported or ordinal should be documented as `from jaxboost.objective import ...`.

**[P2-DOC] Module docstring lists `cox_partial_likelihood` which doesn't exist**
`__init__.py:28` mentions `cox_partial_likelihood` in the "Available Objectives" list, but this function is not implemented anywhere in the codebase. This is misleading.

**[P3-QUALITY] `poisson` and `gamma` missing from top-level `__all__`**
These are exported from `jaxboost.objective.__init__` but not from `jaxboost.__init__`. The top-level import statement at line 56-92 doesn't include them either.

---

## Phase 2: Objectives

### 2.1 Regression (`regression.py`) — Correct

All 10 loss functions are mathematically correct. Specific observations:

- **MSE**: Standard `(y_pred - y_true)^2`. No `0.5*` factor, so gradient is `2*(y_pred - y_true)` not `(y_pred - y_true)`. This is fine (XGBoost's learning rate compensates) but worth documenting.
- **Quantile**: Uses smooth approximation with `alpha * error^2` regularization term. Good — ensures non-zero Hessian.
- **Tweedie**: Correct deviance formula with `mu = exp(y_pred)` and clipping. The `jnp.clip(mu, 1e-10, 1e10)` is appropriate.
- **Poisson/Gamma**: Correct log-link formulations.
- **Log-cosh**: Note that `jnp.cosh(error)` can overflow for very large errors (>~89). JAX's implementation handles this gracefully via softplus, but worth awareness.

No issues found in this module.

### 2.2 Binary Classification (`binary.py`) — Correct

- **Focal loss**: Correct implementation following Lin et al. 2017. Clipping `p` at `[1e-7, 1-1e-7]` is appropriate.
- **Hinge loss**: Uses `softplus(margin - z)` as smooth approximation — good choice for non-zero Hessians.

No issues found.

### 2.3 Multi-class (`multiclass.py`) — Minor Issue

**[P3-PERF] Diagonal Hessian uses O(n_classes) separate grad calls**
`multiclass.py:68-81` — The diagonal Hessian is computed by calling `jax.grad(jax.grad(loss_i(i)))` for each class independently via `jax.vmap(hess_ii)`. This computes the full gradient `n_classes` times. For small `n_classes` this is fine, but for large `n_classes` it could be expensive. `jax.jacfwd(jax.grad(...))` followed by `jnp.diag()` would be more efficient.

**[P3-MISSING] No `lgb_objective` property**
`MultiClassObjective` has `xgb_objective` but no `lgb_objective`, unlike `AutoObjective`. The module docstring claims LightGBM support.

### 2.4 Ordinal Regression (`ordinal.py`) — Significant Duplication

**[P1-QUALITY] Massive code duplication across SLACE-family objectives**
`SORDObjective`, `OLLObjective`, and `SLACEObjective` (lines 843-1327) each duplicate:
- `_ensure_2d()` (identical in all 3)
- `gradient()` / `hessian()` / `grad_hess()` (near-identical pattern)
- `get_xgb_objective()` / `xgb_objective` property (identical)
- `predict()` (identical)
- `_probs_grad_hess()` / `sklearn_objective` (same structure)

These 3 classes are ~490 lines that could be refactored into a common `MultiClassOrdinalObjective` base class with ~150 lines, where each subclass only overrides `_loss_single()`. This would reduce bugs and maintenance burden.

**[P2-QUALITY] SORD/OLL/SLACE don't inherit from OrdinalObjective**
Unlike `QWKOrdinalObjective` and `SquaredCDFObjective`, the SLACE-paper objectives are standalone classes with no shared base. They can't use the metric properties (`qwk_metric`, `mae_metric`, etc.) that `OrdinalObjective` provides.

**[P3-QUALITY] Hessian computation uses `jacfwd(grad(...))` for SORD/OLL/SLACE**
These compute the full Jacobian matrix and take the diagonal, which is O(n_classes^2) work when only the diagonal is needed. This is correct but wasteful.

**[P3-EDGE] `_ensure_2d` silently returns zeros for wrong-shaped input**
`ordinal.py:871` — If `y_pred.size != n_samples * n_classes`, it returns `np.zeros(...)`. This silently masks shape errors rather than raising an exception.

### 2.5 Survival Analysis (`survival.py`) — Correct

- **AFT**: Proper log-normal formulation with uncensored/right-censored/interval-censored branches.
- **Weibull AFT**: Correct Weibull survival function. The `k=1` special case (exponential) is properly handled.
- Both use `jnp.where` for branch selection — correct for JAX tracing.

**[P3-EDGE] Right-censoring detection uses `upper > 1e10` instead of `jnp.isinf`**
`survival.py:60,121` — This is intentional (documented as "inf check") but could fail if users pass `upper = 1e11` to indicate right-censoring.

### 2.6 Multi-task & Multi-output — Generally Sound

**[P2-QUALITY] MaskedMultiTaskObjective Hessian clamped to 1e-6 even for masked entries**
`multi_task.py:150` — `hess = jnp.maximum(hess, 1e-6)` is applied after masking. This means masked (missing) entries get `hess = 1e-6` instead of 0. While small, this causes XGBoost to make tiny updates for missing-label tasks, which is incorrect. The clamp should be: `hess = jnp.where(mask > 0, jnp.maximum(hess, 1e-6), 0.0)`.

**[P3-QUALITY] `multi_task_quantile` overrides `_compute_grad_hess_single` with manual gradients**
`multi_task.py:561-585` — The `QuantileMTL` subclass computes gradients analytically rather than using JAX autodiff. This works but breaks the pattern and means the `task_loss_fn` is ignored. It also sets `hess = 1.0` as a constant, which is correct for quantile loss but inconsistent with the "autodiff everything" philosophy.

**[P3-MISSING] `MultiOutputObjective` has no `lgb_objective`**
Same issue as MultiClassObjective — documented LightGBM support but no `lgb_objective` property.

---

## Phase 3: Metrics

### Overall: Clean and Correct

The metrics module is well-designed with a clean `Metric` base class supporting both XGBoost and LightGBM interfaces.

**[P3-QUALITY] Duplicated `_sigmoid()` function**
`classification.py:12-14` and `bounded.py:12-14` both define identical `_sigmoid()` functions. Should be in `base.py` or a shared utils module.

**[P3-QUALITY] AUC computation is custom instead of using scipy**
`classification.py:17-50` — The `_compute_auc()` function implements AUC from scratch. While correct, `scipy.metrics` or a well-tested library function would be more robust for edge cases (e.g., all predictions identical, single class).

**[P3-EDGE] `r2_metric` returns 0.0 when all targets are identical**
`regression.py:102-103` — When `ss_tot < 1e-10`, returns 0.0. The conventional R^2 is undefined in this case; returning `float('nan')` or `1.0` (for a perfect constant predictor) might be more appropriate depending on context.

---

## Phase 4: Test Suite

### 4.1 Coverage Assessment — Good but Gaps Exist

**233 tests passing**, covering all modules. Key findings:

**[P1-GAP] Most objectives lack finite-difference numerical verification**
`conftest.py` provides `numerical_gradient()` and `numerical_hessian_diag()` utilities, but they're only used for MSE, Huber, and log-loss in `test_auto_objective.py`. The following objectives have NO numerical gradient verification against finite differences:
- All regression losses except MSE (huber, quantile, tweedie, etc.)
- Focal loss
- All multiclass objectives
- All ordinal objectives
- All survival objectives
- All multi-task objectives

This is the single biggest testing gap. Autodiff correctness should be verified numerically for each loss function.

**[P2-GAP] No end-to-end integration tests in the main test suite**
The `integration/` directory exists but is tested separately via CI against PyPI. The main test suite never actually trains an XGBoost or LightGBM model, so objective format compatibility is untested in regular development.

**[P2-GAP] SLACE-family objectives only test shapes, not gradient correctness**
`test_ordinal.py` tests SORD/OLL/SLACE for gradient shapes and basic properties but never verifies the actual gradient values against known analytical formulas or numerical approximations.

**[P3-GAP] Metric values not verified against known reference implementations**
`test_metrics.py` checks shapes and basic properties (perfect prediction = 1.0) but never compares metric values against sklearn or other reference implementations.

### 4.2 Test Quality

- Reproducible: All tests use `np.random.seed(42)` or parameterized seeds
- Not flaky: 233/233 pass consistently
- Some tolerances are very loose: `rtol=0.2` in pseudo-Huber test, `rtol=0.1` in hybrid ordinal test

---

## Phase 5: Experimental Module

**Assessment**: Research-grade code, appropriately marked with import-time warnings.

**[P3-INFO] Large monolithic `__init__.py`**
The entire experimental module is in a single `__init__.py` file with heavy try/except blocks for optional dependencies. This makes it hard to import individual components without triggering warnings for unrelated features. Consider splitting into submodules.

---

## Phase 6: Infrastructure

### 6.1 Build & Packaging — Sound

- Hatchling build system, version sourced from `_version.py`
- Dependency bounds are reasonable (lower bounds, no upper pins)
- Ruff configuration is appropriate

**[P2-CONFIG] `dependency-groups.dev` duplicates `project.optional-dependencies.dev`**
`pyproject.toml:97-104` has a `[dependency-groups] dev` section with different packages (matplotlib, openml, pandas, scikit-survival, torch) than the `[project.optional-dependencies] dev` section (pytest, ruff, mypy). This is confusing — the dependency-group `dev` should probably be named `research` or `experiments`.

**[P3-CONFIG] `benchmark` optional dep missing `lightgbm`**
`pyproject.toml:49-52` — The `benchmark` group includes `xgboost` and `scikit-learn` but not `lightgbm`, despite the library supporting both.

### 6.2 CI/CD — Good

- Multi-Python (3.10-3.12) testing
- Coverage via codecov
- Separate lint, docs, integration, publish workflows
- Trusted publishing to PyPI

No issues found.

---

## Phase 7: Cross-cutting Concerns

### API Consistency

**[P2-CONSISTENCY] Inconsistent LightGBM support across objective types**
| Objective Type | `xgb_objective` | `lgb_objective` | `sklearn_objective` |
|---|---|---|---|
| AutoObjective | Yes | Yes | Yes |
| MultiClassObjective | Yes | **No** | No |
| MultiOutputObjective | Yes | **No** | No |
| MaskedMultiTaskObjective | Yes | Yes | No |
| OrdinalObjective | Yes | **No** | No |
| SORD/OLL/SLACE | Yes | **No** | Yes |

Three of six objective types are missing `lgb_objective`. Users who need LightGBM with multiclass, multi-output, or ordinal objectives have no path.

### Code Quality

- Consistent style throughout (ruff-enforced)
- Good type annotations on all public methods
- Comprehensive docstrings with examples
- Clean separation of concerns between modules

### Security

No issues. The library is computation-only with no I/O, deserialization, or network access.

---

## Summary: Issues by Priority

| Priority | Count | Description |
|---|---|---|
| P1 (Bug/Critical) | 3 | Missing ordinal exports, Hessian clamping with masks, missing numerical verification |
| P2 (Important) | 6 | Fused grad_hess, docstring phantom API, missing lgb_objective, config confusion, no integration tests, SLACE gradient verification |
| P3 (Minor) | 12 | Duplicated code, perf opportunities, edge cases, minor inconsistencies |

### Top 5 Recommended Actions

1. **Add ordinal objectives to top-level `__all__`** or document the import path
2. **Fix `MaskedMultiTaskObjective` Hessian clamping** to respect mask (bug)
3. **Add finite-difference gradient verification** for all loss functions in tests
4. **Refactor SORD/OLL/SLACE** into a shared base class to eliminate ~340 lines of duplication
5. **Add `lgb_objective`** to `MultiClassObjective`, `MultiOutputObjective`, and ordinal objectives
