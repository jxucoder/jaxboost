# JAXBoost Integration Tests

End-user integration tests for JAXBoost v0.5.0+. See [DESIGN.md](DESIGN.md) for detailed test design.

## Quick Start

```bash
# Option 1: Fresh virtual environment
python -m venv /tmp/jaxboost_test
source /tmp/jaxboost_test/bin/activate
pip install jaxboost xgboost lightgbm scikit-learn pytest
pytest -v

# Option 2: Using uv (from this directory)
uv run pytest -v

# Option 3: Run specific test category
pytest test_t1_imports.py -v  # Just imports
pytest -m "not slow" -v       # Skip slow tests
```

## Test Structure

| Test File | Category | Tests | Priority |
|-----------|----------|-------|----------|
| `test_t1_imports.py` | Core imports | 12 | P0 |
| `test_t2_regression.py` | Regression objectives | 8 | P0 |
| `test_t3_classification.py` | Binary/multi-class | 7 | P0 |
| `test_t4_custom.py` | Custom objectives | 5 | P0 |
| `test_t5_ordinal.py` | Ordinal regression | 7 | P1 |
| `test_t6_metrics.py` | Evaluation metrics | 6 | P1 |
| `test_t7_lightgbm.py` | LightGBM integration | 6 | P1 |
| `test_t8_sklearn.py` | Sklearn API | 5 | P2 |
| `test_t9_multitask.py` | Multi-task learning | 4 | P2 |
| `test_t10_edge_cases.py` | Edge cases | 9 | P2 |

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| macOS ARM64 (M1/M2/M3) | Full | Metal GPU optional |
| macOS x86_64 (Intel) | Full | Requires JAX 0.4.26 |
| Linux x86_64 | Full | CPU/GPU |
| Windows x86_64 | CPU | No GPU support |

## Troubleshooting

### Intel Mac (x86_64)

JAX 0.5+ does not provide wheels for Intel Macs. Pin to the last supported version:

```bash
pip install jaxboost xgboost jax==0.4.26 jaxlib==0.4.26
```

### Windows

JAX on Windows is CPU-only. GPU acceleration is not supported. Install normally:

```bash
pip install jaxboost xgboost
```

### Known Issues

| Issue | Workaround |
|-------|------------|
| `log_cosh` produces NaN on large targets | Use `huber` or scale your target values |
| Ordinal objectives don't support LightGBM | Use XGBoost for ordinal regression |

### Testing Against Local Development Version

To test against your local JAXBoost changes instead of PyPI:

```bash
# From the repository root
pip install -e . xgboost lightgbm scikit-learn pytest
cd integration
pytest -v
```

## Documentation

- [DESIGN.md](DESIGN.md) - Detailed test design and criteria
