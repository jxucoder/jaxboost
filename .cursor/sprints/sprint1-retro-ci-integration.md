# Sprint 1 Retrospective: CI Integration

**Date:** 2026-01-04  
**Duration:** ~30 minutes  
**Status:** Complete

## Summary

Established continuous integration infrastructure with GitHub Actions for automated testing and linting.

## Deliverables

| Deliverable | Status | Notes |
|-------------|--------|-------|
| `tests.yml` workflow | ✅ Done | Python 3.10-3.12 matrix, coverage on 3.11 |
| `lint.yml` workflow | ✅ Done | ruff check + format |
| README badges | ✅ Done | Tests, Lint, Python version, License |
| Lint fixes | ✅ Done | 18 issues fixed across src/ and tests/ |
| License change | ✅ Done | MIT → Apache 2.0 |

## What Went Well

1. **Fast execution**: Sprint completed in ~30 minutes
2. **Clean separation**: Tests and lint in separate workflows for clearer feedback
3. **Coverage integration**: Optional codecov upload on Python 3.11
4. **JAX Metal handled**: `JAX_PLATFORMS=cpu` already in conftest.py from Sprint 0

## What Could Be Better

1. **jax-metal dependency**: Had to work around macOS-only dependency in CI
   - Solution: Manual pip install of JAX deps without jax-metal
   - Future: Consider making jax-metal optional in pyproject.toml

2. **Experimental module ignores**: Had to add ruff per-file-ignores for experimental module
   - Acceptable tradeoff for research-stage code

3. **CI workflow command execution**: Initial workflows used `uv run` which failed
   - Issue: `uv run` requires package to be fully installed in environment
   - Solution: Use direct `pytest` and `ruff` commands after explicit installation
   - Fixed in commit `a1f5d79`

## Technical Decisions

### CI Dependency Installation

Instead of `uv sync --dev` (which would fail due to jax-metal on Linux), we use:

```yaml
- run: |
    # Install dev dependencies (includes pytest)
    uv pip install pytest ruff
    # Install jaxboost without jax-metal (macOS only)
    uv pip install -e . --no-deps
    uv pip install "jax>=0.4.20" "jaxlib>=0.4.20" "optax>=0.1.7" "chex>=0.1.8"
```

This skips jax-metal while installing all other dependencies.

### CI Command Execution

**Initial approach (failed):**
- Used `uv run pytest` and `uv run ruff`
- Failed because `uv run` requires package to be fully installed in environment

**Final approach (working):**
- Install `pytest` and `ruff` explicitly via `uv pip install`
- Run commands directly: `pytest tests/` and `ruff check src/ tests/`
- Commands are available in PATH after installation

### Ruff Configuration

Added per-file ignores in `pyproject.toml`:

```toml
[tool.ruff.lint.per-file-ignores]
"src/jaxboost/experimental/__init__.py" = ["E402", "F401", "SIM105"]
```

Rationale: Experimental module uses try/except patterns for optional imports.

## Lint Fixes Applied

| Category | Count | Files Affected |
|----------|-------|----------------|
| Trailing whitespace (W291/W293) | 10 | multi_task.py, regression.py, conftest.py |
| zip() strict parameter (B905) | 2 | multi_output.py |
| Ternary simplification (SIM108) | 2 | multiclass.py |
| Import sorting (I001) | 4 | __init__.py files |

## Metrics

- **Workflows created**: 2 (tests.yml, lint.yml)
- **Workflow fixes**: 1 (command execution fix)
- **Lint issues fixed**: 18
- **All 149 tests**: Still passing
- **Python versions**: 3.10, 3.11, 3.12
- **CI status**: ✅ All checks passing after fixes

## Next Steps (Sprint 2 Candidates)

1. **jaxtyping annotations**: Add shape safety across objective module
2. **Make jax-metal optional**: Move to `[extras]` in pyproject.toml
3. **mypy integration**: Type checking workflow
4. **Integration tests**: Full XGBoost/LightGBM training loops
5. **Benchmark CI**: Optional performance regression tests

## Files Changed

```
.github/workflows/tests.yml     (new, fixed in a1f5d79)
.github/workflows/lint.yml      (new, fixed in a1f5d79)
README.md                       (badges, license)
LICENSE                         (Apache 2.0)
pyproject.toml                  (license, ruff ignores)
src/jaxboost/__init__.py        (import sorting)
src/jaxboost/objective/*.py     (lint fixes)
tests/conftest.py               (whitespace)
```

## Post-Sprint Fixes

**Commit `a1f5d79`**: Fixed CI workflow command execution
- Changed from `uv run pytest` to direct `pytest` command
- Changed from `uv run ruff` to direct `ruff` command
- Explicitly install pytest and ruff before running
- All CI checks now passing ✅

