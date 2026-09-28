---
paths:
  - "**/*.py"
  - "**/*.ipynb"
---

# Python Rules

**General Python rules** for editing `.py` / `.ipynb` files in this repository. Auto-loaded across every region that contains Python code (`backend/`, `pipeline/`, tests, notebooks).

For experiment-case directory structure see `.claude/rules/backend/pipeline.md`. For pytest conventions see `.claude/rules/backend/tests.md`.

`pyproject.toml` / `uv.lock` / `.python-version` sit at `backend/` root, and `uv run ...` commands are expected to execute from `backend/`. From the repo root, prefer the `dev/*` wrappers.

## Backend Module Architecture (`backend/src/**`)

`backend/src/` holds the shared libraries. Each subdirectory is exposed as a top-level package via `[tool.hatch.build.targets.wheel] packages`, so imports are bare (`from simulate.jaxenv import ...`, `from gpu.common import ...`):

```
backend/src/
  simulate/        Environment tooling
    jaxenv/        Batched JAX environments (state / env / benchmark) + a template README
  gpu/             GPU provider CLIs (one subpackage per provider):
    common/        Provider-agnostic credentials + run metadata
    runpod/        RunPod pod control CLI (python -m gpu.runpod)
    vast/          Vast.ai pod control CLI (python -m gpu.vast)
    kaggle/        Kaggle Notebook GPU runner (python -m gpu.kaggle)
```

`backend/pipeline/` holds the **experiments**:

```
backend/pipeline/
  <family>/case<N>/   train.py / evaluate.py / env.py / policy.py / config.yaml
```

## Style

- **Type hints are required** on all function signatures and module-level constants. `mypy` runs in CI with the config in `backend/pyproject.toml`.
- Use `from __future__ import annotations` at the top of modules that need forward references.
- Format and lint with Ruff — never hand-format. `dev/format` writes, `dev/lint` checks + autofixes.
- Prefer `pathlib.Path` over `os.path`.
- Prefer f-strings; no `%`-formatting or `.format()`.
- Dataclasses (or Pydantic models where validation is genuinely needed) over ad-hoc dicts for structured state. Configs are loaded from YAML into a dataclass once, at startup.

## Logging

- Use the `logging` module, never bare `print`, in `backend/src/**` and in `train.py`.
- Scalars that a human will plot go to `metrics.json` in the run directory, not to the log.

## Reproducibility (`backend/pipeline/**`)

- Every source of randomness is seeded from `config.seed`: `numpy`, `torch`, `jax.random.PRNGKey`, and the environment's `reset(seed=...)`.
- JAX code threads `PRNGKey` explicitly; never rely on global RNG state inside a `jit`ted function.
- The resolved config is written to the run directory before the first update. If it is not on disk, the run did not happen.
- Checkpoints include optimizer state and the RNG key, so resume is bit-for-bit continuous.

## JAX-specific

- Keep array shapes static inside `jit`: pad to a fixed maximum and mask, rather than branching on data-dependent lengths.
- Batched environments are `(B, ...)` everywhere; a single env is `B = 1`, not a separate code path.
- Scatter (`.at[].set`) is cheap on CUDA and pathological on Metal — see `docs/develop/reference/simulator/jax-gpu.md`. Benchmark on the target backend before optimising.
- Every port from NumPy / PyTorch is anchored by a parity test (`python-to-jax` skill).

## Numeric work

- Prefer `numpy` / `jax.numpy` vectorisation over per-element Python loops on anything called inside the training loop.
- Use `polars` for offline metrics aggregation across runs, `pandas` only when a dependency forces it.
