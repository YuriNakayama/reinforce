# jaxenv — batched JAX environment template

This directory was copied from the kaggriculture project's JAX port of its
farming-sim environment. **The game rules are irrelevant here**; what is
worth reusing is the structure:

| File | Pattern it demonstrates |
|---|---|
| `state.py` | `EnvState` as a pytree of `(B, …)` arrays; constants as module-level ints; `initial_state(B)` |
| `env.py` | One `step(state, action) -> state` with every rule as a masked `jnp.where` / `.at[].set`, no Python branching on data |
| `town.py` | Sub-system with per-env RNG threaded explicitly via `jax.random.split` |
| `benchmark.py` | Warm-up compile excluded, throughput = `B * steps / wall`, sweep over batch sizes, JSON output |

Measured results and the Metal-vs-CUDA caveat are in
`docs/develop/reference/simulator/jax-gpu.md`. Short version: on CUDA scatter
costs the same as an element-wise op and batch 65536 gave ~3.6e7 env-steps/s;
on Apple Metal scatter is ~9× slower than element-wise and the batched env is
slower than plain Python.

## Building a new environment

1. Copy `state.py` / `env.py` into `pipeline/<family>/case<N>/env.py` (or a new
   `simulate/jaxenv/<name>/` package if several cases will share it).
2. Keep shapes static: pad to a maximum and mask.
3. Write the parity test first (`python-to-jax` skill): a NumPy reference
   `step` and `jnp.allclose` against the JAX one for random states/actions.
4. Run the benchmark below on the target backend before committing to the port.

`benchmark.py` times only the JAX port (the original also timed the
kaggle-environments engine and an interpreter-direct loop, which do not exist
here). Docstrings in `env.py` / `state.py` that mention "the official engine"
refer to that original kaggle-environments engine.

```bash
uv run --directory backend python -m simulate.jaxenv.benchmark --steps 720 --batches 1 64 1024
```
