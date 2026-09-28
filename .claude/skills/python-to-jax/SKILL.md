---
name: python-to-jax
description: >-
  Port Python / NumPy / PyTorch code (RL environments, policies, rollout
  collection, GAE / return computation, PPO / DQN update loops) to JAX,
  producing a `<name>_jax.py` (or a `*_jax/` package for directory-level
  ports) that is jit/vmap-friendly, plus a numerical-parity pytest that proves
  the JAX output matches the original within float32 tolerance. Use this skill
  WHENEVER the user asks to "convert to JAX", "JAX 化", "jax に書き換え",
  "port this env/policy to jax", "make this jittable / vmappable", "GPU 化したい",
  "rewrite the rollout in JAX", or names a file/directory (e.g. `env.py`,
  `pipeline/ppo/case2/`) and asks to translate its Python/NumPy/Torch logic
  into JAX. Trigger even when the user only gives a path or a vague "this" —
  the skill resolves the target and proceeds TDD-style. Do NOT use for writing
  brand-new JAX code with no Python source, for plain JAX bug-fixes, or for
  PyTorch-only work that isn't being ported.
---

# Python → JAX Conversion

Port Python/NumPy/PyTorch code to JAX so it runs jit-compiled and vmap-batched
on GPU, while staying numerically faithful to the original. Every port is
anchored by a **parity test** that compares the JAX output to the Python
original. That test is the contract — write it first, port until it passes.

The in-repo reference implementation is `backend/src/simulate/jaxenv/`
(`state.py` pytree of `(B, …)` arrays, `env.py` masked-update `step`,
`town.py` explicit PRNG threading, `benchmark.py` throughput sweep — see its
`README.md`). Mirror its structure for new environments.

## First, check the port can actually pay off (throughput, not latency)

JAX/GPU wins **throughput** — many independent units (environments, minibatch
samples, candidate actions) computed in parallel under one `vmap`. It does
**not** win **latency**: a single small computation (one env step for one
environment, one short function call) runs *slower* on GPU than in Python,
because the fixed per-call dispatch + host↔device transfer cost exceeds the
work. In the earlier project this setup was carried over from, the same solver
hit 681× on a batched grid but was 5.6× *slower* when called once per step
inside a Python loop.

So before porting *for speed*, ask: **is there a batch axis to amortize over?**
- Many parallel environments, rollout collection over T steps, minibatch
  updates, evaluation over many seeds → **yes, port it**; the vmap/scan axis is
  the win.
- A single-env interactive loop, any one-shot sub-10 ms call with no batch
  dimension → **GPU can't help**. Port only if it feeds a batched consumer, and
  benchmark the *consumer's* unit (see `references/benchmark-and-tune.md` §2).

Also check the backend: scatter (`.at[].set`) is as cheap as an element-wise op
on CUDA but ~9× more expensive on Apple Metal, and the batched env was slower
than plain Python on Metal. See `docs/develop/reference/simulator/jax-gpu.md`.

Porting for *correctness/reuse* (autodiff through the env, sharing a kernel
with a JAX training loop) can be worth it without a speed win — just don't
claim a speedup the usage pattern can't deliver.

## The core loop (TDD)

A JAX port that "looks right" but silently diverges (a wrong axis, an
off-by-one in a mask, terminal handling lost under `scan`) is worse than no
port, because the bug surfaces as a flat learning curve hours into a GPU run.
The parity test catches it in seconds on CPU.

1. **Locate & read the source.** Resolve the target. Read the full
   implementation plus anything it imports that also needs porting. Note the
   public contract: names, argument shapes, return shapes, dtypes.
2. **Find a structural reference.** Look for an existing `*_jax.py` sibling or
   a `test_*_parity.py` under `backend/tests/`, and read
   `backend/src/simulate/jaxenv/` for env-level structure.
3. **Write the parity test FIRST (RED).** A pytest that runs the Python
   original and the (not-yet-existing) JAX version on identical inputs and
   asserts they match within tolerance. See `## Parity test`.
4. **Port the code (GREEN).** Follow `references/conversion-patterns.md`.
   Fixed shapes, `lax` control flow, `eqx.Module` for parameters. Iterate until
   the parity test is green.
5. **Verify the whole suite.** `dev/test` (or the targeted pytest) and
   `dev/lint` (ruff + mypy). A port isn't done until lint and types pass.
6. **Benchmark & tune (on the target GPU backend).** Benchmark the original vs
   the JAX version on a CUDA GPU (RunPod) — local CPU and Metal numbers do not
   predict CUDA behaviour. Tune until it's actually faster, **re-running the
   parity test after every tuning change**. See
   `references/benchmark-and-tune.md`. If no GPU pod is available, write the
   bench script and report the figure as pending — never substitute a CPU
   number or guess.

```bash
uv run --directory backend pytest tests/unit/<path>/test_<name>_jax_parity.py -x -q
```

## Resolving the target

- **Single file** → produce `<name>_jax.py` next to it (`env.py` →
  `env_jax.py`, `policy.py` → `policy_jax.py`). Classes/functions get a `Jax`
  suffix (`ActorCritic` → `ActorCriticJax`, `compute_gae` → `compute_gae_jax`).
- **Directory / "一式"** → port file-by-file in dependency order (constants →
  state → step → reset → rollout → loss → update). For environments shared by
  several cases, place the port under `backend/src/simulate/jaxenv/<name>/`
  following the existing `state.py` / `env.py` split. Green one module's parity
  test before moving up the chain.

### Scope by the hot path, not by file size

When the goal is "make training faster", the leaf-first *order* is right, but
leaf-first must not become leaf-only. Find where wall-clock actually goes —
usually the env `step`, rollout collection, or a per-sample Python loop in the
loss — and port that, even when it's the hard part (variable-length entity
lists, early termination, auto-reset). Replace the Python loop with one
batched JAX call. Host-side bookkeeping that is not compute-bound (logging,
checkpoint I/O) can stay in Python — say so explicitly.

`references/conversion-patterns.md` §12 covers the hard constructs:
fixed-iteration loops with early exit, variable-length collections → padded
arrays + masks, and host-side splits for ragged data.

If the target is ambiguous, ask which file or directory — don't guess.

## The five conversions that matter most

1. **Fixed shapes + masks, never dynamic length.** Pad to a `MAX_*` constant
   and carry a `*_valid` boolean array; compute over the full padded array and
   `jnp.where(valid, value, neutral)`. Batched envs are `(B, …)` everywhere; a
   single env is `B = 1`, not a separate code path.

2. **`lax.scan` / `vmap` / `cond` replace Python loops.** A `for t in range(T)`
   rollout becomes `jax.lax.scan(step_fn, carry, xs)`; B parallel envs become
   `jax.vmap`; an `if` on traced values becomes `jax.lax.cond` or `jnp.where`.
   Mutations become returning a new carry (`.at[idx].set`, `_replace`).

3. **`eqx.Module` for anything with parameters.** PyTorch `nn.Module` →
   Equinox module; gradients via `eqx.filter_value_and_grad`; optimizers via
   Optax (`optax.adam` + `optax.clip_by_global_norm` for `clip_grad_norm_`).

4. **Weights load by name, 1:1.** When porting a trained PyTorch model, load
   the `.pt` with `torch.load(path, weights_only=True)`, copy leaf-by-leaf by
   parameter name, and assert nothing is missing. Mirror layer order and sizes
   exactly.

5. **Explicit PRNG and one trace.** Thread `jax.random.PRNGKey` through every
   stochastic function (`split` per step / per env). Never pass a varying
   Python `int` (e.g. `seed`) into a jitted function — it recompiles per value
   and can blow up the compile cache. Pass arrays, or mark genuinely static
   args `static_argnums`.

## Parity test

Run the Python original and the JAX port on identical inputs and assert
closeness:

- **Pure numeric** (network forward, GAE, loss terms): float32 tolerance
  `1e-4`. Cast float64 NumPy inputs to float32 first.
- **Multi-step env trajectories**: compare discrete state (positions on a
  grid, done flags, integer counters) exactly; allow a wider band on
  accumulated floats, or record disagreements with a clear message.
- **Stochastic code**: make randomness an input (pre-sampled noise / actions)
  so both sides consume the same numbers.

```python
# backend/tests/unit/pipeline/ppo/case2/test_gae_jax_parity.py
import jax.numpy as jnp
import numpy as np

from pipeline.ppo.case2.gae import compute_gae
from pipeline.ppo.case2.gae_jax import compute_gae_jax


def test_gae_parity() -> None:
    rng = np.random.default_rng(0)
    T, B = 64, 8
    rewards = rng.normal(size=(T, B)).astype(np.float32)
    values = rng.normal(size=(T + 1, B)).astype(np.float32)
    dones = (rng.random((T, B)) < 0.05).astype(np.float32)

    ref = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    out = compute_gae_jax(jnp.asarray(rewards), jnp.asarray(values),
                          jnp.asarray(dones), 0.99, 0.95)

    np.testing.assert_allclose(np.asarray(out), ref, atol=1e-4)
```

For a ported network, also assert the **parameter count matches** — a count
mismatch means the architecture diverged.

## Conventions

- JAX file beside its source; parity test under `backend/tests/unit/`
  mirroring the source path (`pipeline/ppo/case2/env_jax.py` →
  `tests/unit/pipeline/ppo/case2/test_env_jax_parity.py`).
- Top-of-file docstring states **which Python file it mirrors** and any
  intentional deviations.
- Python 3.13 types, annotate arrays as `jax.Array`, avoid `Any` / `cast` /
  `type: ignore`. State containers are `NamedTuple`, frozen `@dataclass`, or
  `eqx.Module`.
- Never mutate; return new instances.

## When you're done

Report: files created, parity test path and tolerance, green test output, the
**speed comparison** (baseline vs JAX, device, batch size / horizon, scaling
curve), tuning changes applied and that parity still held, and intentional
deviations. If the GPU number is pending a RunPod run, say so.
