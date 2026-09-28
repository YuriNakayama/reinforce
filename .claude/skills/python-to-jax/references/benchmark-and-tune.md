# Benchmarking & Tuning a JAX Port

The point of porting to JAX is **throughput on GPU**. A correct-but-slow port is
a failure. This file is the timing idiom and the tuning playbook — read it when
you reach the benchmark/tune step.

**Benchmark and tune on GPU only — never on the local CPU.** The dev laptop has
no CUDA GPU, and CPU wall-clock does not predict GPU behavior (the vmap-scaling
and memory characteristics that matter only appear on-device). Measurement and
tuning happen on a RunPod GPU pod via `dev/runpod` (see §4 for where the bench
script lives). Apple Metal numbers are not a proxy either — scatter-heavy envs
run slower on Metal than on CPU (`docs/develop/reference/simulator/jax-gpu.md`).
If no GPU pod is available right now, **do not run a CPU
benchmark as a stand-in and do not fabricate a number** — write the GPU bench
script, state that the figure is pending a RunPod run, and stop there.

Golden rule: **re-run the parity test after every tuning change.** Speed that
breaks correctness is not speed. A reordered reduction, an `f64→f32` switch, or
a `donate_argnums` mistake can silently shift outputs past the parity band.

## Table of contents

1. [Measuring honestly (on GPU)](#1-measuring-honestly-on-gpu)
2. [Apples-to-apples comparison](#2-apples-to-apples-comparison)
3. [The tuning playbook](#3-the-tuning-playbook)
4. [Where to put the benchmark](#4-where-to-put-the-benchmark)
5. [Reporting](#5-reporting)

---

## 1. Measuring honestly (on GPU)

Two mistakes produce fake numbers. Avoid both.

**Warm up — don't count compile time.** The first call to a jitted function
triggers XLA compilation (often seconds). Call once untimed to compile, then
time later calls.

**Block — don't time async dispatch.** JAX runs asynchronously: a jitted call
returns immediately and the compute happens later. If you stop the clock before
forcing completion you measure dispatch, not the kernel. Force completion with
`.block_until_ready()` (one output) or `jax.block_until_ready(pytree)` inside
the timed region. Use `time.perf_counter()` (as
`backend/src/simulate/jaxenv/benchmark.py` does).

```python
import time, jax
runner = jax.jit(run)                      # run: state -> (final_state, outs)
final, _ = runner(state0); final.step.block_until_ready()   # warm-up, NOT timed
t0 = time.perf_counter()
for s in states:
    final, _ = runner(s); final.step.block_until_ready()    # block each iter
per_unit = (time.perf_counter() - t0) / len(states)
```

## 2. Apples-to-apples comparison

Run the Python/Torch original and the JAX version on the **identical workload**
(same horizon, number of envs, seed) on the GPU pod. Report each side's
throughput and the ratio. For environments and rollouts the number that
justifies the port is **env-steps/s = B × T / wall-clock** at realistic batch
sizes, vs the NumPy / Gymnasium-vector baseline. Keep the baseline cheap if it
is slow — you need its per-unit cost, not a large sample.

### Batched vmap vs per-call dispatch — the result that decides everything

Measured in the earlier project this setup was carried over from (L4 GPU): a
solver `vmap`ped over a whole candidate grid beat the Python double loop by
**66× / 273× / 681×** at grids 8², 16², 24². The *same* kernel called once per
candidate from a Python loop ran **5.6× slower** than pure Python, and was worse
on GPU than on CPU. Same kernel, opposite verdict. The batched JAX env in
`docs/develop/reference/simulator/jax-gpu.md` shows the same shape: batch 1 on
CUDA is unimpressive, batch 65536 is ~4×10⁴× the reference engine.

Why: a GPU call has high fixed dispatch + host↔device transfer latency. Amortized
over a large vmap it's nothing; paid once per small call it dominates. So:

- **The speedup is in the batching, not the kernel.** Porting a function to JAX
  buys nothing — sometimes loses — unless the *caller* hands it the whole loop
  domain at once. Bench the batched grid AND the real per-call usage; report
  both. If your only number is the grid, you haven't shown the entry point got
  faster.
- A sequential consumer (a single-env Python training loop calling a jitted
  `step` once per transition) **cannot** be sped up by swapping in a JAX
  function call-for-call. It needs restructuring to "B envs × T steps in one
  `scan`" — say so honestly rather than reporting the batched number as if the
  training loop got that much faster.

## 3. The tuning playbook

Order: **profile → JIT & loops → memory & precision → parallelize → data
pipeline.** Biggest lever first; re-run parity after each change.

1. **Profile — measure, don't guess.** Find the bottleneck with
   `jax.profiler.trace` before changing anything. Beware async timing (block
   explicitly). Inspect the HLO (`jax.jit(f).lower(*args).compile().as_text()`)
   for stray `copy`/`transpose` or inter-device communication that snuck in.

2. **JIT optimization.** Wrap the *whole* hot loop in one `jit` to cut
   compile-boundary overhead. Fix shapes/config with `static_argnums` so varying
   input shape/dtype doesn't trigger recompiles (the per-seed recompile trap in
   `conversion-patterns.md` pattern 8 — it blew the earlier project's
   cache past 3 GB → SIGABRT). For large arrays overwritten each step, reuse buffers with
   `donate_argnums`.

3. **Loop structure.** Replace Python `for` with `lax.scan` (compile time drops
   by orders of magnitude — the body compiles once). Batch with `vmap`; nested
   `vmap` is fine. Consider `scan`'s `unroll` argument to trade compile time for
   per-step speed. **`vmap` batch size is the #1 GPU lever** — sweep it (e.g.
   {1, 64, 1024, 8192, 65536}) and report the scaling curve, not a single
   point; throughput saturates and then drops past a device-specific size.

4. **Memory & precision.** Introduce mixed precision (`bfloat16`/`float16` for
   matmuls) while keeping accumulation in `float32`. For big models or long
   `scan`s, trade memory for recompute with `jax.checkpoint` (remat); use
   `remat_policy` to control which intermediates are saved. JAX defaults to f32 —
   only force f64 if parity genuinely requires it (f64 on GPU is far slower).

5. **Parallelize (multi-device).** When several devices are available, shard with
   `jax.sharding.NamedSharding` + `PartitionSpec`. Pin intermediate shardings
   with `with_sharding_constraint` and minimize collective communication
   (`psum`, etc.) frequency.

6. **Data pipeline.** Once compute is fast, data feeding becomes the bottleneck.
   Exploit `device_put`'s asynchrony to prefetch and cut host→device transfers.

Profile first, change one thing at a time, attribute each speedup, re-check
parity.

## 4. Where to put the benchmark

- **Environment throughput** → extend or mirror
  `backend/src/simulate/jaxenv/benchmark.py` (warm-up excluded, `B × steps /
  wall`, batch sweep, JSON output under `data/output/benchmark/`). Run it with
  `uv run --directory backend python -m simulate.jaxenv.benchmark`.
- **Case-specific benchmarks** (rollout, update step) → a `bench.py` inside the
  case directory (`backend/pipeline/<family>/case<N>/bench.py`) that writes its
  JSON into the run directory, so the RunPod onstart uploads it with the other
  run artifacts.

Either way, **execute it on the GPU pod** (`dev/runpod dev` for an interactive
pod, destroyed afterwards), not locally.

## 5. Reporting

Give the user an actionable table — all rows measured on the GPU pod:

```
T=500, seed=0, device=RunPod RTX 4090   (illustrative layout, not real numbers)
config                      env-steps/s     speedup vs numpy baseline
numpy (baseline, B=1)          1.2e4          1.0x
jax   (gpu, B=64)              2.0e5         17x
jax   (gpu, B=8192)            2.5e7       2100x
jax   (gpu, B=65536)           3.5e7       2900x
```

State the device, horizon/batch settings, the vmap-scaling curve, which tuning
changes you applied, and confirm parity still passed after tuning. If the GPU
pod isn't available, report "GPU bench pending RunPod" with the script ready —
never a CPU substitute or a guessed figure.
