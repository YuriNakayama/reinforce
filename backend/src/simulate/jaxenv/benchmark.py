"""Measure whether batching the JAX environment on a GPU actually pays off.

Reports environment-steps per second for the batched JAX port at several batch
sizes on the current default backend (``jax.devices()``). The ``vs B=1`` column
is the speed-up over the smallest batch in the sweep.

The original kaggriculture version also timed the official
kaggle-environments engine and an interpreter-direct loop; those engines do
not exist in this repository and were removed. The measured comparison is
recorded in ``docs/develop/reference/simulator/jax-gpu.md``.

A batched step does ``B`` times the work of a single step, so throughput is
``B * steps / wall_clock``. Compile time is excluded by running one warm-up
step first and reported separately, because it is paid once per shape.

Read the numbers with the scope limits in ``env.py`` in mind: the JAX port
implements the crop/market core, not hired hands, animals, or land purchase.
It is a lower bound on how much work a full port would have to do per step, so
treat the throughput as optimistic for a complete environment.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp

from . import env as E
from .state import EnvState, initial_state


@dataclass
class BenchResult:
    """One measured configuration."""

    engine: str
    device: str
    batch_size: int
    steps: int
    wall_sec: float
    compile_sec: float

    @property
    def env_steps(self) -> int:
        return self.batch_size * self.steps

    @property
    def steps_per_sec(self) -> float:
        return self.env_steps / self.wall_sec if self.wall_sec > 0 else 0.0


def bench_jax(batch_size: int, steps: int) -> BenchResult:
    """Time the batched JAX environment.

    The whole episode is rolled into one ``lax.fori_loop`` so the measurement
    is not dominated by per-step Python dispatch — that is the point of a
    device-side environment.
    """
    device = jax.devices()[0].platform

    def rollout(state: EnvState, n_steps: jnp.ndarray) -> EnvState:
        def body(i: int, s: EnvState) -> EnvState:
            # Actions depend only on turn-of-day, computed on device.
            hour = jnp.remainder(i, 24)
            op = jnp.where(
                hour == 1,
                E.OP_PLANT,
                jnp.where(
                    (hour == 2) | (hour == 3),
                    E.OP_WATER,
                    jnp.where(hour == 5, E.OP_HARVEST, E.OP_PASS),
                ),
            )
            mop = jnp.where(
                hour == 0,
                E.MARKET_BUY_SEED,
                jnp.where(hour == 6, E.MARKET_SELL, E.MARKET_NONE),
            )
            qty = jnp.where(hour == 0, 1, jnp.where(hour == 6, 5, 0))
            shape = (batch_size, 2)
            stepped: EnvState = E.step(
                s,
                jnp.full(shape, op, dtype=jnp.int32),
                jnp.zeros(shape, dtype=jnp.int32),
                jnp.full(shape, mop, dtype=jnp.int32),
                jnp.zeros(shape, dtype=jnp.int32),
                jnp.full(shape, qty, dtype=jnp.int32),
            )
            return stepped

        out: EnvState = jax.lax.fori_loop(0, n_steps, body, state)
        return out

    # n_steps is traced, so one compilation serves any episode length. Baking it
    # in as a Python int instead makes XLA specialise per length, which on the
    # Metal backend turned a 0.3s compile into minutes.
    compiled = jax.jit(rollout)

    state = initial_state(batch_size)
    n = jnp.int32(steps)
    t0 = time.perf_counter()
    jax.block_until_ready(compiled(state, n))  # type: ignore[no-untyped-call]
    compile_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    jax.block_until_ready(compiled(state, n))  # type: ignore[no-untyped-call]
    wall = time.perf_counter() - t0

    return BenchResult(
        engine="jax",
        device=device,
        batch_size=batch_size,
        steps=steps,
        wall_sec=wall,
        compile_sec=compile_sec,
    )


def render(results: list[BenchResult]) -> str:
    baseline = results[0].steps_per_sec if results else 0.0
    lines = [
        f"{'engine':<10}{'device':<8}{'batch':>7}{'env-steps':>12}"
        f"{'wall (s)':>11}{'steps/sec':>13}{'vs B=1':>13}{'compile':>10}",
    ]
    for r in results:
        ratio = r.steps_per_sec / baseline if baseline else 0.0
        lines.append(
            f"{r.engine:<10}{r.device:<8}{r.batch_size:>7}{r.env_steps:>12,}"
            f"{r.wall_sec:>11.3f}{r.steps_per_sec:>13,.0f}"
            f"{ratio:>12.1f}x{r.compile_sec:>9.1f}s"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 8, 64, 256, 1024],
    )
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    results: list[BenchResult] = []
    for batch in sorted(args.batches):
        results.append(bench_jax(batch, args.steps))

    print(f"device: {jax.devices()}")
    print()
    print(render(results))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)


if __name__ == "__main__":
    main()
