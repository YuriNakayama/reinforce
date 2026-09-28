# Reference: simulator measurements (carried over)

These three documents are condensed from the kaggriculture project and kept
here **as measurements and lessons, not as plans**. Implementation plans, CLI
usage, and test sections that cannot be reproduced here were removed. Names
(the farming engine, `kaggriculture.py`, 720-turn seasons) are historical and
intentionally left as they were; each section adds an "RL への読み替え" note
where the lesson transfers.

| File | Why it is here |
|---|---|
| `design.md` | Evaluation lessons: failures show up only as low reward, so crashes / timeouts / ignored actions need their own counters; deterministic seeding for parallel eval |
| `fast-sim.md` | Profile showing the framework (deepcopy / validation), not the game logic, dominated step time — check this before rewriting an environment "for speed" |
| `jax-gpu.md` | Batched JAX env throughput: CUDA 42,900× vs the original engine, Apple Metal *slower* than CPU because of scatter cost. Read before choosing a backend |

## torch CUDA wheels (recipe from ptcg)

`backend/pyproject.toml` keeps torch CPU-only. When a pod needs CUDA torch,
the ptcg project used an explicit index + exclusive extra:

```toml
[project.optional-dependencies]
torch-cu124 = ["torch>=2.3.0,<3"]

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu124", extra = "torch-cu124", marker = "sys_platform == 'linux'" }]

[tool.uv]
conflicts = [[{ extra = "torch-cu124" }, { group = "cuda" }]]  # cuDNN 9.1 vs 9.8
```

The `conflicts` entry matters: the cu124 wheel hard-pins cuDNN 9.1 while
`jax-cuda12-plugin` needs 9.8, so one lock file cannot hold both GPU stacks.
Pick one per pod.
