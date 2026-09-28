# reinforce: Reinforcement Learning Research

Research repository for **reinforcement learning (RL)**. The goal is to survey algorithms (policy-gradient, value-based, actor-critic, model-based), reproduce them on small environments, and measure what actually moves the learning curve — sample efficiency, wall-clock, stability across seeds. There is no competition deadline and no leaderboard; the deliverable is a **reproducible experiment record** under `docs/experiment/` backed by DVC-tracked artifacts.

Environments come from two sources:

- **Gymnasium** standard tasks (CartPole, LunarLander, MuJoCo, Atari) for algorithm reproduction against published numbers.
- **Custom JAX environments** under `backend/src/simulate/jaxenv/` for throughput work — `jit` + `vmap` batching so one GPU runs tens of thousands of environments in lockstep. See `docs/develop/reference/simulator/` for the measured speed-ups (and the Metal caveat).

## Training Pipeline

Every experiment is one **case** directory that owns its config, environment factory, policy, and training loop:

1. **Environment**: `make_env()` returns a Gymnasium env or a batched JAX env with a fixed observation / action spec.
2. **Policy / value network**: PyTorch (`nn` group) or JAX + Equinox + Optax (`jax` is in the base deps).
3. **Rollout**: collect trajectories (vectorised on CPU or `vmap`ped on GPU).
4. **Update**: the algorithm step (PPO clip, TD target, SAC twin-Q, …). Log per-update scalars to `metrics.json`.
5. **Evaluation**: a deterministic evaluation against a **fixed seed set** every N updates — the training reward curve alone is not evidence of progress.

> Runs can be preempted at any moment (RunPod Community pods vanish). Every case must checkpoint periodically and resume from the latest checkpoint, and `best` weights must be streamed to S3 as soon as they improve. See `.claude/rules/backend/pipeline.md`.

## Technology Stack

- **Language**: Python 3.13
- **RL / numerics**: JAX + Equinox + Optax (default), PyTorch (`dev/setup --nn`), numpy / polars
- **Environments**: Gymnasium, custom JAX envs
- **Data**: DVC + S3 for checkpoints, metrics, and replays
- **Testing**: Pytest, Ruff, Mypy
- **Package management**: uv
- **GPU**: RunPod (primary), Vast.ai, Kaggle Notebook — all driven from `dev/*`
- **Infra**: `infra/runtime/` (Vast.ai GPU image). Terraform for the S3 DVC remote / OIDC / ECR is planned but not yet ported (see `.claude/rules/infra.md`)

## Folder Structure

```
backend/                Python implementation (pyproject.toml / uv.lock live here)
  src/                  Shared dev libs — importable as bare top-level packages
    simulate/           Environment tooling: jaxenv/ (batched JAX envs + benchmark)
    gpu/common/         Provider-agnostic: credentials, run metadata
    gpu/runpod/         RunPod pod control (python -m gpu.runpod)
    gpu/vast/           Vast.ai pod control (python -m gpu.vast)
    gpu/kaggle/         Kaggle Notebook GPU runner (python -m gpu.kaggle)
  pipeline/             Experiments: <family>/case<N>/ (family = algorithm family)
    ppo/case1/          e.g. PPO on CartPole — the smoke-test baseline
  tests/                Pytest (unit / integration / e2e)
infra/                  runtime/ (GPU base image); Terraform planned (S3 DVC remote, OIDC)
data/                   4 layers (lake / processed / mart / output) — DVC-managed, gitignored
dev/                    Command wrappers (each cd's to root and runs uv under backend/)
.github/workflows/      ci.yml (ruff / mypy / pytest)
docs/
  develop/              Explanations of existing code + reference/ (measurements carried over)
  experiment/           Experiment plans and results — see .claude/rules/docs.md
  plans/                Feature plans (one directory per feature)
```

`uv run ...` is expected to run under `backend/`. From the repo root, use `dev/*`. See [`.claude/rules/command.md`](rules/command.md) for the full command catalog.

## Glossary

| Term | Description |
|------|-------------|
| Family | Algorithm family directory under `backend/pipeline/` (`ppo`, `dqn`, `sac`, …) |
| Case | One concrete experiment variant within a family. Append-only: `case1`, `case2`, … |
| Fixed-seed eval | Deterministic evaluation on a frozen seed set, run every N updates; the primary progress metric |
| Run | One training launch, identified by `run_id` (`YYYYMMDD-HHMMSS`), stored under `data/output/models/<family>/<case>/runs/<run_id>/` |
| `hypotheses.md` | Per-case hypothesis list driving the `experiment` skill loop (see `.claude/rules/docs.md`) |
| jaxenv | Batched JAX environment: `(B, …)` state arrays, one `jit`ted step for all `B` envs |

## What counts as a result

An experiment is reportable when it has **≥ 3 seeds**, a fixed-seed evaluation curve, and the wall-clock / hardware it ran on. A single-seed reward curve is a smoke test, not a finding. Adopt / reject decisions are recorded in the case's `hypotheses.md` and the `docs/experiment/` result file.

## Rules

| Rule file | Auto-loaded for | When to read manually |
|-----------|----------------|----------------------|
| `.claude/rules/python.md` | `**/*.py`, `**/*.ipynb` | Module architecture, checkpoint / resume contract |
| `.claude/rules/backend/pipeline.md` | `backend/pipeline/**` | Family / case layout, run directory contract, evaluation discipline |
| `.claude/rules/backend/tests.md` | `backend/tests/**` | unit / integration / e2e classification |
| `.claude/rules/data.md` | `data/**` | 4-layer structure, DVC conventions |
| `.claude/rules/infra.md` | `infra/**` | Terraform, OIDC, required env vars |
| `.claude/rules/command.md` | `dev/**` | Command catalog |
| `.claude/rules/docs.md` | `docs/**` | Experiment doc naming |
| `.claude/rules/security.md` | Always loaded | Commits, secrets, CI/CD |

## Secrets

`backend/.env` holds provider API keys (`RUNPOD_KEY`, `VAST_API_KEY`, `KAGGLE_*`) and is gitignored. The DVC bucket name is passed via `REINFORCE_DVC_BUCKET`, never hardcoded. Before any commit, verify no `terraform.tfstate`, `terraform.tfvars`, `.env`, or `data/` contents are staged.

## Response Language And Interface

- Answer user questions concisely, organizing the response as a table, chart, list, short sentence, ASCII art, or similar structured format.
- Keep user-facing replies under 800 characters, excluding tables, charts, code blocks, and ASCII art (which can exceed the limit when needed).
- Use the `AskUserQuestion` tool when asking questions to the user
- Internal reasoning, tool calls, and intermediate notes: English.
- User-facing output (final replies, reports, summaries): Japanese.(全てのユーザー向けの出力は日本語で行うこと)
