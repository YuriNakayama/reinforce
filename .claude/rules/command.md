---
paths:
  - "dev/**"
---

# Command Catalog (`dev/**`)

`dev/*` are the canonical entry points. Each one `cd`s to the repo root and runs `uv` under `backend/` internally, so **they work from any directory**. Prefer them over invoking `uv` / `dvc` directly — the wrappers pin the working directory and the interpreter.

## Development

| Command | What it does |
|---|---|
| `dev/setup` | `uv sync` under `backend/`. Run once after clone and after dependency changes. `--nn` adds torch, `--gpu` adds the RunPod / Vast / Kaggle SDKs, `--cuda` adds the JAX CUDA plugin (Linux only), `--all` adds everything |
| `dev/format` | `ruff format` — writes |
| `dev/lint` | `ruff check --fix` + `mypy`. Non-zero exit on any failure |
| `dev/test [args...]` | `pytest` under `backend/`. Extra args pass through (`dev/test tests/unit -x`) |

## Experiments

| Command | What it does |
|---|---|
| `dev/train --case <family>/<caseN> [--config PATH] [--smoke]` | Run a case's `train.py` locally. `--smoke` = 1 update on CPU, verifies the run-directory contract |
| `dev/evaluate --case <family>/<caseN> --run <run_id>` | Fixed-seed evaluation of a run's `best` checkpoint |

## Worktrees

| Command | What it does |
|---|---|
| `dev/create-worktree <branch>` | Create `.claude/worktrees/<branch>/`, symlink `data/` to the main repo, share the DVC cache, copy `.env` files. Inside cmux it also opens a pane running Claude on the new tree |
| `dev/delete-worktree [<branch>]` | Remove a worktree (no arg = the current one). Warns on uncommitted/unpushed work, offers to delete the branch |
| `dev/sync-data` | Merge a worktree's real `data/` back into the main repo and replace it with a symlink |

**Always use `dev/create-worktree`, not `git worktree add`.** A bare `git worktree add` gives the new tree its own `data/` and its own empty DVC cache, so `dvc pull` re-downloads every checkpoint from S3.

## Data (DVC)

| Command | What it does |
|---|---|
| `dev/dvc setup` | One-time per machine: shared cache dir + AWS profile into `.dvc/config.local` (gitignored) |
| `dev/dvc pull <target>.dvc` | Fetch real data from S3. **Always name the target** — a bare `pull` finds no tracked data and deletes the directory (see `rules/data.md`) |
| `dev/dvc push [targets]` | Upload to S3 |
| `dev/dvc status --cloud` | Is my local cache in sync with the remote? |
| `dev/dvc add <path>` | Start tracking a directory |
| `dev/dvc <anything>` | Pass-through to `dvc` |

**Do not run two `dvc push` concurrently** — they contend on the cache lock. Reads (`pull`, `status`) are fine in parallel.

## GPU

| Command | What it does |
|---|---|
| `dev/runpod ...` | RunPod pod lifecycle (train / dev / ps / stock / status / logs / pull / promote / cost-report / destroy). Primary backend |
| `dev/runpod train <sha> --case <key> --resume-from <run_id>` | Pull that run's `.dvc`, copy `checkpoint_latest.*` into the new run dir, and let `train.py` resume |
| `dev/vast ...` | Vast.ai pod lifecycle (train / pull / promote / volume / cost-report). Cheaper spot capacity |
| `dev/kaggle ...` | Kaggle Notebook GPU (T4×2 / P100, ~30 h/week free). `dataset push` snapshots `backend/` first; `--data-file data/...` ships extra files and restores them at the same relative path in the kernel |

Needs the optional GPU deps: `dev/setup --gpu`. Trainable cases are registered in `backend/src/gpu/runpod/config/cases.py`; add an entry there before launching a new one. `REINFORCE_DVC_BUCKET` (in `backend/.env` or the shell) is required for any RunPod / Vast launch; `GIT_PAT` is required when the repo is private or when the pod must push run `.dvc` stubs.

**Rent a GPU only when a CPU run has been measured and is too slow**, or when the case is a batched JAX env whose whole point is `vmap` throughput. `case1` of every family runs on a laptop.

## Infrastructure

Terraform lives under `infra/`. `terraform apply` and `destroy` are a human decision — Claude runs `fmt`, `validate`, and `plan` only. See `.claude/rules/infra.md`.
