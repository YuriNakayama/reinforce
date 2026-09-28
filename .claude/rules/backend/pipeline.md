---
paths:
  - "backend/pipeline/**"
---

# Experiment Pipeline Rules (`backend/pipeline/**`)

`backend/pipeline/` holds the **experiments**. Everything under `backend/src/` is shared tooling; a case may import from it freely (there is no submission tarball to keep small).

## Family / case layout

```
backend/pipeline/
  <family>/            algorithm family: ppo | dqn | sac | ...
    case<N>/           one experiment variant
      README.md        what this case tests and what changed vs the previous case
      config.yaml      every hyper-parameter, seed list, env id — nothing hardcoded in .py
      env.py           make_env() -> Gymnasium env or batched JAX env
      policy.py        network definitions
      train.py         python -m pipeline.<family>.case<N>.train [--config PATH]
      evaluate.py      fixed-seed evaluation of a checkpoint
```

- **family** = the algorithm family. Add a new family directory when the update rule changes (PPO vs DQN), not when a hyper-parameter changes.
- **case** = one concrete, reproducible variant. Cases are append-only: `case1`, `case2`, … A case is the historical record of an experiment, so **do not rewrite an old case in place** once results are recorded in `docs/experiment/` — create the next case instead.
- `case1` of every family is the **smoke baseline**: smallest env, runs end-to-end on a laptop CPU in under a minute. It exists so that CI and the RunPod onstart have a target that proves the plumbing before any GPU is rented.

## Run directory contract

`train.py` reads `REINFORCE_RUN_DIR` (absolute path, injected by the GPU onstart) and falls back to `data/output/models/<family>/<case>/runs/<run_id>/` locally. It must write:

| File | When | Purpose |
|---|---|---|
| `config.resolved.yaml` | at start | The exact config used, with defaults filled in |
| `metrics.json` | every update | Append-only list of scalars: step, wall_clock, train_return, eval_return, loss terms |
| `checkpoint_latest.*` | every N updates | Resume point. **Training must resume from it if it exists** |
| `best.*` | when eval improves | Best weights so far; streamed to `REINFORCE_BEST_S3_PREFIX` when set |
| `eval/seed<K>.json` | every eval | Per-seed returns so variance is inspectable |

The onstart scripts and `dev/runpod status` parse `metrics.json`; keep its schema flat.

## Long-running training

Assume the run **can fail or be preempted at any moment**:

- Checkpoint on a wall-clock interval (default 10 min), not only on epoch boundaries.
- Emit a progress marker (`progress.mark(step)`) at least every 5 minutes — the RunPod watcher kills a pod that is silent for 15 minutes.
- Never depend on state that lives only in the container; the run directory is on the persistent volume.

## Evaluation discipline

- Report the **fixed-seed evaluation**, never the training-batch return, as the headline number.
- **≥ 3 seeds** before drawing a conclusion. Record mean and per-seed values.
- Compare against a fixed baseline from the same family (`case1` unless the hypothesis says otherwise) on the same env id and the same evaluation seeds.
- Wall-clock and hardware go in the result file. A 2× sample-efficiency gain that costs 10× compute is a different finding.

## Local verification before renting a GPU

```bash
dev/train --case ppo/case1 --smoke        # 1 update, CPU, seconds
dev/test tests/unit -x
```

A case is not launchable on a pod until the smoke run completes and writes every file in the run directory contract.
