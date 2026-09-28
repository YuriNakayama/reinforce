---
name: experiment-execution
description: >
  Execution runner for reinforce RL experiments under
  `backend/pipeline/<family>/case<N>/`. Takes a single hypothesis (one line in
  `hypotheses.md` or one `iterN_plan.md`) and drives the full cycle inline in
  the main session: case implementation (config.yaml / train.py / policy.py /
  env.py / evaluate.py) → local smoke (`dev/train --smoke`, verifies the
  run-directory contract; mandatory) → `dev/test` → push & RunPod GPU launch →
  in-flight monitoring (progress / steps / loss / eval return / GPU·CPU·memory)
  → on failure, stop & terminate the pod and relaunch (auto-recover) →
  fixed-seed evaluation (`dev/evaluate`, ≥ 3 seeds, same eval seeds as the
  baseline) → `iterN_result.md`. Honors the `hypotheses.md` skip list (e.g.
  `10 seed 確認は行わない`, `smoke test (dev/train --smoke) を skip`,
  `RunPod を使わない (local CPU only)` → full local pipeline). Use whenever
  the user types `/experiment-execution`, or asks to run / execute / iterate /
  kick off an experiment, train a new agent, launch a RunPod run, propose a new
  case, or write up an experiment result — even if they don't explicitly say
  "execute", phrases like "ppo/case1 で GAE λ を試したい", "dqn/case2 を作って
  LunarLander で学習させて", "runpod で学習を回して結果まとめて", "新しい case を
  切って学習させたい", "iter5_plan.md に従って実装して回して", "H3 を実行して"
  all count. Don't trigger this skill for hypothesis-only discussion (use
  `experiment-plan` / `experiment-hypothesize`), interactive result
  interpretation after a run finishes (use `experiment-analysis`), read-only
  code review, or plain bug fixes.
---

# Experiment Execution Skill (reinforce)

Drives one hypothesis through plan → implementation → smoke → test → push → RunPod → monitoring → evaluation → `iterN_result.md` inline in the main session. Honors the `hypotheses.md` skip list to short-circuit phases (skip the 10-seed confirmation, skip smoke, skip RunPod, skip auto-recover).

## When this skill is in charge

- The user typed `/experiment-execution` (explicit trigger), **or**
- The user asked, in plain language, for any of:
  - launching a new case and training it
  - iterating an existing case and retraining
  - kicking off a RunPod training run
  - writing up the result of a recent run
- The user did **not** already pre-specify the entire pipeline themselves (e.g. "just run `dev/runpod train abc123`" — at that point, just run it).

If the user is asking for **read-only review**, **dead-code cleanup**, or **infra-only work**, redirect to the matching skill/agent.

## What "an experiment" means in this repo

- **Family** — the algorithm family directory under `backend/pipeline/` (`ppo` / `dqn` / `sac` / …).
- **Case** — `case<N>` under `backend/pipeline/<family>/`, append-only. `case1` of every family is the CPU smoke baseline.
- **Hypothesis** — one sentence + adoption threshold on the fixed-seed eval return.
- **Seeds** — the case's `config.yaml` seed list; ≥ 3 for anything reportable.
- **Compute target** — local CPU only / RunPod GPU.
- **Skip list** — read from `hypotheses.md > 実施しない検証 / 評価` and used to short-circuit phases.

Comply with `.claude/rules/docs.md` and `.claude/rules/backend/pipeline.md` (run-directory contract, checkpoint / resume, evaluation discipline).

## Skill flow

8 phases. Phase 4.5 (smoke test) is mandatory. Phase 6 specifies which `dev/runpod tail` source to use. Phase 6.5 makes pod terminate explicit on auto-recover.

### Phase 1 — Resolve the input source

Resolution priority:

1. **Explicit `iterN_plan.md`** — if a plan already exists, hypothesis / scope / metric / skip list are read from the plan; the skill only does minimal confirmation.
2. **`hypotheses.md` H{n} reference** — e.g. "H3 を実行して" → pull the line from `hypotheses.md`, internally call `experiment-plan` to write `iterN_plan.md`, then continue to Phase 2.
3. **Free-text hypothesis** — internally write `iterN_plan.md` first, then continue.

Items to confirm (extract from `iterN_plan.md` / `hypotheses.md`; ask 1–2 questions only if missing):

1. family / case
2. one-sentence hypothesis + scope
3. primary metric + seed count + baseline (case / run_id, same env id and eval seeds)
4. compute target (RunPod GPU / local CPU only)
5. skip list (smoke skip / dev/test skip / 学習曲線分析 skip / RunPod 不使用 / auto-recover 不使用 / 10 seed 確認なし)

### Phase 2 — Reality check

Parallel `Bash`:

```bash
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
ls backend/pipeline/<family>/
ls docs/experiment/<family>/
```

If the tree is dirty and a RunPod launch is planned, decide with the user (commit / stash / proceed without RunPod).

### Phase 3 — Confirm or write `iterN_plan.md`

If Phase 1 found no `iterN_plan.md`, run an `experiment-plan`-equivalent flow internally (skip list honored to auto-shorten `## 検証方法`). Otherwise reuse the existing plan.

If the user scoped the request to "plan only", stop here.

### Phase 4 — Implement the case

Comply with `.claude/rules/backend/pipeline.md`:

- The case directory has `README.md`, `config.yaml`, `env.py` (`make_env()`), `policy.py`, `train.py`, `evaluate.py`.
- Every hyper-parameter, the env id, and the seed list live in `config.yaml` — nothing hardcoded in `.py`.
- `train.py` is runnable as `python -m pipeline.<family>.case<N>.train [--config PATH] [--smoke]`, reads `REINFORCE_RUN_DIR` (falls back to `data/output/models/<family>/case<N>/runs/<run_id>/`), and writes `config.resolved.yaml`, `metrics.json`, `checkpoint_latest.*`, `best.*`, `eval/seed<K>.json`.
- Training resumes from `checkpoint_latest.*` when present; checkpoints include optimizer state and the RNG key.
- A progress marker is emitted at least every 5 minutes (the RunPod watcher kills a pod silent for 15 minutes).
- Every source of randomness is seeded from the config (numpy / torch / `jax.random.PRNGKey` / `env.reset(seed=...)`).

Never rewrite a case whose results are already recorded in `docs/experiment/` — create `case<N+1>` instead. For a pure config change on an unrecorded case, add a new config file rather than editing the baseline config, so the baseline stays reproducible.

When the case will run on RunPod, register it in `backend/src/gpu/runpod/config/cases.py` (`<family>_case<N>[_variant]` key).

### Phase 4.5 — Local smoke test (mandatory)

**Always run unless the skip list explicitly sets `smoke test (dev/train --smoke) を skip`**. When skipped, surface a one-line note before moving to Phase 5.

In order:

1. Import sanity:
   ```bash
   uv run --directory backend python -c "import pipeline.<family>.case<N>.train, pipeline.<family>.case<N>.evaluate"
   ```
2. **Smoke training run** (1 update on CPU; verifies the run-directory contract):
   ```bash
   dev/train --case <family>/case<N> --smoke
   ```
   Then check that the smoke run directory contains every file in the run-directory contract (`config.resolved.yaml`, `metrics.json`, `checkpoint_latest.*`, `best.*`, `eval/seed<K>.json`). A missing file means **no remote launch** — the RunPod preflight runs the same `--smoke` and refuses to launch on failure anyway.
3. **`dev/test`** (mandatory unless the skip list disables it):
   ```bash
   dev/test
   ```
4. `dev/lint` when new modules were added (mypy strict runs in CI).

On failure, optionally delegate to `python-build-resolver`. Always pass before progressing to RunPod.

### Phase 5 — Launch RunPod GPU

If the skip list sets `RunPod を使わない (local CPU only)`, **skip Phases 5 / 6 / 6.5 entirely**, run `dev/train --case <family>/case<N>` locally (once per seed if `train.py` takes a single seed), and go to Phase 7.

Otherwise:

```bash
git push origin <branch>
dev/runpod train <commit-sha> --case <registry key> [--cloud-type SECURE|COMMUNITY|ALL]
```

Pre-flight:
- `git status` clean
- commit pushed
- case registered in `backend/src/gpu/runpod/config/cases.py`
- `REINFORCE_DVC_BUCKET` set

Consider delegating GPU sizing and stock / cost checks to the `gpu-handler` agent. Save `run_id` / commit SHA / case / start time. Report 3 lines after launch and immediately set up the Phase 6.5 cron monitor.

### Phase 6 — Inline reactive monitoring during the run

Respond to user requests with the right command:

| user says | command | what you get |
|---|---|---|
| "今どこ?" / "進捗教えて" | `dev/runpod status <run_id>` | pod state / S3 progress marker / latest `metrics.json` scalars |
| "train ログ見せて" | `dev/runpod tail <run_id> --source train` (live) or `dev/runpod logs <run_id>` (post-mortem) | training stdout (step / loss terms / eval return) |
| "setup ログ見せて" | `dev/runpod tail <run_id> --source onstart` | onstart.log (env setup / uv sync / dvc / driver) |
| "GPU 使用率は?" | `dev/runpod tail <run_id> --source gpu` | nvidia-smi 10s sample (utilization / VRAM / temp) |
| "成果物今ある?" | `dev/runpod pull <run_id>` | DVC, falling back to S3 |
| "止めて" | confirm, then `dev/runpod stop <run_id>` | (surface forfeited cost in 1 line) |
| "学習率も変えたい" | treat as a new iter; do not touch the in-flight run; queue `iterN+1_plan.md` | — |

**NEVER** claim "succeeded" until artifacts (`best.*`, `metrics.json`, `eval/seed<K>.json`) are pulled and inspected.

For asynchronous notification on completion, use `dev/runpod watch <run_id>` (desktop notification) or `dev/runpod train --watch` at launch time.

### Phase 6.5 — Cron-driven periodic health check & auto-recover

If the skip list sets `auto-recover loop を使わない`, register the cron in monitor-only mode (notify, no stop / fix / relaunch).

Otherwise (default):

1. Use the `schedule` skill (cron) at 10-min cadence (extend to 15–20 min for very long ETAs).
2. Each tick runs `dev/runpod status <run_id>` (+ `dev/runpod logs <run_id> --tail 20` when unhealthy) and applies:

| state | action |
|---|---|
| RUNNING + step / metrics advancing | no-op |
| RUNNING + metrics stalled for 2 consecutive checks | soft warning to user; no auto-stop |
| RUNNING + eval return NaN / loss diverged | warn the user with the last 5 metric rows; ask before stopping |
| EXITED + success marker (`best.*` + `metrics.json` in S3) | cancel cron → Phase 7 |
| EXITED + failure (non-zero / OOM / crash) | **trigger auto-recover loop** |
| Cost cap exceeded | stop pod → surface to user immediately (no auto-relaunch) |

**Auto-recover loop** (failure path):

1. **Stop & terminate the pod** — `dev/runpod stop <run_id>`. Then **terminate / detach volume** to ensure the pod stops billing. Always perform an explicit terminate even if the pod looks crashed.
2. **Pull artifacts & logs**: `dev/runpod pull <run_id> --from s3` + `dev/runpod logs <run_id>`
3. **Diagnose**: env/setup error (onstart.log: uv sync, CUDA / cuDNN mismatch, DVC credentials) / code bug / config / OOM
4. **Fix** in the worktree:
   - Code/config bug → edit, `dev/train --smoke`, `dev/test`, commit + push
   - OOM → batch size / number of parallel envs / GPU class change
   - Onstart / env → fix the `backend/src/gpu/runpod/execution/onstart_lib/` script or case-level setup
   - Infra flake (RunPod outage / GPU node lottery) → surface to user and stop (no auto-relaunch)
5. **Re-launch** with the new SHA: `dev/runpod train <new-sha> --case <registry key>`. `train.py` resumes from `checkpoint_latest.*` if the persistent volume kept it.
6. **Re-arm cron** for the new `run_id`, continue Phase 6.5.

Each step emits a Japanese status message ≤ 3 lines. When Phase 7 starts or the user terminates, cancel the cron and confirm via `dev/runpod ps` that no stale pod remains.

### Phase 7 — Evaluation and `iterN_result.md`

Honor the skip list to determine evaluation scope:

- **default** → fixed-seed evaluation over the case's seed list (≥ 3 seeds), same evaluation seeds and env id as the baseline.
- **`10 seed 確認は行わない` OFF and the result is borderline** → propose a 10-seed confirmation run as the `次の一手` rather than running it silently.
- **fewer than 3 seeds finished** (crash, budget) → verdict is fixed at `inconclusive` (`seed<3 の結果で結論を出さない`).

In order:

1. `dev/runpod pull <run_id>` to fetch artifacts to `data/output/models/<family>/case<N>/runs/<run_id>/`.
2. `dev/evaluate --case <family>/case<N> --run <run_id>` — fixed-seed evaluation of `best.*`. Do the same for the baseline if its eval files are missing.
3. Decision is based on the **fixed-seed eval return only** (mean ± std and per-seed values vs baseline). The training-batch return is never the headline (`学習バッチの return は引用しない`).
4. Write `iterN_result.md` next to the plan:

```markdown
# {Family}/case{N} — {Topic} (iter{N}) RESULT

> 関連: iterN_plan.md / hypotheses.md
> run_id: {run_id} / commit: {sha} / case: {family}/case{N}
> env: {env id} / seeds: {list} / baseline: {case / run_id}
> 開始: {start} / 終了: {end} / wall-clock: {h} / ハードウェア: {GPU or CPU} / コスト: ${cost}

## Summary
{1 段落: 仮説は支持されたか?}

## Numbers

| metric | this run | baseline | Δ | note |
|---|---|---|---|---|
| eval return (mean ± std, n seeds) | | | | 固定 seed 評価 |
| eval return (per seed) | | | | |
| 目標 return 到達 step | | | | サンプル効率 |
| wall-clock | | | | |

## Diagnosis
{なぜ機能した / しなかったか。学習曲線のどこで差が出たか}

## Decision
- 採否: {adopted | rejected | inconclusive}
- 次の一手: {follow-up 仮説 / 10 seed 確認 / `dev/runpod promote <run_id>` (user 確認必須)}

## Artifacts
- run dir: `data/output/models/<family>/case<N>/runs/<run_id>/`
- best: `.../best.*` / metrics: `.../metrics.json` / eval: `.../eval/`
```

5. Report to the user in 3–5 lines: hypothesis result / eval return vs baseline / decision / result.md path.

### Phase 8 — Provisional write-back to `hypotheses.md`

Even if `experiment-analysis` will run later, append exactly **one row** to the `hypotheses.md` Iteration log when `iterN_result.md` is finalized:

```
| {N} | {start} | H{n} | iterN_plan.md | {run_id} | {eval return mean ± std (n seeds)} | {adopted/rejected/inconclusive} | iterN_result.md | (analysis 未実施) |
```

Do **not** flip the hypothesis-row checkbox here (that is `experiment-analysis` Phase 4.5's job). Phase 8 only adds the Iteration log row. A failed run that produced no evaluation gets a row with verdict `失敗`.

Skip Phase 8 entirely when `hypotheses.md` does not exist (one-shot invocation).

## Risk gates this skill enforces

- **Cost-cap overruns auto-stop.** Phase 6.5 stops the pod and never auto-relaunches when the cost cap is exceeded. Recovery from crash / OOM / bug is fine; budget overruns require user approval.
- **Smoke (`dev/train --smoke`) is mandatory by default.** Honor the skip list only when the user explicitly disabled it. This is the last line of defense before remote training spend.
- **Auto-recover always terminates the pod.** Stop alone is insufficient; ensure the pod is terminated and the volume detached so charges stop.
- **Cancel the Phase 6.5 cron when Phase 7 starts or the user terminates the run.**
- **Honor the `hypotheses.md` skip list** (10 seed / 学習曲線分析 / smoke / RunPod / auto-recover flags all map to phase short-circuits).
- **Hypothesis-row checkboxes belong to `experiment-analysis`.** This skill only appends an Iteration log row.
- **Fewer than 3 seeds → `inconclusive`.** A single-seed curve is a smoke test, not a finding.
- **Never use the training-batch return as evidence.** Headline = fixed-seed eval return.
- **Same env id and evaluation seeds as the baseline**, or the comparison is invalid.
- **Don't launch RunPod from a dirty tree.**
- **Canonical-weights update (`dev/runpod promote`) is out of scope.** The user must approve it individually.

## Common shapes

| User says… | Skill behavior |
|---|---|
| "ppo/case1 で GAE λ を 0.98 にして回して" | Phase 1 (1 question to confirm RunPod vs local) → Phases 2–8 inline. CartPole-scale → likely local CPU. |
| `/experiment-execution H3` | Phase 1 pulls H3 from `hypotheses.md` → internally write the plan → Phases 2–8. |
| "dqn/case2 の plan.md だけ先に書いて" | Phases 1–3 only; skip Phases 4 onwards. |
| "さっきの run xxx の結果を docs にまとめて" | Phase 1 + 7 + 8 only (pull + `dev/evaluate` + result.md + Iteration log row). |
| "今 RunPod で回ってる run は終わった?" | Phase 6 inline status. |
| "やっぱり学習率も変えたい" (mid-flight) | Treat as a new iter; do not touch the in-flight run; queue `iterN+1_plan.md`. |
| "smoke 飛ばして直接 RunPod" | Check the skip list; if absent, ask one confirmation; note that the RunPod preflight runs `--smoke` anyway. |

## Things to avoid

- Spawning a subagent for the long-running portion (Phases 5–6). Inline is required so the user can interject.
- Forgetting to cancel the Phase 6.5 cron at Phase 7 / on terminate.
- Auto-relaunching after a cost-overrun stop. Crashes / bugs / OOM are recoverable; budget overruns need explicit user approval.
- Polling more frequently than every 5 minutes (cache waste, no signal).
- Quoting the training-batch return, or a single seed, as the result.
- Rewriting a case whose results are already recorded (create the next case).
- Creating a new experiment directory when the user is iterating (use `iterN_*.md` instead).
- Putting machine-generated artifacts under `docs/experiment/` (those go to `data/output/experiment/`).
- Claiming a run finished before pulling and inspecting artifacts.
- Flipping the hypothesis-row checkbox in `hypotheses.md` (Phase 8 only adds Iteration log rows).
- Stopping a pod without terminating it. Attached volume = ongoing charges.

## Communication cadence

- One short message at each phase boundary (plan written / case implemented / smoke green / tests green / RunPod launched / run finished / result written). Each ≤ 3 lines.
- After RunPod launch, surface run_id + ETA, then return control. Do not narrate while waiting.
- Status checks: ≤ 3 lines (status / latest step + eval return / ETA delta).
- End each turn with what's next, e.g. "run_id=abc123, ETA ~45min — `status` で進捗、`止めて` で中断、`結果まとめて` で完了後の集計に進みます".

## Language

- Internal reasoning and thinking should be in English
- **All user-facing output, AskUserQuestion labels/descriptions, and result reports must be in Japanese** (per `.claude/CLAUDE.md`)
