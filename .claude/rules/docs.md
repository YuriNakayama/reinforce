---
paths:
  - "docs/**"
---

# Documentation Rules

Conventions for the `docs/` tree. Keep human-readable narrative under `docs/`; put machine-generated artifacts (JSON metrics, PNG plots, replay dumps) under `data/output/` so docs stay diff-friendly and lightweight.

## Top-level layout

```
docs/
  develop/        Explanations of code that already exists (see below)
  experiment/     Experiment plans and results (see structure below)
  plans/          Feature plans for tooling work (one directory per feature, e.g. docs/plans/terraform_dvc_remote/plan.md)
```

`docs/develop/reference/` holds measurements and design notes carried over from earlier projects (e.g. the JAX env throughput study). They do not change with each experiment iteration.

## `docs/develop/` structure

Explanations of code that **already exists** — how a module is laid out, what each piece is responsible for, how a request actually flows through it. Distinct from `docs/plans/`, which describes work not yet done.

```
docs/develop/code_explanation/{yyyymmdd}_{scope}.md
docs/develop/code_explanation/{yyyymmdd}_{scope}/     # when one document is not enough
```

- Written by hand in a three-pass shape: file tree → responsibilities and I/O → step-by-step trace of one real call (e.g. one `dev/runpod train` launch through onstart to `metrics.json`).
- `{scope}` is the subsystem being explained in snake_case (`simulate_jaxenv`, `gpu_runpod_onstart`).
- A single Markdown file is the default. Use a directory when the explanation splits into several pages; add a `README.md` there as the index.

### HTML exception

`docs/` is otherwise Markdown-only so diffs stay readable. **Code explanations may ship as HTML** when diagrams carry information Markdown cannot express (inline SVG, overlaid charts). When they do:

- Put them in their own dated directory with a `README.md` index explaining why HTML was chosen.
- State in that README that the HTML is **excluded from diff review** — reviewers read the rendered page, not the source.
- Keep the pages self-contained (inline CSS/SVG, no external assets) and make any cross-page links **relative**, so the directory works offline after a plain `git clone`.
- Numbers quoted in the explanation should say whether they were measured or read from a spec, and against which dependency version — measured figures go stale when the dependency moves.

## `docs/experiment/` structure

Use this exact path scheme:

```
docs/experiment/{family}/{yyyymmdd}_case{N}_{topic}/{plan|result|etc.}.md
```

- **First segment**: the algorithm family the experiment targets — the directory name under `backend/pipeline/` (`ppo`, `dqn`, `sac`, …).
- **Directory name**: `{yyyymmdd}_case{N}_{topic}` — the date the experiment started, the case number it operates on, and a short snake_case topic.
- **File name**: `plan.md` for proposals (before running), `result.md` for outcomes (after running), `hypotheses.md` for the case-level hypothesis list + skip list (single source of truth for the `experiment` loop driver — created by `experiment-hypothesize`, updated by `experiment-execution` / `experiment-analysis`), or another short noun (e.g. `analysis.md`, `notes.md`) when none fit. A single experiment directory may contain `hypotheses.md` together with paired `plan.md` / `result.md` (or their `iter{N}_*` variants).
- **Iterations**: when the same experiment goes through multiple iterations that each warrant their own document, do NOT create new directories. Add files within the existing directory using an `iter{N}_` prefix (e.g. `iter1_result.md`, `iter2_plan.md`, `iter2_result.md`). `hypotheses.md` is **not** iter-prefixed — there is one per directory, and iterations are tracked inside it via the `Iteration log` section + checkbox state on each hypothesis. Keep all iterations under the same directory so the experiment's history stays together.
- **Iteration migration**: when introducing the `iter{N}_` scheme into a directory that already has a plain `result.md` / `plan.md`, rename the existing file to `iter1_result.md` / `iter1_plan.md` first, then add `iter2_*.md` for the new work. Don't leave the original unprefixed file alongside `iter2_*.md` — it makes the iteration order ambiguous.

### Examples

```
docs/experiment/ppo/20260926_case1_cartpole_baseline/result.md
docs/experiment/ppo/20261003_case2_gae_lambda/plan.md
docs/experiment/dqn/20261010_case1_lunarlander/plan.md
docs/experiment/ppo/20261003_case2_gae_lambda/iter1_result.md   # iteration variant
docs/experiment/ppo/20261003_case2_gae_lambda/iter2_result.md
docs/experiment/ppo/20261003_case2_gae_lambda/hypotheses.md     # case-level hypothesis list (one per dir, no iter prefix)
```

### `hypotheses.md` (case-level hypothesis list)

`hypotheses.md` is the **single source of truth** for the `experiment` loop driver. It captures the hypothesis list to be tested, the priority order, and — critically — the **skip list** (`実施しない検証 / 評価`) that downstream skills (`experiment-plan` / `experiment-execution` / `experiment-analysis`) honor to short-circuit their own phases.

Required sections:

- **Header frontmatter** (Markdown blockquote): created date / last updated / state (`in_progress` / `paused` / `stopped` / `completed`) / max iterations / primary metrics (default: fixed-seed eval return) / default seed count (≥ 3) and eval episodes per seed / env id / baseline case.
- **`## 実施しない検証 / 評価 (skip list)`** — must always be present. Subsections:
  - `### 評価` (Evaluation) — e.g. `do not run the 10-seed confirmation (adopt/reject on 3 seeds)`, `do not quote the training-batch return`, `decide adoption on the fixed-seed eval only`
  - `### 分析` (Analysis) — e.g. `do not run trajectory replay analysis`, `do not draw conclusions from n<3 seeds`
  - `### 実行` (Execution) — e.g. `skip smoke test (dev/train --smoke)` ⚠️, `skip dev/test` ⚠️, `do not use RunPod GPU (local CPU only)`, `do not use the auto-recover loop`
  - `### 例外条件` (Exception conditions) — per-hypothesis overrides (e.g. `add a 10-seed confirmation for H4 only when inconclusive`)
- **`## 仮説リスト (priority 順)`** (Hypothesis list, in priority order) — checkbox + priority label + (optional) `depends on H{m}`:
  ```markdown
  - [ ] (P1) H1: entropy coef 0.01→0.02 — expect eval return +20 from better exploration
  - [x] (P2) H3: GAE λ 0.95→0.98 — adopted (iter9)
  - [ ] (P2, depends on H1) H4: entropy 0.02 + normalise advantages per minibatch
  - [ ] (deferred) H5: recurrent (GRU) policy — high effort
  ```
- **`## Iteration log`** — table (iter / start / hypothesis# / plan path / run_id / primary metrics / adopt-reject / result path / analysis path). `experiment-execution` Phase 8 appends a row; `experiment-analysis` Phase 4.5 updates the checkbox + adoption note.

Project-default skip-list entries (always include unless the user explicitly removes them):

- `学習バッチの return は引用しない` (headline numbers come from the fixed-seed eval)
- `seed<3 の結果で結論を出さない`

Lifecycle:

- Created **once** per case directory by `experiment-hypothesize`.
- Updated by `experiment-execution` (Iteration log row append) and `experiment-analysis` (checkbox + adoption note + state-transition flag).
- Read by `experiment-plan` (skip list → auto-shortens the plan's `## 検証方法` section) and `experiment` (loop driver — picks the next `- [ ]` hypothesis).

> **Note:** The Japanese strings quoted in backticks above (`## 実施しない検証 / 評価`, `## 仮説リスト`, `### 評価` / `### 分析` / `### 実行` / `### 例外条件`, `## 検証方法`, and the default skip-list entries `学習バッチの return は引用しない` / `seed<3 の結果で結論を出さない`) are **literal contract strings** that the `experiment-*` skills emit and pattern-match against. They are part of the `hypotheses.md` schema, not prose — do not translate them, or the skill cross-references will break.
- Not iter-prefixed. One per case directory.

### Anti-patterns

- Don't put dates in file names (`20260419_diagnosis_result.md`). The date belongs to the directory.
- Don't omit `case{N}` — every experiment is scoped to a specific case.
- Don't drop non-Markdown artifacts (`*.json`, `*.png`, replay dumps) into `docs/experiment/`. Write them under `data/output/experiment/` and link from the Markdown if needed.
- Don't mix experiment results into `docs/develop/reference/`. Reference material is static; experiments are work logs.

## Cross-references

- Use repo-rooted paths (e.g. `docs/experiment/ppo/20260926_case1_cartpole_baseline/result.md`) when referencing across categories — they survive future reorgs better than relative paths.
- Within the same category subtree, relative paths (`../20261003_case2_gae_lambda/result.md`) are fine.
- When referencing artifacts under `data/output/`, write the path as-is (`data/output/experiment/ppo_case1_eval_curve.json`); these files are not git-tracked but are reproducible via DVC.

## When to add a new experiment directory

A new directory is warranted when an experiment has a distinct hypothesis, dataset, or evaluation. Iterations of the same hypothesis stay in the same directory: either appended into the existing `result.md`, or split across `iter{N}_result.md` files when each iteration is large enough to warrant its own document.
