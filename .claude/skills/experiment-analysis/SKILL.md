---
name: experiment-analysis
description: >
  Run-driven post-mortem skill for finished reinforce RL experiments. Selects
  the analysis target by priority (most recent iter only / last N iters /
  inconclusive only / specific iter), honors the case's `hypotheses.md` skip
  list (e.g. `学習曲線分析は実施しない`, `seed<3 の結果で結論を出さない`), and
  converts the selected run directories (`metrics.json`, `eval/seed<K>.json`,
  `config.resolved.yaml`) plus the baseline run into one compact Markdown via
  a Python script. Claude reads that file and produces three views — where the
  learning curve diverged from the baseline / which change helped / which
  seeds or training phases to inspect next — plus a summary and NEXT ACTION.
  Updates `hypotheses.md` only with the **adoption write-back** (matching
  hypothesis checkbox flipped to `- [x]`, verdict note appended, Iteration log
  row updated). The next-iteration decision (deepen / broaden / list-consume)
  is **not** made here — that belongs to `experiment-plan`. Read-only on code:
  does NOT re-train, NOT launch RunPod, NOT edit `backend/pipeline/`. Use
  whenever the user types `/experiment-analysis`, or asks to interpret / dig
  into / discuss / explain a finished experiment result — phrases like
  "学習曲線を分析して", "seed ごとの結果を比較して", "baseline とどこで差が
  ついたか見て", "なぜ seed 2 だけ崩壊したのか調べて", "iter4 の結果を深掘り
  したい", "loss の推移から原因を読み取って", "直近の inconclusive iter まとめて
  分析して" all count. Don't trigger this skill for designing a NEW experiment
  from scratch (use `experiment-plan` / `experiment-hypothesize`),
  full-pipeline execution / re-training (use `experiment-execution`),
  read-only code review, or plain bug fixes.
---

# Experiment Analysis Skill (reinforce)

Run-driven post-mortem. Honors the `hypotheses.md` skip list, selects the analysis target by priority, converts run directories to Markdown via a Python script, has Claude analyze, summarizes + suggests NEXT ACTION, and finally writes back the **verdict** for the analyzed iter to `hypotheses.md`. Raw `metrics.json` never enters Claude's context (the script does the conversion).

**Out of scope**: deciding what the next iteration should be (deepen / broaden / list-consume). That decision lives in `experiment-plan` Step 0.5, which reads this skill's verdict + the recent Iteration log to choose. The NEXT ACTION report from this skill is informational guidance, not a state mutation.

## Skill flow

6 phases. Phase 1.5 (target prioritization) and Phase 4.5 (`hypotheses.md` write-back) bracket the analysis. A skip mode bypasses the learning-curve conversion.

### Phase 1 — Identify the target case and `hypotheses.md`

`AskUserQuestion`, 1 question to confirm the target case (existing cases from `ls backend/pipeline/*/`, e.g. `ppo/case2` / `dqn/case1` / `Other` free-text). Skip when obvious from recent conversation.

After confirmation:

- Search and read `docs/experiment/<family>/{yyyymmdd}_case{N}_{topic}/hypotheses.md`.
- Cache the skip list / primary metric / default seed count / baseline / Iteration log.
- If `学習曲線分析は実施しない` is in the skip list, run in **skip mode** (no script conversion; analysis uses `iter*_result.md` only).

If `hypotheses.md` does not exist (one-shot experiments), proceed without skip-list awareness and ask for the baseline run in Phase 2.

If the analyzed iter has no `iterN_result.md` yet, redirect: result writing is `experiment-execution`'s job (Phase 7). Do not write `iterN_analysis.md` without an existing result file.

### Phase 1.5 — Prioritize the analysis target

`AskUserQuestion`, 1 question:

| Q | options |
|---|---|
| 分析対象の選び方 | `直前 iter のみ` ⭐推薦 / `直近 3 iter 全部 (横断比較)` / `直近 5 iter 全部` / `採否 inconclusive の iter のみ` / `特定 iter を指定` (Other で iter 番号) / `Other` |

Read the `iterN_result.md` of the selected iters to collect their `run_id`s and the baseline `run_id`.

When multiple iters are involved, pass all of their run directories to **one** script invocation (the script renders them as columns side by side). Cap at 5 runs + 1 baseline per invocation to stay within the Markdown size budget.

### Phase 2 — Resolve run directories (skippable)

If skip mode is on, **skip Phases 2–3** and go to Phase 4 (analyze `iter*_result.md` only).

Otherwise resolve absolute paths:

```
data/output/models/<family>/case<N>/runs/<run_id>/
```

- If a run directory is missing locally, fetch it with `dev/runpod pull <run_id>` (read-only for the pod; it only downloads artifacts). If it still is missing, report and continue in skip mode for that run.
- Worktree `data/` is a symlink to the main repo's `data/`; resolve and **pass absolute paths to the script**.
- The baseline is the run named in `hypotheses.md` (`baseline:`) or in `iterN_result.md`. If there is none, ask once; "baseline なし" is allowed (the divergence section is then omitted).

### Phase 3 — Markdown conversion via Python script

```bash
python3 .claude/skills/experiment-analysis/scripts/summarize_runs.py \
  --run <ABS_RUN_DIR_A> --label iter3 \
  [--run <ABS_RUN_DIR_B> --label iter4] \
  [--baseline <ABS_BASELINE_RUN_DIR> --baseline-label ppo_case1] \
  --out data/output/experiment/{family}/case{N}/analysis/{yyyymmdd_HHMM}/summary.md
```

stdout is exactly one line (`wrote <path> (N lines)`). **Do not paste `metrics.json` rows or per-step tables into chat.** Output goes under `data/output/experiment/`, never `docs/experiment/`; reference the path from `iterN_analysis.md`.

### Phase 4 — Read the Markdown and analyze

Claude `Read`s `summary.md` (in skip mode, only `iter*_result.md`) and must cover three views:

1. **where_diverged** — at which step the learning curve separated from the baseline (the script's `Divergence vs baseline` section), whether the gap is in the slope (sample efficiency) or the plateau (asymptotic return), and what the loss terms were doing at that point (e.g. entropy collapsed early, value loss spiked). Up to 3 candidate explanations, each with supporting / refuting evidence from the tables.
2. **what_helped** — which change in this iter plausibly caused the improvement (or lack of it), cross-checked against the config diff in the Meta section. Separate "all seeds moved" from "one seed moved the mean".
3. **where_to_focus_next** — the seeds or training phases to inspect next, e.g. "seed 2 は 40k step で eval return が崩壊 → その区間の entropy と grad norm を確認", "baseline との差は 20k step 以降の傾き → 学習率スケジュールを疑う".

Apply the evaluation discipline from `.claude/rules/backend/pipeline.md`:
- Headline = fixed-seed eval return. `学習バッチの return は引用しない`.
- With fewer than 3 seeds (the script prints a WARNING), frame findings as "what would confirm this" rather than conclusions (`seed<3 の結果で結論を出さない`).
- A gain that costs disproportionate wall-clock is reported as such.

Write `iterN_analysis.md` next to `iterN_result.md` with the three views, the summary.md path, and the confirmation plan.

### Phase 4.5 — Adoption write-back to `hypotheses.md`

Run only when `hypotheses.md` exists. Read → edit → write back. **Only the verdict for the analyzed iter is written.** The next-iteration decision (deepen / broaden / list-consume) is made by `experiment-plan` Step 0.5 when it reads this file later — do not touch `状態:` or append new hypothesis rows here.

**What to write back**:

1. **Hypothesis checkbox update** — change the `- [ ]` of the hypothesis tied to the analyzed iter (e.g. H3) to `- [x]` and append an adoption note inline:
   ```markdown
   - [x] (P1) H3: GAE λ 0.95→0.98 — adopted (iter3: eval return 480±12 vs 410±25, 3 seeds, 全 seed で baseline 超え)
   ```
2. **Update / replace the iter's row in the Iteration log table** — iter / start / hypothesis# / plan path / run_id / primary metric / verdict / result path / analysis path. (`experiment-execution` Phase 8 already added a provisional row marked `(analysis 未実施)`; replace that row with the final verdict + analysis path.)
3. Update **last-updated date** (`最終更新`) to session `currentDate`.

**Out of scope for this skill** (handled by `experiment-plan` Step 0.5 instead):

- ❌ Do **not** rewrite `状態:` (no auto `paused` here).
- ❌ Do **not** append `- [ ] (auto, deep) ...` follow-up hypotheses.
- ❌ Do **not** count consecutive rejections to trigger broaden.

The `where_to_focus_next` and follow-up suggestions go into `iterN_analysis.md` and Phase 6's NEXT ACTION report — `experiment-plan` reads those when deciding the next iter's mode.

When `hypotheses.md` does not exist, skip Phase 4.5 entirely.

### Phase 5 — Summary report

Structured Japanese report to the user (always include the paths to `summary.md` and `iterN_analysis.md`):

- **Headline** (omit in skip mode): eval return mean ± std per run vs baseline, n seeds, wall-clock
- **where_diverged**: divergence step + slope-vs-plateau + up to 3 candidate explanations with evidence
- **what_helped**: 1–3 findings, marking whether all seeds agree
- **where_to_focus_next**: seeds / step ranges / metrics to watch
- **`hypotheses.md` updates**: verdict written for the analyzed iter (checkbox + Iteration log row), in 1 line. The next-iter decision is left to `experiment-plan`.

### Phase 6 — NEXT ACTION suggestions

Suggest 2–3 concrete follow-ups, e.g.:

- "10 seed で同一傾向か確認" (increase n on the same case)
- "seed 2 の 30k–50k step を `metrics.json` から抽出して entropy / grad norm を確認"
- "次 iter で <仮説に基づく修正> を実装" (→ `/experiment-plan` will pick the mode and may auto-append a deepen hypothesis)
- "ジャンル全体の筋を見直したい" (→ `/experiment-plan` will detect broaden and recommend `/experiment-hypothesize`)
- "採用候補の重みを canonical にする" (→ `dev/runpod promote <run_id>`, user approval required)

Do not auto-spawn other skills. Suggest by name only.

## Python script: `scripts/summarize_runs.py`

Stdlib only; run with the system `python3` (no `uv` needed).

| arg | required | default | description |
|---|---|---|---|
| `--run DIR` | yes (1+) | — | absolute path to a run directory; repeat for side-by-side comparison |
| `--label LABEL` | no | run dir name | column label for each `--run`, in order |
| `--baseline DIR` | no | — | baseline run directory; enables the divergence section |
| `--baseline-label LABEL` | no | `baseline` | column label for the baseline |
| `--out PATH` | yes | — | output Markdown path |
| `--checkpoints N` | no | `10` | rows in the learning-curve table |

Inputs it understands (per the run-directory contract in `.claude/rules/backend/pipeline.md`):

- `metrics.json` — a JSON list of flat dicts (JSON Lines also accepted). `step`, `wall_clock`, `train_return`, `eval_return` are recognized; every other numeric key is treated as a loss / diagnostic term.
- `eval/seed<K>.json` — a dict with `eval_return` (or `mean_return` / `return`), a dict with a `returns` list, or a list of such dicts over time (final = last, best = max).
- `config.resolved.yaml` — top-level scalar keys only (first 20).

Output Markdown layout:

```markdown
# Run summary — {labels}

## Meta
- per run: path, rows, last step, wall_clock, steps/s, key config values

## Fixed-seed evaluation
| run | n seeds | final mean ± std | best mean | per-seed final |
> WARNING line when any run has fewer than 3 seeds

## Learning curve (eval_return)
| step | run A | run B | baseline |   (~N evenly spaced checkpoints)

## Divergence vs baseline
- per run: first step from which the curve stays >10% of the baseline's scale above / below

## Loss-term trends (first / middle / last)
| term | first | middle | last |   (per run, up to 10 terms)
```

Size guard: ≤ 400 lines / ≤ 30 KB; overflow is truncated with a marker line.

## Risk gates this skill enforces

- **Honor the `hypotheses.md` skip list.** When `学習曲線分析は実施しない` is set, Phases 2–3 are skipped.
- **Adoption write-back only.** Update the hypothesis-row checkbox + Iteration log row + `最終更新`. Do **not** touch `状態:`, do **not** append new hypothesis rows. State transitions belong to `experiment-plan` Step 0.5.
- **Fixed-seed eval return is the headline.** Never the training-batch return.
- **Fewer than 3 seeds → hypotheses, not conclusions.** Always include a confirmation plan.
- **Never paste raw `metrics.json` into chat.** Always go via the script.
- **No code edits / commits / training / RunPod launches.** `dev/runpod pull` for downloading artifacts is the only pod-related command allowed.
- **NEXT ACTION suggestions are advisory only.** Never auto-spawn other skills.

## Common shapes

| User says… | Skill behavior |
|---|---|
| `/experiment-analysis` | Phase 1 confirms the case → Phase 1.5 picks "直前 iter のみ" ⭐ → Phases 2–6. |
| "baseline とどこで差がついたか見て" | Phase 1.5 = "直前 iter"; Phase 4 weights where_diverged. |
| "なぜ seed 2 だけ崩壊したのか調べて" | Phase 3 as usual; Phase 4 focuses on the per-seed table and the loss terms around the collapse step. |
| "iter3 と iter4 を比べて" | Phase 1.5 = specific iters; one script call with two `--run` plus the baseline. |
| "直近 inconclusive iter まとめて" | Phase 1.5 = "inconclusive のみ"; one script call with all of them. |
| "学習曲線は要らない、result.md だけで分析して" | Skip mode (Phase 4 only). |
| "run はあるけど result.md はまだ" | Redirect to `experiment-execution` (Phase 7 writes result.md first). |

## Things to avoid

- Pasting raw `metrics.json` or per-step tables into chat. Use the script.
- Forgetting Phase 4.5 when `hypotheses.md` exists.
- Touching `状態:` or appending new hypothesis rows. Those are `experiment-plan` Step 0.5's domain.
- Treating a mean moved by one seed as a general effect.
- Comparing runs with different env ids or evaluation seeds without saying so.
- Auto-spawning `/experiment-hypothesize` or `/experiment-plan`.
- Putting machine-generated artifacts (summary.md, JSON, plots) under `docs/experiment/`. They go under `data/output/experiment/`.

## Language

- Internal reasoning and thinking should be in English
- **All user-facing output, AskUserQuestion labels/descriptions, and the analysis summary must be in Japanese** (per `.claude/CLAUDE.md`)
