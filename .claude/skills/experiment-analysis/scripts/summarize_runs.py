"""Summarize RL training runs into a compact Markdown file for experiment-analysis.

Reads the run-directory contract from `.claude/rules/backend/pipeline.md`:

    <run_dir>/config.resolved.yaml   exact config used (optional here)
    <run_dir>/metrics.json           list of flat dicts: step, wall_clock,
                                     train_return, eval_return, loss terms...
    <run_dir>/eval/seed<K>.json      per-seed evaluation results

and writes one Markdown file with: meta, per-seed final/best eval return,
a learning-curve table at ~N checkpoints, the divergence point versus an
optional baseline run, loss-term trends, and wall-clock.

Stdout is a single line (`wrote <path> (<n> lines)`) so the calling LLM only
reads the on-disk Markdown, never raw metrics. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_LINES = 400
MAX_BYTES = 30_000
MAX_CONFIG_KEYS = 20
MAX_LOSS_KEYS = 10
DIVERGENCE_REL = 0.10

RESERVED_KEYS = frozenset(
    {"step", "wall_clock", "train_return", "eval_return", "seed", "run_id"}
)
RETURN_KEYS = ("eval_return", "mean_return", "return", "episode_return")
_SCALAR_LINE = re.compile(r"^([A-Za-z_][\w.-]*):\s*([^#{}\[\]]+?)\s*(?:#.*)?$")

Point = tuple[float, float]


@dataclass
class SeedEval:
    seed: str
    final: float | None
    best: float | None


@dataclass
class Run:
    label: str
    path: Path
    config: dict[str, str] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    seeds: list[SeedEval] = field(default_factory=list)

    def series(self, key: str) -> list[Point]:
        out: list[Point] = []
        for i, row in enumerate(self.rows):
            value = _num(row.get(key))
            if value is None:
                continue
            step = _num(row.get("step"))
            out.append((step if step is not None else float(i), value))
        return out


def _num(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if value != 0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _read_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    config: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _SCALAR_LINE.match(line)
        if match:
            config[match.group(1)] = match.group(2).strip().strip("'\"")
        if len(config) >= MAX_CONFIG_KEYS:
            break
    return config


def _eval_points(obj: Any) -> list[Point]:
    """Extract (step, return) pairs from one eval/seed<K>.json payload."""
    if isinstance(obj, list):
        points: list[Point] = []
        for i, item in enumerate(obj):
            points.extend(
                (float(i) if step is None else step, value)
                for step, value in _eval_points(item)
            )
        return points
    if not isinstance(obj, dict):
        value = _num(obj)
        return [] if value is None else [(0.0, value)]
    if isinstance(obj.get("evals"), list):
        return _eval_points(obj["evals"])
    returns = obj.get("returns")
    step = _num(obj.get("step")) or 0.0
    if isinstance(returns, list):
        values = [v for v in (_num(r) for r in returns) if v is not None]
        return [(step, sum(values) / len(values))] if values else []
    for key in RETURN_KEYS:
        value = _num(obj.get(key))
        if value is not None:
            return [(step, value)]
    return []


def _read_seeds(eval_dir: Path) -> list[SeedEval]:
    seeds: list[SeedEval] = []
    for path in sorted(eval_dir.glob("seed*.json")):
        try:
            points = _eval_points(_load_json(path))
        except (OSError, json.JSONDecodeError):
            points = []
        values = [value for _, value in points]
        seeds.append(
            SeedEval(
                seed=path.stem.removeprefix("seed"),
                final=values[-1] if values else None,
                best=max(values) if values else None,
            )
        )
    return seeds


def load_run(path: Path, label: str) -> Run:
    run = Run(label=label, path=path)
    run.config = _read_config(path / "config.resolved.yaml")
    metrics = path / "metrics.json"
    if metrics.is_file():
        payload = _load_json(metrics)
        if isinstance(payload, dict):
            payload = payload.get("rows", [payload])
        run.rows = [row for row in payload if isinstance(row, dict)]
    run.seeds = _read_seeds(path / "eval")
    return run


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def _value_at(series: list[Point], step: float) -> float | None:
    latest: float | None = None
    for s, value in series:
        if s > step:
            break
        latest = value
    return latest


def _checkpoints(runs: list[Run], count: int) -> list[float]:
    steps = [s for run in runs for s, _ in run.series("eval_return")]
    if not steps:
        return []
    lo, hi = min(steps), max(steps)
    if hi == lo or count <= 1:
        return [hi]
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def section_meta(runs: list[Run], baseline: Run | None) -> list[str]:
    lines = ["## Meta", ""]
    for run in [*runs, *([baseline] if baseline else [])]:
        wall = run.series("wall_clock")
        steps = run.series("eval_return") or run.series("train_return")
        last_step = steps[-1][0] if steps else None
        wall_s = wall[-1][1] if wall else None
        rate = last_step / wall_s if last_step and wall_s else None
        lines.append(
            f"- **{run.label}** `{run.path}` — rows={len(run.rows)}, "
            f"last step={_fmt(last_step)}, wall_clock={_fmt(wall_s)}s, "
            f"steps/s={_fmt(rate)}"
        )
        if run.config:
            pairs = ", ".join(f"{k}={v}" for k, v in run.config.items())
            lines.append(f"  - config: {pairs}")
    return [*lines, ""]


def section_seeds(runs: list[Run], baseline: Run | None) -> list[str]:
    lines = [
        "## Fixed-seed evaluation",
        "",
        "| run | n seeds | final mean ± std | best mean | per-seed final |",
        "|---|---|---|---|---|",
    ]
    everyone = [*runs, *([baseline] if baseline else [])]
    for run in everyone:
        finals = [s.final for s in run.seeds if s.final is not None]
        bests = [s.best for s in run.seeds if s.best is not None]
        mean, std = _mean_std(finals)
        best_mean, _ = _mean_std(bests)
        per_seed = ", ".join(f"s{s.seed}={_fmt(s.final)}" for s in run.seeds)
        lines.append(
            f"| {run.label} | {len(run.seeds)} | {_fmt(mean)} ± {_fmt(std)} "
            f"| {_fmt(best_mean)} | {per_seed or '-'} |"
        )
    few = [run.label for run in everyone if len(run.seeds) < 3]
    if few:
        names = ", ".join(few)
        lines += [
            "",
            f"> WARNING: fewer than 3 seeds ({names}) — seed<3 の結果で結論を出さない",
        ]
    return [*lines, ""]


def section_curve(runs: list[Run], baseline: Run | None, count: int) -> list[str]:
    everyone = [*runs, *([baseline] if baseline else [])]
    points = _checkpoints(everyone, count)
    if not points:
        return ["## Learning curve (eval_return)", "", "(no eval_return rows)", ""]
    header = "| step | " + " | ".join(run.label for run in everyone) + " |"
    lines = ["## Learning curve (eval_return)", "", header]
    lines.append("|---" * (len(everyone) + 1) + "|")
    series = [run.series("eval_return") for run in everyone]
    for step in points:
        cells = " | ".join(_fmt(_value_at(s, step)) for s in series)
        lines.append(f"| {_fmt(step)} | {cells} |")
    return [*lines, ""]


def _divergence(run: Run, baseline: Run, points: list[float]) -> str:
    mine, base = run.series("eval_return"), baseline.series("eval_return")
    if not mine or not base:
        return "n/a (missing eval_return)"
    scale = max(abs(v) for _, v in base) or 1.0
    signs: list[tuple[float, int]] = []
    for step in points:
        a, b = _value_at(mine, step), _value_at(base, step)
        if a is None or b is None:
            continue
        diff = (a - b) / scale
        signs.append((step, 0 if abs(diff) < DIVERGENCE_REL else int(diff > 0)))
    for i, (step, sign) in enumerate(signs):
        rest = [s for _, s in signs[i:]]
        if sign != 0 and all(s == sign for s in rest):
            word = "above" if sign == 1 else "below"
            return f"from step {_fmt(step)} stays {word} baseline (>10% of scale)"
    return "no sustained divergence (>10% of baseline scale)"


def section_divergence(runs: list[Run], baseline: Run | None, count: int) -> list[str]:
    if baseline is None:
        return []
    points = _checkpoints([*runs, baseline], count * 2)
    lines = ["## Divergence vs baseline", ""]
    lines += [
        f"- **{run.label}**: {_divergence(run, baseline, points)}" for run in runs
    ]
    return [*lines, ""]


def section_losses(runs: list[Run]) -> list[str]:
    lines = ["## Loss-term trends (first / middle / last)", ""]
    for run in runs:
        keys = sorted(
            {
                k
                for row in run.rows
                for k, v in row.items()
                if k not in RESERVED_KEYS and _num(v) is not None
            }
        )[:MAX_LOSS_KEYS]
        if not keys:
            continue
        lines += [f"**{run.label}**", "", "| term | first | middle | last |"]
        lines.append("|---|---|---|---|")
        for key in keys:
            values = [v for _, v in run.series(key)]
            middle = values[len(values) // 2]
            lines.append(
                f"| {key} | {_fmt(values[0])} | {_fmt(middle)} | {_fmt(values[-1])} |"
            )
        lines.append("")
    return lines


def render(runs: list[Run], baseline: Run | None, count: int) -> list[str]:
    title = ", ".join(run.label for run in runs)
    lines = [f"# Run summary — {title}", ""]
    lines += section_meta(runs, baseline)
    lines += section_seeds(runs, baseline)
    lines += section_curve(runs, baseline, count)
    lines += section_divergence(runs, baseline, count)
    lines += section_losses(runs)
    return lines


def enforce_budget(lines: list[str]) -> list[str]:
    kept: list[str] = []
    size = 0
    for line in lines:
        size += len(line.encode("utf-8")) + 1
        if len(kept) >= MAX_LINES or size > MAX_BYTES:
            kept.append("> (truncated: size budget exceeded)")
            break
        kept.append(line)
    return kept


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--checkpoints", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    labels = list(args.label) + [p.name for p in args.run[len(args.label) :]]
    missing = [p for p in [*args.run, args.baseline] if p and not p.is_dir()]
    if missing:
        print(f"error: run dir not found: {missing[0]}", file=sys.stderr)
        return 2
    runs = [load_run(p, label) for p, label in zip(args.run, labels, strict=False)]
    baseline = load_run(args.baseline, args.baseline_label) if args.baseline else None
    lines = enforce_budget(render(runs, baseline, args.checkpoints))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
