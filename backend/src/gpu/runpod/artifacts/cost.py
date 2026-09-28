"""run.json を集計して RunPod の月次コストレポートを生成する。

run dir は他の GPU 基盤と共有するため、`runpod_offer_snapshot` を持たない
run (単価不明) は集計対象から除外する。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunCost:
    run_id: str
    git_sha: str
    gpu_type_id: str | None
    cloud_type: str | None
    dph_total: float | None
    runtime_seconds: float
    cost_usd: float
    status: str
    created_at: str


@dataclass(frozen=True)
class CostReport:
    month: str  # "YYYY-MM" or "all"
    runs: tuple[RunCost, ...]

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.runs)

    @property
    def adopted_count(self) -> int:
        return sum(1 for r in self.runs if r.status == "adopted")

    @property
    def average_cost_usd(self) -> float:
        if not self.runs:
            return 0.0
        return self.total_cost_usd / len(self.runs)


def _load_run(run_json: Path) -> RunCost | None:
    """run.json を RunCost に変換する。課金情報の無い run は None。"""
    try:
        data: dict[str, Any] = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    snapshot = data.get("runpod_offer_snapshot")
    if not isinstance(snapshot, dict):
        # runpod_offer_snapshot が無い run (interactive 起動や
        # 他基盤経由) は時間単価が不明なので集計対象外。
        return None
    metrics = data.get("train_metrics") or {}
    runtime = float(metrics.get("runtime_seconds", 0.0))
    dph = snapshot.get("dph_total")
    cost = (float(dph) * runtime / 3600.0) if dph else 0.0
    return RunCost(
        run_id=str(data.get("run_id", run_json.parent.name)),
        git_sha=str(data.get("git_sha", "")),
        gpu_type_id=snapshot.get("gpu_type_id"),
        cloud_type=snapshot.get("cloud_type"),
        dph_total=float(dph) if dph else None,
        runtime_seconds=runtime,
        cost_usd=round(cost, 4),
        status=str(data.get("status", "unknown")),
        created_at=str(data.get("created_at", "")),
    )


def _matches_month(created_at: str, month: str | None) -> bool:
    if month is None:
        return True
    return created_at.startswith(month)


def aggregate_runs(runs_root: Path, month: str | None = None) -> CostReport:
    """`runs_root/*/run.json` を全走査し、月単位で RunPod run のみ集計する。"""
    if not runs_root.is_dir():
        return CostReport(month=month or "all", runs=())
    collected: list[RunCost] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_json = run_dir / "run.json"
        if not run_json.is_file():
            continue
        rc = _load_run(run_json)
        if rc is None:
            continue
        if _matches_month(rc.created_at, month):
            collected.append(rc)
    return CostReport(month=month or "all", runs=tuple(collected))


def render_markdown(report: CostReport) -> str:
    """| run_id | gpu | cloud | dph | runtime | cost | status | テーブルと summary。"""
    header = f"# RunPod Cost Report — {report.month}\n\n"
    summary = (
        f"- runs: {len(report.runs)}\n"
        f"- adopted: {report.adopted_count}\n"
        f"- total cost: ${report.total_cost_usd:.4f}\n"
        f"- average cost: ${report.average_cost_usd:.4f}\n\n"
    )
    if not report.runs:
        return header + summary + "(no runs)\n"
    lines = [
        "| run_id | gpu | cloud | dph | runtime (s) | cost (USD) | status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for rc in report.runs:
        lines.append(
            f"| {rc.run_id} | {rc.gpu_type_id or '?'} | "
            f"{rc.cloud_type or '?'} | "
            f"${rc.dph_total or 0.0:.3f} | {rc.runtime_seconds:.1f} | "
            f"${rc.cost_usd:.4f} | {rc.status} |"
        )
    return header + summary + "\n".join(lines) + "\n"


_MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def parse_month(value: str | None) -> str | None:
    """`YYYY-MM` (zero-padded) を validate し、無効なら ValueError。"""
    if value is None:
        return None
    if not _MONTH_PATTERN.match(value):
        raise ValueError(f"month must be 'YYYY-MM' (zero-padded), got {value!r}")
    try:
        datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise ValueError(f"month must be 'YYYY-MM', got {value!r}") from exc
    return value


def iter_run_dirs(runs_root: Path) -> Iterable[Path]:
    """run.json を持つ run dir を yield する (CLI 補助)。"""
    if not runs_root.is_dir():
        return
    for run_dir in sorted(runs_root.iterdir()):
        if run_dir.is_dir() and (run_dir / "run.json").is_file():
            yield run_dir
