"""Kaggle Kernel run の月次 free-hour 集計 (cost-report 出力)。

vast / runpod と異なり金額は 0、``runtime_seconds`` の合計を出すだけ。
出力先: ``docs/experiment/kaggle_kernel_cost_report_<YYYY-MM>.md``
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KaggleRunRow:
    """1 run の集計結果 (rendering 用)。"""

    run_id: str
    case: str
    started_at: str
    kernel_slug: str
    accelerator: str
    runtime_seconds: int
    status: str


@dataclass(frozen=True)
class CostReport:
    """月次集計レポート。"""

    month: str
    rows: list[KaggleRunRow] = field(default_factory=list)

    @property
    def total_runtime_seconds(self) -> int:
        return sum(r.runtime_seconds for r in self.rows)

    @property
    def total_gpu_hours(self) -> float:
        return self.total_runtime_seconds / 3600.0


def aggregate_runs(runs_root: Path, month: str) -> CostReport:
    """``<runs_root>/**/run.json`` を全走査し、``kaggle_kernel_meta`` を持つ
    run のみを ``month`` (``YYYY-MM``) で filter して集計する。
    """
    if not _MONTH_OK(month):
        raise ValueError(f"invalid month format (expected YYYY-MM): {month!r}")
    rows: list[KaggleRunRow] = []
    for run_json in sorted(runs_root.rglob("run.json")):
        try:
            data = json.loads(run_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("skip unparseable run.json: %s", run_json)
            continue
        kk_meta = data.get("kaggle_kernel_meta")
        if not isinstance(kk_meta, dict):
            continue
        started_at = str(kk_meta.get("started_at") or data.get("created_at") or "")
        if not started_at.startswith(month):
            continue
        rows.append(
            KaggleRunRow(
                run_id=str(data.get("run_id") or run_json.parent.name),
                case=_case_from_path(run_json),
                started_at=started_at,
                kernel_slug=str(kk_meta.get("kernel_slug") or ""),
                accelerator=str(kk_meta.get("accelerator") or ""),
                runtime_seconds=int(kk_meta.get("runtime_seconds") or 0),
                status=str(data.get("status") or ""),
            )
        )
    return CostReport(month=month, rows=rows)


def render_markdown(report: CostReport) -> str:
    """CostReport を Markdown table に整形する。"""
    lines: list[str] = [
        f"# Kaggle Kernel cost report — {report.month}",
        "",
        f"- Total runs: **{len(report.rows)}**",
        f"- Total GPU hours used: **{report.total_gpu_hours:.2f}** "
        f"(= {report.total_runtime_seconds}s)",
        "",
        "| run_id | case | started_at | accelerator | runtime (s) | status |",
        "|---|---|---|---|---:|---|",
    ]
    for row in report.rows:
        lines.append(
            f"| {row.run_id} | {row.case} | {row.started_at} | "
            f"{row.accelerator} | {row.runtime_seconds} | {row.status} |"
        )
    return "\n".join(lines) + "\n"


def default_report_path(docs_experiment_dir: Path, month: str) -> Path:
    return docs_experiment_dir / f"kaggle_kernel_cost_report_{month}.md"


def _MONTH_OK(month: str) -> bool:
    return (
        len(month) == 7
        and month[4] == "-"
        and month[:4].isdigit()
        and month[5:].isdigit()
        and 1 <= int(month[5:]) <= 12
    )


def _case_from_path(run_json: Path) -> str:
    """``data/output/models/<family>/<case>/runs/<run_id>/run.json`` から
    case を抽出する helper。"""
    parts = run_json.parts
    if "runs" in parts:
        idx = parts.index("runs")
        if idx >= 1:
            return parts[idx - 1]
    return ""
