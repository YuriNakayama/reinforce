"""1 つの run の "現在の状態 + 成果物" を 1 dataclass にまとめる集約器。

`dev/runpod summary <run_id>` の本体。複数の source (launch.json / run.json /
S3 marker / S3 artifacts / DVC meta) を merge して以下を返す:

- 現在の status (running / succeeded / failed / stalled / unknown)
- last_step + 何秒前から進捗なし
- 推定コスト (実行時間 × dph_total)
- 成果物の存在状況 (local / S3 artifacts / S3 DVC meta の 3 列)
- final_step / final_train_return / final_eval_return (run dir の metrics.json から)

外部副作用 (S3, SDK) は呼び出し側で stub できるよう module 引数で受ける。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu.runpod.artifacts.launch import read_launch_json
from gpu.runpod.config.cases import runs_root_for
from gpu.runpod.execution import progress as progress_mod


@dataclass(frozen=True)
class ArtifactStatus:
    """1 ファイルの存在状況を 3 経路で確認した結果。"""

    name: str
    in_run_dir: bool  # ローカル run_dir に存在
    in_s3_artifacts: bool  # s3://.../runpod_artifacts/<run>/<name>


@dataclass(frozen=True)
class RunSummary:
    """1 run の集約状態。"""

    run_id: str
    pod_id: str | None
    status: str  # "succeeded" / "failed" / "stalled" / "running" / "unknown"
    last_step: str | None
    last_step_age_seconds: int | None
    elapsed_seconds: int | None
    estimated_cost_usd: float | None
    artifacts: tuple[ArtifactStatus, ...]
    final_step: int | None
    final_train_return: float | None
    final_eval_return: float | None
    dvc_meta_committed: bool


def _safe_load_metrics(path: Path) -> list[dict[str, Any]]:
    """Load metrics.json as a list of dict rows; anything else yields []."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _exists_in_run_dir(run_dir: Path, name: str) -> bool:
    if name.endswith(".*"):
        return any(p.is_file() for p in run_dir.glob(name))
    return (run_dir / name).is_file()


def _exists_in_s3(s3_files: set[str], name: str) -> bool:
    if name.endswith(".*"):
        prefix = name[:-1]
        return any(f.startswith(prefix) for f in s3_files)
    return name in s3_files


def _classify_status_from_markers(steps: list[str]) -> str:
    """marker 列から `succeeded`/`failed`/`running` のいずれかを返す。"""
    if "99_done" in steps:
        return "succeeded"
    cleanup_failures = [
        s for s in steps if s.startswith("90_cleanup_exit_") and not s.endswith("_0")
    ]
    if cleanup_failures:
        return "failed"
    return "running"


def build_summary(
    run_id: str,
    *,
    repo_root: Path,
    case: str,
    aws_profile: str | None = None,
    progress_module: Any = progress_mod,
) -> RunSummary:
    """主要な source を merge して RunSummary を返す。"""
    relative = runs_root_for(case) / run_id
    run_dir = repo_root / relative

    # 1) launch.json
    pod_id: str | None = None
    dph_total: float | None = None
    try:
        launch = read_launch_json(run_dir)
        pod_id = str(launch.get("pod_id") or "") or None
        dph_total = launch.get("dph_total")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # 2) S3 markers
    try:
        markers = progress_module.list_markers(run_id, profile=aws_profile)
    except Exception:  # noqa: BLE001
        markers = []
    steps = [m.step for m in markers]
    last_step: str | None = steps[-1] if steps else None
    status = _classify_status_from_markers(steps)
    last_step_age_seconds: int | None = None
    elapsed_seconds: int | None = None
    if markers:
        from datetime import UTC, datetime

        try:
            first_ts = datetime.fromisoformat(
                markers[0].timestamp.replace("Z", "+00:00")
            )
            last_ts = datetime.fromisoformat(
                markers[-1].timestamp.replace("Z", "+00:00")
            )
            now = datetime.now(UTC)
            elapsed_seconds = int((last_ts - first_ts).total_seconds())
            last_step_age_seconds = int((now - last_ts).total_seconds())
        except ValueError:
            pass

    # 3) metrics.json (run-dir contract: append-only list of flat scalar rows,
    #    e.g. {"step", "wall_clock", "train_return", "eval_return", ...}).
    #    Rows without eval_return are normal between evaluations, so the last
    #    eval_return is searched backwards independently of the last step.
    final_step: int | None = None
    final_train_return: float | None = None
    final_eval_return: float | None = None
    rows = _safe_load_metrics(run_dir / "metrics.json")
    if rows:
        last = rows[-1]
        step = last.get("step")
        if isinstance(step, int):
            final_step = step
        final_train_return = _as_float(last.get("train_return"))
        for row in reversed(rows):
            value = _as_float(row.get("eval_return"))
            if value is not None:
                final_eval_return = value
                break

    # 4) artifact 存在確認
    try:
        s3_files = set(progress_module.list_artifacts(run_id, profile=aws_profile))
    except Exception:  # noqa: BLE001
        s3_files = set()

    # `best.*` / `checkpoint_latest.*` carry a framework-specific suffix
    # (.eqx / .pt / ...), so they are matched by prefix.
    artifact_names = (
        "best.*",
        "checkpoint_latest.*",
        "metrics.json",
        "run.json",
        "onstart.log",
    )
    artifacts = tuple(
        ArtifactStatus(
            name=name,
            in_run_dir=_exists_in_run_dir(run_dir, name),
            in_s3_artifacts=_exists_in_s3(s3_files, name),
        )
        for name in artifact_names
    )

    # 5) DVC meta が origin に上がっているか
    dvc_meta = repo_root / f"{relative}.dvc"
    dvc_meta_committed = dvc_meta.is_file()

    # 6) 推定コスト
    estimated_cost_usd: float | None = None
    if dph_total is not None and elapsed_seconds is not None:
        try:
            estimated_cost_usd = round(float(dph_total) * (elapsed_seconds / 3600.0), 4)
        except (TypeError, ValueError):
            pass

    return RunSummary(
        run_id=run_id,
        pod_id=pod_id,
        status=status,
        last_step=last_step,
        last_step_age_seconds=last_step_age_seconds,
        elapsed_seconds=elapsed_seconds,
        estimated_cost_usd=estimated_cost_usd,
        artifacts=artifacts,
        final_step=final_step,
        final_train_return=final_train_return,
        final_eval_return=final_eval_return,
        dvc_meta_committed=dvc_meta_committed,
    )


__all__ = ["ArtifactStatus", "RunSummary", "build_summary"]
