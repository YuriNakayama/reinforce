"""`runs/<run_id>/launch.json` の読み書き。

`train` サブコマンドが pod を作成した直後に書き出すローカル専用 metadata。
onstart 内で生成される `run.json` (DVC 管理対象) とは別ファイルで、目的は
`run_id -> pod_id` を後から逆引きして `dev/runpod ps/status/logs` で使うこと。

DVC/Git 管理対象外。`runs/<run_id>` ディレクトリ自体は train 直後にローカル
作成するが、`run.json` が onstart で生成されるまでは launch.json のみ存在する。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu.runpod.config.cases import runs_root_for

LAUNCH_FILENAME = "launch.json"


def write_launch_json(
    run_dir: Path,
    *,
    run_id: str,
    pod_id: str,
    commit_sha: str,
    branch: str,
    case: str,
    cloud_type: str,
    gpu_type_id: str,
    dph_total: float,
    data_center_id: str | None = None,
    mode: str = "oneshot",
) -> Path:
    """train 直後にローカル launch metadata を書き出す。

    Args:
        mode: "oneshot" (auto-terminate あり) / "interactive" (sleep infinity 保持)。
            `dev/runpod ps` / `destroy` 系で参照する。
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "pod_id": pod_id,
        "commit_sha": commit_sha,
        "branch": branch,
        "case": case,
        "cloud_type": cloud_type,
        "gpu_type_id": gpu_type_id,
        "dph_total": dph_total,
        "data_center_id": data_center_id,
        "mode": mode,
        "launched_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = run_dir / LAUNCH_FILENAME
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_launch_json(run_dir: Path) -> dict[str, Any]:
    """launch.json を読む。存在しない場合 FileNotFoundError。"""
    path = run_dir / LAUNCH_FILENAME
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def find_run_dir(repo_root: Path, run_id: str, case: str) -> Path:
    """case ディレクトリ配下の run_dir パスを返す。存在チェックは呼び出し側。

    `runs_root_for(case)` を介して family (ppo / dqn / ...) を
    解決する。reinforce_case1 のような新 family のキーも追加すれば自動的に
    `data/output/models/<family>/<subdir>/runs/<run_id>` に解決される。
    """
    return repo_root / runs_root_for(case) / run_id
