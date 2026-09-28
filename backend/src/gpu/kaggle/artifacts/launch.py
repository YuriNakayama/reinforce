"""launch.json (per-run launch metadata) の write/read。

``dev/kaggle train`` が起動時に書く正本。後の ``pull`` / ``status`` /
``promote`` から参照する。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LAUNCH_JSON_FILENAME = "launch.json"


@dataclass(frozen=True)
class LaunchMeta:
    """1 kaggle-kernel run の launch メタ。"""

    run_id: str
    case: str
    commit_sha: str
    branch: str
    kernel_slug: str
    kernel_version: int | None
    dataset_slug: str
    dataset_version: str | None
    accelerator: str
    enable_internet: bool
    started_at: str
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def write_launch_json(run_dir: Path, meta: LaunchMeta) -> Path:
    """``<run_dir>/launch.json`` を atomic に書く。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / LAUNCH_JSON_FILENAME
    payload = json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n"
    _atomic_write(out, payload)
    return out


def read_launch_json(run_dir: Path) -> LaunchMeta:
    """``<run_dir>/launch.json`` を読み LaunchMeta に復元する。"""
    raw = (run_dir / LAUNCH_JSON_FILENAME).read_text(encoding="utf-8")
    data = json.loads(raw)
    return LaunchMeta(**data)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
