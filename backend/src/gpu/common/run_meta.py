"""Run metadata model + run_id 生成 + case config hash + atomic JSON I/O.

`<run_dir>/run.json` は学習 1 run の lineage を完全記録する正本ファイル。
schema_version=1 で今後の互換性維持を予定。RunPod / Vast.ai / Kaggle の全基盤が
この 1 モジュールを共有する (provider 固有フィールドは Optional)。

`params_hash` は case が学習に使う config ファイル
(`backend/pipeline/<family>/<case>/config.yaml`、または registry の
`--config <path>`) の内容ハッシュ。解決は `gpu.runpod.config.cases` に委ねる。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from gpu.runpod.config.cases import case_config_path

SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^\d{8}-\d{6}__[A-Za-z0-9_-]+__[0-9a-f]{7}__seed\d+$")
_SLUG_INVALID = re.compile(r"[^A-Za-z0-9_-]+")

# 'running' / 'pushed' / 'evaluated' / 'adopted' / 'failed'
RunStatus = str


@dataclass
class RunMetadata:
    """1 学習 run の完全 lineage。"""

    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    git_sha: str = ""
    git_branch: str = ""
    params_hash: str = ""
    seed: int = 0
    vast_instance_id: int | None = None
    runpod_pod_id: str | None = None
    gpu_name: str | None = None
    vast_offer_snapshot: dict[str, Any] | None = None
    runpod_offer_snapshot: dict[str, Any] | None = None
    kaggle_kernel_meta: dict[str, Any] | None = None
    command: str = ""
    weights_path: str = ""
    train_metrics: dict[str, Any] = field(default_factory=dict)
    local_eval_results: dict[str, Any] | None = None
    status: RunStatus = "running"
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _branch_slug(branch: str) -> str:
    """branch 名を run_id 内で使える文字種にスラッグ化する。

    `feature/gpu-basis` -> `feature-gpu-basis`。
    連続する不正文字は 1 個の `-` に潰し、先頭末尾の `-` は除去する。
    """
    slug = _SLUG_INVALID.sub("-", branch).strip("-")
    return slug or "unknown"


def _utc_now_iso(now: datetime | None = None) -> str:
    """ISO8601 (秒精度) UTC タイムスタンプ。"""
    if now is None:
        now = datetime.now(UTC)
    return now.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def generate_run_id(
    branch: str,
    sha: str,
    seed: int,
    *,
    now: datetime | None = None,
) -> str:
    """`<YYYYMMDD-HHMMSS>__<branch_slug>__<sha7>__seed<N>` を生成する。

    `now` を渡せば deterministic にできる（テスト用）。
    """
    if not sha or len(sha) < 7:
        raise ValueError(f"git sha must be >= 7 chars: {sha!r}")
    if now is None:
        now = datetime.now(UTC)
    ts = now.strftime("%Y%m%d-%H%M%S")
    slug = _branch_slug(branch)
    sha7 = sha[:7].lower()
    return f"{ts}__{slug}__{sha7}__seed{seed}"


def hash_params(config_path: Path) -> str:
    """config YAML の内容を canonical JSON に変換し sha256 の先頭 12 文字を返す。

    dict キー順序に依存しない。`yaml.safe_load → json.dumps(sort_keys=True)` で正規化。
    ファイルが存在しない (case がまだ config.yaml を持たない) 場合は "" を返す。
    """
    if not config_path.is_file():
        return ""
    raw = config_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    canonical = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:12]


def hash_case_config(repo_root: Path, case: str) -> str:
    """registry key から case の config ファイルを解決して `hash_params` する。"""
    return hash_params(repo_root / case_config_path(case))


def _atomic_write(path: Path, payload: str) -> None:
    """tmp file → rename で atomic に書き込む。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _serialize(meta: RunMetadata) -> str:
    return json.dumps(meta.to_dict(), ensure_ascii=False, indent=2) + "\n"


def write_run_json(run_dir: Path, meta: RunMetadata) -> Path:
    """`<run_dir>/run.json` に atomic 書き込み。

    created_at / updated_at が空なら現在時刻で埋める。
    """
    if not meta.created_at:
        meta.created_at = _utc_now_iso()
    meta.updated_at = _utc_now_iso()
    out = run_dir / "run.json"
    _atomic_write(out, _serialize(meta))
    return out


def read_run_json(run_dir: Path) -> RunMetadata:
    """`<run_dir>/run.json` を読み RunMetadata に復元する。"""
    raw = (run_dir / "run.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    return RunMetadata(**data)


def update_run_json(run_dir: Path, **patch: Any) -> RunMetadata:
    """既存 run.json を読み込み、patch を適用して書き戻す。

    `updated_at` は常に上書き、`created_at` は維持する。
    """
    meta = read_run_json(run_dir)
    for key, value in patch.items():
        if not hasattr(meta, key):
            raise AttributeError(f"RunMetadata has no field {key!r}")
        setattr(meta, key, value)
    meta.updated_at = _utc_now_iso()
    _atomic_write(run_dir / "run.json", _serialize(meta))
    return meta
