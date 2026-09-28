"""ローカルと RunPod pod の間で backend/ ディレクトリを rsync する。

`dev/runpod dev` で起動した interactive pod 上でコードを試行錯誤する用途。
ローカル IDE で編集 → `dev/runpod sync <run_id> --push` → pod 上で再実行、と
いうループを想定する。逆方向 (`--pull`) は pod 上で生成された artifact を
ローカルに取り出すときに使う。

rsync の `-e` オプションに ssh コマンドを渡すため、`SshEndpoint` から
ssh 引数列を組み立て、`shlex.join` でクォートして 1 文字列に直す。
direct / proxy どちらの endpoint kind でも動作する。
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from gpu.runpod.runpod.ssh import SshEndpoint

# pod 側で git clone される標準パス (onstart_lib/30_persist_and_clone.sh 参照)。
DEFAULT_REMOTE_REPO_ROOT = "/workspace/reinforce"
DEFAULT_REMOTE_BACKEND_DIR = f"{DEFAULT_REMOTE_REPO_ROOT}/backend"

# ローカル → pod へ送らない (重い・無意味・破壊的) パス。
# .venv は pod 側 (Linux) と OS が違うため絶対に上書きしてはならない。
# data は DVC 管理。__pycache__/*.pyc は実行時に再生成。
DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".venv/",
    ".venv.*/",
    "data/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "*.egg-info/",
    ".DS_Store",
)


@dataclass(frozen=True)
class SyncPlan:
    """rsync 1 回分の起動引数。`subprocess.run(cmd)` でそのまま実行できる。"""

    cmd: list[str]
    src: str
    dst: str
    direction: str  # "push" or "pull"

    def describe(self) -> str:
        return f"{self.direction}: {self.src} -> {self.dst}"


def _build_rsh(endpoint: SshEndpoint) -> str:
    """rsync の -e に渡せる ssh コマンド文字列を生成する。

    SshEndpoint.to_command(None) は末尾に `user@host` (direct) または
    `<pod_id>@ssh.runpod.io` (proxy) を含むが、rsync の -e は ssh の
    *接続オプションのみ* を期待する (接続先は rsync が SOURCE/DEST から推測)。
    そのため末尾の host 部分を取り除く。
    """
    full = endpoint.to_command()
    # 末尾の `[user@]host` を除去する。直前に "-p" がある場合 (direct) を考慮済。
    return shlex.join(full[:-1])


def _remote_target(endpoint: SshEndpoint, remote_path: str) -> str:
    """rsync の `<[user@]host>:<path>` 形式を組む。

    - direct: `endpoint.host` は IP のみなので `user@host:path` で連結する
      (rsync は user 省略時にローカルログインユーザーで接続しようとしてしまう)。
    - proxy:  `endpoint.host` は既に `<pod_id>@ssh.runpod.io` 形式なので
      そのまま `host:path` で OK (user 部分は host に含まれる)。
    """
    if endpoint.kind == "direct":
        return f"{endpoint.user}@{endpoint.host}:{remote_path}"
    return f"{endpoint.host}:{remote_path}"


def build_sync_plan(
    *,
    endpoint: SshEndpoint,
    local_backend_dir: Path,
    direction: str,
    remote_backend_dir: str = DEFAULT_REMOTE_BACKEND_DIR,
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
    dry_run: bool = False,
    delete: bool = False,
) -> SyncPlan:
    """rsync 起動引数を組み立てる (実行はしない)。

    Args:
        endpoint: 接続先 SSH endpoint (direct or proxy)。
        local_backend_dir: ローカルの `backend/` 絶対パス。末尾 `/` 付きとして扱う
            (rsync の慣例で、ディレクトリ内容を転送する)。
        direction: "push" (local → pod) または "pull" (pod → local)。
        remote_backend_dir: pod 上の backend ディレクトリ絶対パス。
        excludes: rsync `--exclude` に渡すパターン群。
        dry_run: True なら `--dry-run` を付ける。
        delete: True なら `--delete` を付ける (rsync 既定は累積コピー)。
    """
    if direction not in ("push", "pull"):
        raise ValueError(f"direction must be push/pull, got {direction!r}")
    if not local_backend_dir.is_absolute():
        raise ValueError(f"local_backend_dir must be absolute: {local_backend_dir}")

    # rsync は末尾 `/` の有無で挙動が変わる。ディレクトリ内容を転送するため
    # 必ず末尾 `/` 付きで src を組む。
    local_str = str(local_backend_dir).rstrip("/") + "/"
    remote_str = remote_backend_dir.rstrip("/") + "/"
    remote_full = _remote_target(endpoint, remote_str)

    if direction == "push":
        src, dst = local_str, remote_full
    else:
        src, dst = remote_full, local_str

    cmd: list[str] = [
        "rsync",
        "-avz",
        "--human-readable",
        "--itemize-changes",
        "-e",
        _build_rsh(endpoint),
    ]
    for pattern in excludes:
        cmd.extend(["--exclude", pattern])
    if dry_run:
        cmd.append("--dry-run")
    if delete:
        cmd.append("--delete")
    cmd.extend([src, dst])

    return SyncPlan(cmd=cmd, src=src, dst=dst, direction=direction)


__all__ = [
    "DEFAULT_EXCLUDES",
    "DEFAULT_REMOTE_BACKEND_DIR",
    "DEFAULT_REMOTE_REPO_ROOT",
    "SyncPlan",
    "build_sync_plan",
]
