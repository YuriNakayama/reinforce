"""Kaggle Kernel の output (``/kaggle/working/``) をローカル run_dir に取り込む。

flow:
    1. ``kaggle kernels output`` で tmp_dir に download (pagination 対応)
    2. ``runs/<run_id>/`` 配下を run_dir にコピー (cell E が保存した artifact)
    3. ``dvc add`` で DVC に登録

注意 (2026-05-21): Kaggle SDK ``kernels_output`` は ``next_page_token`` を
返すが、SDK 内では追加ページを取得しない (バグ)。出力ファイルが多い kernel
では ``best.*`` / ``checkpoint_latest.*`` 等が漏れるため、本モジュールは
``file_pattern`` で必要な artifact のみに絞って download する戦略を取る。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class _KaggleApiLike(Protocol):
    def authenticate(self) -> None: ...

    def kernels_output_cli(
        self,
        kernel: str,
        kernel_opt: str | None = None,
        path: str | None = None,
        force: bool = False,
        quiet: bool = False,
        file_pattern: str | None = None,
    ) -> object: ...


def pull_kernel_output(
    api: _KaggleApiLike,
    slug: str,
    tmp_dir: Path,
    *,
    force: bool = True,
    file_pattern: str | None = None,
) -> Path:
    """``/kaggle/working/`` の中身を tmp_dir にダウンロードする。

    ``file_pattern`` を渡すと SDK の regex filter (re.search) で絞り込む。
    既定では絞らず全 file 取得 (1 page 上限の制約あり)。
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    api.kernels_output_cli(
        kernel=slug,
        path=str(tmp_dir),
        force=force,
        quiet=False,
        file_pattern=file_pattern,
    )
    return tmp_dir


def place_into_run_dir(tmp_dir: Path, run_dir: Path, *, run_id: str) -> Path:
    """tmp_dir 内の ``runs/<run_id>/*`` を run_dir にコピーする。

    Kaggle Kernel が cell E で ``/kaggle/working/runs/<run_id>/`` に集約しているため、
    そこを最優先で探す。見つからない場合は tmp_dir 直下を fallback で取り込む。
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    nested = tmp_dir / "runs" / run_id
    src = nested if nested.is_dir() else tmp_dir
    copied = 0
    for entry in src.iterdir():
        target = run_dir / entry.name
        if entry.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)
        copied += 1
    logger.info("placed %d entries into %s (src=%s)", copied, run_dir, src)
    return run_dir


def dvc_add(run_dir: Path, repo_root: Path) -> None:
    """``dvc add <run_dir>`` を repo_root で実行する。"""
    subprocess.run(
        ["dvc", "add", str(run_dir)],
        cwd=str(repo_root),
        check=True,
    )
