"""backend/ から Kaggle Dataset 用 snapshot ディレクトリを作る。

除外規則:
    - data/, .venv/, __pycache__/, *.pyc, .dvc/, .git/, docs/, infra/,
      backend/tests/, node_modules/

包含:
    - backend/src/, backend/pipeline/, backend/pyproject.toml, backend/uv.lock,
      backend/.python-version
    - include_wheels で渡された .whl は <dest>/wheels/ に配置
    - include_data_files で渡された data/... 配下のファイルは
      ``kdata__<a>__<b>__<file>`` の top-level ファイルに flatten して配置
      (kernel 側 cell B が data/<a>/<b>/<file> に復元する)
    - Kaggle 側の repo root 判定 (backend/pyproject.toml + .git) のため
      空の <dest>/.git/HEAD を作成
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

EXCLUDE_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        ".dvc",
        "__pycache__",
        "node_modules",
        "data",
        "docs",
        "infra",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "target",
    }
)

INCLUDE_RELATIVE_PATHS: tuple[str, ...] = (
    "backend/src",
    "backend/pipeline",
    "backend/pyproject.toml",
    "backend/uv.lock",
    "backend/.python-version",
)

#: Prefix for flattened ``data/...`` files (see ``build_snapshot``).
DATA_FLAT_PREFIX = "kdata__"
_DATA_FLAT_SEP = "__"


def build_snapshot(
    repo_root: Path,
    dest_dir: Path,
    *,
    include_wheels: Iterable[Path] | None = None,
    include_data_files: Iterable[Path] | None = None,
    include_dotenv: bool = False,
) -> Path:
    """``repo_root`` から ``dest_dir`` にコード snapshot を作成する。

    ``dest_dir`` が既存なら中身を削除して作り直す。
    ``include_wheels`` で渡された .whl は ``<dest_dir>/wheels/`` にコピー。
    ``include_data_files`` で渡された ``data/...`` 配下のファイルは
    ``kdata__<a>__<b>__<file>`` に flatten して ``dest_dir`` 直下に置く。
    Kaggle 側が ``backend/pyproject.toml`` + ``.git`` のペアで repo root を
    認識するため、空の ``<dest_dir>/.git/HEAD`` を必ず作成する。
    ``include_dotenv`` を True にすると ``backend/.env`` を snapshot に同梱する
    (interactive mode の AWS creds 配送用、private dataset 前提)。
    戻り値は ``dest_dir`` (絶対 path)。
    """
    repo_root = repo_root.resolve()
    dest_dir = dest_dir.resolve()

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)

    for rel in INCLUDE_RELATIVE_PATHS:
        src = repo_root / rel
        if not src.exists():
            logger.debug("skip absent path: %s", src)
            continue
        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, target)
        else:
            shutil.copytree(
                src, target, ignore=shutil.ignore_patterns(*_ignore_patterns())
            )

    if include_wheels:
        wheels_dir = dest_dir / "wheels"
        wheels_dir.mkdir(parents=True, exist_ok=True)
        for wheel in include_wheels:
            wheel = wheel.resolve()
            if not wheel.is_file():
                raise FileNotFoundError(f"wheel not found: {wheel}")
            if wheel.suffix != ".whl":
                raise ValueError(f"not a wheel file: {wheel}")
            shutil.copy2(wheel, wheels_dir / wheel.name)

    if include_data_files:
        # Kaggle dataset の zip mode は top-level dir を <name>.zip にして
        # アップロードする。`data.zip` のような top-level zip は Upload
        # Successful と表示されても処理後の dataset には含まれないという挙動を
        # 観測した。回避策: data/ 配下のファイルを flat な top-level ファイル
        # (`kdata__<a>__<b>__<file>`) として配置し、kernel 側で復元する。
        for src in include_data_files:
            _copy_data_file(repo_root, dest_dir, src)

    git_dir = dest_dir / ".git"
    git_dir.mkdir(exist_ok=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/kaggle-snapshot\n", encoding="utf-8")

    if include_dotenv:
        env_src = repo_root / "backend" / ".env"
        if not env_src.is_file():
            raise FileNotFoundError(
                f"include_dotenv requested but {env_src} does not exist"
            )
        env_dst = dest_dir / "backend" / ".env"
        env_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(env_src, env_dst)

    return dest_dir


def _copy_data_file(repo_root: Path, dest_dir: Path, src: Path) -> None:
    """``data/<a>/<b>/<file>`` を ``<dest_dir>/kdata__<a>__<b>__<file>`` に置く。"""
    src_norm = Path(os.path.normpath(str(src.absolute())))
    src_real = src.resolve()
    if src_norm.is_file():
        src_for_copy = src_norm
    elif src_real.is_file():
        src_for_copy = src_real
    else:
        raise FileNotFoundError(f"data file not found at {src_norm} (real: {src_real})")
    try:
        rel_path = src_norm.relative_to(repo_root)
    except ValueError:
        parts = src_norm.parts
        if "data" not in parts:
            raise
        rel_path = Path(*parts[parts.index("data") :])
    if rel_path.parts[0] != "data" or len(rel_path.parts) < 2:
        raise ValueError(f"data file must live under data/: {rel_path}")
    inner = rel_path.parts[1:]
    if any(_DATA_FLAT_SEP in part for part in inner):
        raise ValueError(
            f"path component contains {_DATA_FLAT_SEP!r}, cannot flatten: {rel_path}"
        )
    target = dest_dir / (DATA_FLAT_PREFIX + _DATA_FLAT_SEP.join(inner))
    shutil.copy2(src_for_copy, target)


def _ignore_patterns() -> tuple[str, ...]:
    return tuple(EXCLUDE_DIR_NAMES) + ("*.pyc", "*.pyo")
