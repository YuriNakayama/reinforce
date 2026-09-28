"""Kaggle Datasets CRUD の thin wrapper (KaggleApi 経由)。

SDK ``kaggle`` の遅延 import で circular import / 認証エラーの伝播を防ぐ。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from gpu.kaggle.dataset.metadata import (
    build_version_notes,
    parse_commit_from_version_notes,
)

logger = logging.getLogger(__name__)


class _KaggleApiLike(Protocol):
    """KaggleApi の最小 surface (テストで mock するため)。"""

    def authenticate(self) -> None: ...

    def dataset_create_new(
        self,
        folder: str,
        public: bool,
        quiet: bool,
        convert_to_csv: bool,
        dir_mode: str,
    ) -> object: ...

    def dataset_create_version(
        self,
        folder: str,
        version_notes: str,
        quiet: bool,
        convert_to_csv: bool,
        dir_mode: str,
    ) -> object: ...

    def dataset_status(self, dataset: str) -> object: ...

    def dataset_list_files(self, dataset: str) -> object: ...


@dataclass(frozen=True)
class DatasetVersionResult:
    """dataset_create_new / dataset_create_version の戻り値正規化。"""

    slug: str
    version_notes: str
    commit_sha: str | None
    raw: object


def build_api() -> Any:
    """KaggleApi インスタンスを作って authenticate を済ませて返す。

    SDK は遅延 import (``kaggle`` 認証は import 時に env を読むため、
    auth.load_kaggle_creds で既に環境を整えた後に呼ぶ前提)。
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def create_new_dataset(
    api: _KaggleApiLike,
    dataset_dir: Path,
    *,
    commit_sha: str,
    public: bool = False,
) -> DatasetVersionResult:
    """``dataset_create_new`` を呼んで初回 dataset を作成する。"""
    notes = build_version_notes(commit_sha)
    logger.info("creating new dataset from %s (commit=%s)", dataset_dir, commit_sha[:7])
    raw = api.dataset_create_new(
        folder=str(dataset_dir),
        public=public,
        quiet=False,
        convert_to_csv=False,
        dir_mode="zip",
    )
    slug = _slug_from_dataset_dir(dataset_dir)
    return DatasetVersionResult(
        slug=slug,
        version_notes=notes,
        commit_sha=commit_sha[:7],
        raw=raw,
    )


def push_dataset_version(
    api: _KaggleApiLike,
    dataset_dir: Path,
    *,
    commit_sha: str,
    label: str | None = None,
) -> DatasetVersionResult:
    """既存 dataset の新 version を push する。"""
    notes = build_version_notes(commit_sha, label=label)
    logger.info(
        "pushing dataset version from %s (commit=%s, label=%s)",
        dataset_dir,
        commit_sha[:7],
        label,
    )
    raw = api.dataset_create_version(
        folder=str(dataset_dir),
        version_notes=notes,
        quiet=False,
        convert_to_csv=False,
        dir_mode="zip",
    )
    slug = _slug_from_dataset_dir(dataset_dir)
    return DatasetVersionResult(
        slug=slug,
        version_notes=notes,
        commit_sha=commit_sha[:7],
        raw=raw,
    )


def dataset_status(api: _KaggleApiLike, slug: str) -> dict[str, object]:
    """``dataset_status`` の戻り値を dict 化して返す。"""
    raw = api.dataset_status(slug)
    if isinstance(raw, dict):
        return dict(raw)
    return {"status": str(raw)}


def latest_version_commit(api: _KaggleApiLike, slug: str) -> str | None:
    """最新 version の version_notes から commit SHA を抽出する。

    Kaggle API は dataset の version 履歴を直接列挙する method が無いので、
    ``dataset_status`` 経由で取れた notes から parse する。取れない場合は None。
    """
    raw = dataset_status(api, slug)
    notes = raw.get("versionNotes") or raw.get("description") or ""
    if isinstance(notes, str):
        return parse_commit_from_version_notes(notes)
    return None


def _slug_from_dataset_dir(dataset_dir: Path) -> str:
    """dataset-metadata.json から id を読む。"""
    import json

    meta_path = dataset_dir / "dataset-metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"dataset-metadata.json not found under {dataset_dir}. "
            "Run write_dataset_metadata() before pushing."
        )
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    slug = data.get("id")
    if not isinstance(slug, str):
        raise ValueError(f"id missing or not a string in {meta_path}")
    return slug
