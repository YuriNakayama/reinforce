"""Kaggle Dataset の dataset-metadata.json を生成する。"""

from __future__ import annotations

import json
import re
from pathlib import Path

DATASET_METADATA_FILENAME = "dataset-metadata.json"

_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*$")


def write_dataset_metadata(
    dest_dir: Path,
    *,
    slug: str,
    title: str,
    commit_sha: str,
    is_private: bool = True,
    license_name: str = "Apache-2.0",
    subtitle: str | None = None,
) -> Path:
    """``<dest_dir>/dataset-metadata.json`` を書き出す。"""
    if not _SLUG_RE.match(slug):
        raise ValueError(
            f"invalid Kaggle dataset slug {slug!r}: expected '<owner>/<name>'"
        )
    payload = {
        "id": slug,
        "title": title,
        "licenses": [{"name": license_name}],
        "isPrivate": is_private,
        "keywords": ["reinforce", "snapshot", f"commit={commit_sha[:7]}"],
    }
    if subtitle:
        payload["subtitle"] = subtitle
    out = dest_dir / DATASET_METADATA_FILENAME
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def build_version_notes(commit_sha: str, label: str | None = None) -> str:
    """`dataset_create_version` の version_notes 文字列を生成する。

    形式: ``commit=<sha7>`` または ``commit=<sha7> | <label>``。
    後で ``latest_version_commit`` で逆解析する。
    """
    if not commit_sha or len(commit_sha) < 7:
        raise ValueError(f"commit_sha too short: {commit_sha!r}")
    base = f"commit={commit_sha[:7]}"
    if label:
        return f"{base} | {label}"
    return base


_COMMIT_NOTE_RE = re.compile(r"commit=([0-9a-fA-F]{7,40})")


def parse_commit_from_version_notes(notes: str) -> str | None:
    """``commit=abc1234`` から commit SHA を抽出する。"""
    m = _COMMIT_NOTE_RE.search(notes or "")
    if not m:
        return None
    return m.group(1).lower()
