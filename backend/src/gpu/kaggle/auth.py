"""Kaggle API credentials のローカル読み込みヘルパ。

3 段 fallback (env → backend/.env → ~/.kaggle/kaggle.json) で
``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` を解決する。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from gpu.vast.auth import CredentialsError

__all__ = [
    "CredentialsError",
    "KaggleCreds",
    "load_kaggle_creds",
]


@dataclass(frozen=True)
class KaggleCreds:
    """Kaggle API 認証情報。"""

    username: str
    key: str


def load_kaggle_creds(
    *,
    env_path: Path | None = None,
    kaggle_json_path: Path | None = None,
) -> KaggleCreds:
    """KAGGLE_USERNAME と KAGGLE_KEY を 3 段 fallback で解決する。

    1. process env (``KAGGLE_USERNAME`` + ``KAGGLE_KEY``)
    2. ``backend/.env`` (env_path で override 可)
    3. ``~/.kaggle/kaggle.json`` (kaggle_json_path で override 可)

    どれにも値が無い場合は ``CredentialsError`` を raise する。
    """
    username, key = _from_process_env()
    if username and key:
        return KaggleCreds(username=username, key=key)

    if env_path is None:
        env_path = _default_env_path()
    if env_path.is_file():
        values = dotenv_values(env_path)
        e_user = (values.get("KAGGLE_USERNAME") or "").strip()
        e_key = (values.get("KAGGLE_KEY") or "").strip()
        if e_user and e_key:
            return KaggleCreds(username=e_user, key=e_key)

    if kaggle_json_path is None:
        kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json_path.is_file():
        try:
            data = json.loads(kaggle_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise CredentialsError(
                f"Failed to parse {kaggle_json_path} as JSON: {e}"
            ) from e
        j_user = (data.get("username") or "").strip()
        j_key = (data.get("key") or "").strip()
        if j_user and j_key:
            return KaggleCreds(username=j_user, key=j_key)

    raise CredentialsError(
        "Kaggle credentials not found. Provide them via one of:\n"
        "  1. Environment: export KAGGLE_USERNAME=<name>; export KAGGLE_KEY=<key>\n"
        f"  2. {env_path}: KAGGLE_USERNAME=<name>\\nKAGGLE_KEY=<key>\n"
        f'  3. {kaggle_json_path}: {{"username": ..., "key": ...}}\n'
        "Generate a key at https://www.kaggle.com/settings -> "
        "'Create New API Token'."
    )


def _from_process_env() -> tuple[str, str]:
    return (
        os.environ.get("KAGGLE_USERNAME", "").strip(),
        os.environ.get("KAGGLE_KEY", "").strip(),
    )


def _default_env_path() -> Path:
    """backend/ 直下の .env を返す。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / ".env"
    return Path(".env").resolve()
