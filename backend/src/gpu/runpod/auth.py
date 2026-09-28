"""AWS credentials と RUNPOD_API_KEY のローカル読み込みヘルパ。

AWS は gpu.common.auth の `load_aws_creds()` をそのまま再利用する (重複実装を避ける)。
RunPod 用 `load_runpod_api_key()` のみ追加する。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

from gpu.common.auth import (
    DEFAULT_AWS_PROFILE,
    DEFAULT_AWS_REGION,
    AwsCreds,
    CredentialsError,
    load_aws_creds,
    load_dvc_bucket,
)

# `.env.example` names the key RUNPOD_KEY; RUNPOD_API_KEY is what the RunPod SDK
# itself uses. Both are accepted, RUNPOD_API_KEY first.
_RUNPOD_KEY_NAMES = ("RUNPOD_API_KEY", "RUNPOD_KEY")

__all__ = [
    "DEFAULT_AWS_PROFILE",
    "DEFAULT_AWS_REGION",
    "AwsCreds",
    "CredentialsError",
    "load_aws_creds",
    "load_dvc_bucket",
    "load_git_pat",
    "load_runpod_api_key",
]


def load_runpod_api_key(
    *,
    env_path: Path | None = None,
) -> str:
    """`backend/.env` または環境変数から RunPod API key を読む。

    env file → process env の順に、`RUNPOD_API_KEY` / `RUNPOD_KEY` を検査し、
    最初に値があれば返す。
    """
    if env_path is None:
        env_path = _default_env_path()
    if env_path.is_file():
        values = dotenv_values(env_path)
        for name in _RUNPOD_KEY_NAMES:
            api_key = values.get(name)
            if api_key and api_key.strip():
                return api_key.strip()
    for name in _RUNPOD_KEY_NAMES:
        fallback = os.environ.get(name, "").strip()
        if fallback:
            return fallback
    raise CredentialsError(
        "RunPod API key not found. Add `RUNPOD_API_KEY=<your-key>` to "
        f"{env_path} or export it as an environment variable. "
        "Get a key from https://runpod.io/console/user/settings."
    )


def load_git_pat(*, env_path: Path | None = None) -> str | None:
    """`backend/.env` または環境変数から GIT_PAT (GitHub Personal Access Token)
    を読む。Pod 内で `git push origin <branch>` する際の auth に使う。

    env file → process env の順に検査。値が無くても例外は投げず None を返す。
    PAT が無い場合 onstart の git push (`.dvc` メタの永続化) のみ skip される。
    """
    if env_path is None:
        env_path = _default_env_path()
    if env_path.is_file():
        values = dotenv_values(env_path)
        pat = values.get("GIT_PAT")
        if pat:
            return pat.strip()
    fallback = os.environ.get("GIT_PAT", "").strip()
    return fallback or None


def _default_env_path() -> Path:
    """backend/ 直下の .env を返す。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / ".env"
    return Path(".env").resolve()
