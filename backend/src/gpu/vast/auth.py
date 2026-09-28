"""AWS credentials と VAST_API_KEY のローカル読み込みヘルパ。

AWS は `~/.aws/credentials` の `reinforce` profile を `aws configure get` 経由で読み、
Vast の API key は `backend/.env` か環境変数から読む。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

DEFAULT_AWS_PROFILE = "reinforce"
DEFAULT_AWS_REGION = "ap-northeast-1"


class CredentialsError(RuntimeError):
    """credentials の取得に失敗した場合に投げる。actionable なメッセージを保持する。"""


@dataclass(frozen=True)
class AwsCreds:
    access_key_id: str
    secret_access_key: str
    region: str = DEFAULT_AWS_REGION


def _aws_get(profile: str, key: str) -> str:
    """`aws configure get <key> --profile <profile>` の出力を返す。"""
    aws_bin = shutil.which("aws")
    if aws_bin is None:
        raise CredentialsError(
            "aws CLI not found in PATH. Install AWS CLI v2 to read credentials."
        )
    try:
        result = subprocess.run(
            [aws_bin, "configure", "get", key, "--profile", profile],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise CredentialsError(
            f"aws configure get {key} --profile {profile} failed. "
            "Run `dev/dvc setup` and confirm `~/.aws/credentials` has "
            f"a [{profile}] entry."
        ) from exc
    value = result.stdout.strip()
    if not value:
        raise CredentialsError(
            f"aws profile {profile!r} has no value for {key!r}. "
            f"Add it to ~/.aws/credentials under [{profile}]."
        )
    return value


def load_aws_creds(
    *,
    profile: str = DEFAULT_AWS_PROFILE,
    region: str | None = None,
) -> AwsCreds:
    """`aws configure get` で access_key_id / secret_access_key を取得する。"""
    access_key = _aws_get(profile, "aws_access_key_id")
    secret_key = _aws_get(profile, "aws_secret_access_key")
    resolved_region = region or _try_aws_get_region(profile) or DEFAULT_AWS_REGION
    return AwsCreds(
        access_key_id=access_key,
        secret_access_key=secret_key,
        region=resolved_region,
    )


def _try_aws_get_region(profile: str) -> str | None:
    """region は無くてもデフォルトでよいので、失敗を握り潰す。"""
    try:
        return _aws_get(profile, "region")
    except CredentialsError:
        return None


def load_vast_api_key(
    *,
    env_path: Path | None = None,
) -> str:
    """`backend/.env` または環境変数から VAST_API_KEY を読む。

    env file → process env の順に検査し、最初に値があれば返す。
    """
    if env_path is None:
        env_path = _default_env_path()
    if env_path.is_file():
        values = dotenv_values(env_path)
        api_key = values.get("VAST_API_KEY")
        if api_key:
            return api_key.strip()
    fallback = os.environ.get("VAST_API_KEY", "").strip()
    if fallback:
        return fallback
    raise CredentialsError(
        "VAST_API_KEY not found. Add `VAST_API_KEY=<your-key>` to "
        f"{env_path} or export it as an environment variable. "
        "Get a key from https://cloud.vast.ai/manage-keys/."
    )


def _default_env_path() -> Path:
    """backend/ 直下の .env を返す。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / ".env"
    return Path(".env").resolve()
