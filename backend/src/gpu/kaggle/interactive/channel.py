"""S3 command channel for Kaggle Kernel interactive mode.

Layout under ``s3://<bucket>/kaggle_interactive/<run_id>/``:

    inbox/<uuid>.json    Claude が put したコマンド (kernel が拾って実行)
    outbox/<uuid>.json   kernel が put した結果 (Claude が読む)
    state/heartbeat.json kernel が ~10s 間隔で put する生存信号
    state/files/<path>   sync 用ファイル転送のステージング

コマンド形式 (inbox):
    {
      "argv": ["python", "-c", "print('hi')"],
      "cwd": "/tmp/reinforce-repo/backend",
      "timeout": 300,
      "env": {"FOO": "bar"}
    }

結果形式 (outbox):
    {
      "returncode": 0,
      "stdout": "...",
      "stderr": "...",
      "elapsed_seconds": 1.23,
      "started_at": "2026-05-21T...Z",
      "finished_at": "2026-05-21T...Z"
    }
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from gpu.common.auth import load_dvc_bucket

logger = logging.getLogger(__name__)

# Same bucket as the DVC remote (REINFORCE_DVC_BUCKET from backend/.env or the
# environment); "" when unset — pass `--s3-bucket` explicitly in that case.
DEFAULT_BUCKET = load_dvc_bucket(required=False)
PREFIX_ROOT = "kaggle_interactive"
STDOUT_CAP = 1_000_000  # 1MB cap per side
STDERR_CAP = 1_000_000


class _S3ClientLike(Protocol):
    """boto3 S3 client の最小 surface (テストで mock するため)。"""

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> object: ...

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]: ...

    def delete_object(self, *, Bucket: str, Key: str) -> object: ...

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str, MaxKeys: int = ...
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Command:
    """inbox に投入するコマンド。"""

    argv: list[str]
    cwd: str | None = None
    timeout: float = 300.0
    env: dict[str, str] = field(default_factory=dict)
    command_id: str = ""

    def with_id(self, command_id: str) -> Command:
        return Command(
            argv=self.argv,
            cwd=self.cwd,
            timeout=self.timeout,
            env=self.env,
            command_id=command_id,
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "argv": self.argv,
                "cwd": self.cwd,
                "timeout": self.timeout,
                "env": self.env,
                "command_id": self.command_id,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> Command:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        return cls(
            argv=list(data["argv"]),
            cwd=data.get("cwd"),
            timeout=float(data.get("timeout", 300.0)),
            env=dict(data.get("env") or {}),
            command_id=str(data.get("command_id", "")),
        )


@dataclass(frozen=True)
class CommandResult:
    """outbox に書き戻される実行結果。"""

    command_id: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "command_id": self.command_id,
                "returncode": self.returncode,
                "stdout": self.stdout[-STDOUT_CAP:],
                "stderr": self.stderr[-STDERR_CAP:],
                "elapsed_seconds": self.elapsed_seconds,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> CommandResult:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        return cls(
            command_id=str(data["command_id"]),
            returncode=int(data["returncode"]),
            stdout=str(data.get("stdout") or ""),
            stderr=str(data.get("stderr") or ""),
            elapsed_seconds=float(data.get("elapsed_seconds") or 0.0),
            started_at=str(data.get("started_at") or ""),
            finished_at=str(data.get("finished_at") or ""),
            error=data.get("error"),
        )


class ChannelTimeoutError(TimeoutError):
    """``submit_command`` が ``await_timeout`` 内に結果を取得できなかった。"""


def session_prefix(run_id: str) -> str:
    """``kaggle_interactive/<run_id>`` の prefix を返す。"""
    return f"{PREFIX_ROOT}/{run_id}"


def inbox_key(run_id: str, command_id: str) -> str:
    return f"{session_prefix(run_id)}/inbox/{command_id}.json"


def outbox_key(run_id: str, command_id: str) -> str:
    return f"{session_prefix(run_id)}/outbox/{command_id}.json"


def heartbeat_key(run_id: str) -> str:
    return f"{session_prefix(run_id)}/state/heartbeat.json"


def file_key(run_id: str, relpath: str) -> str:
    """sync 用の path 変換。``/`` 始まりは禁止。"""
    if relpath.startswith("/") or ".." in relpath.split("/"):
        raise ValueError(f"unsafe relpath: {relpath!r}")
    return f"{session_prefix(run_id)}/state/files/{relpath}"


def put_command(
    s3: _S3ClientLike,
    run_id: str,
    cmd: Command,
    *,
    bucket: str = DEFAULT_BUCKET,
) -> Command:
    """inbox にコマンドを投入する。``command_id`` が空なら uuid を割り当てる。"""
    command_id = cmd.command_id or uuid.uuid4().hex
    cmd_with_id = cmd.with_id(command_id)
    s3.put_object(
        Bucket=bucket,
        Key=inbox_key(run_id, command_id),
        Body=cmd_with_id.to_json().encode("utf-8"),
    )
    logger.info(
        "put command %s to s3://%s/%s",
        command_id,
        bucket,
        inbox_key(run_id, command_id),
    )
    return cmd_with_id


def try_fetch_result(
    s3: _S3ClientLike,
    run_id: str,
    command_id: str,
    *,
    bucket: str = DEFAULT_BUCKET,
) -> CommandResult | None:
    """outbox に結果があれば取得 + delete する。無ければ ``None``。"""
    try:
        obj = s3.get_object(Bucket=bucket, Key=outbox_key(run_id, command_id))
    except Exception as e:
        if _is_no_such_key(e):
            return None
        raise
    body = (
        obj["Body"].read() if hasattr(obj.get("Body"), "read") else obj.get("Body", b"")
    )
    result = CommandResult.from_json(body)
    s3.delete_object(Bucket=bucket, Key=outbox_key(run_id, command_id))
    return result


def await_result(
    s3: _S3ClientLike,
    run_id: str,
    command_id: str,
    *,
    bucket: str = DEFAULT_BUCKET,
    poll_interval: float = 2.0,
    timeout: float = 300.0,
    sleep: object = time.sleep,
) -> CommandResult:
    """outbox を polling して結果を取得する。``timeout`` 超過で例外。"""
    if poll_interval <= 0:
        raise ValueError(f"poll_interval must be positive: {poll_interval}")
    if timeout <= 0:
        raise ValueError(f"timeout must be positive: {timeout}")
    elapsed = 0.0
    while elapsed <= timeout:
        result = try_fetch_result(s3, run_id, command_id, bucket=bucket)
        if result is not None:
            return result
        sleep(poll_interval)  # type: ignore[operator]
        elapsed += poll_interval
    raise ChannelTimeoutError(
        f"no result for command {command_id} within {timeout}s "
        f"(s3://{bucket}/{outbox_key(run_id, command_id)})"
    )


def submit_command(
    s3: _S3ClientLike,
    run_id: str,
    cmd: Command,
    *,
    bucket: str = DEFAULT_BUCKET,
    poll_interval: float = 2.0,
    await_timeout: float = 300.0,
    sleep: object = time.sleep,
) -> CommandResult:
    """put + await を 1 ショットで行うヘルパ。"""
    submitted = put_command(s3, run_id, cmd, bucket=bucket)
    return await_result(
        s3,
        run_id,
        submitted.command_id,
        bucket=bucket,
        poll_interval=poll_interval,
        timeout=await_timeout,
        sleep=sleep,
    )


def get_heartbeat(
    s3: _S3ClientLike,
    run_id: str,
    *,
    bucket: str = DEFAULT_BUCKET,
) -> dict[str, Any] | None:
    """kernel が put する heartbeat JSON を取得 (なければ ``None``)。"""
    try:
        obj = s3.get_object(Bucket=bucket, Key=heartbeat_key(run_id))
    except Exception as e:
        if _is_no_such_key(e):
            return None
        raise
    body = (
        obj["Body"].read() if hasattr(obj.get("Body"), "read") else obj.get("Body", b"")
    )
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    parsed: dict[str, Any] = json.loads(body)
    return parsed


def cleanup_session(
    s3: _S3ClientLike,
    run_id: str,
    *,
    bucket: str = DEFAULT_BUCKET,
) -> int:
    """``kaggle_interactive/<run_id>/`` 配下を全削除する。戻り値は削除件数。"""
    deleted = 0
    prefix = f"{session_prefix(run_id)}/"
    continuation: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            s3.delete_object(Bucket=bucket, Key=obj["Key"])
            deleted += 1
        if not resp.get("IsTruncated"):
            break
        continuation = resp.get("NextContinuationToken")
    return deleted


def _is_no_such_key(err: Exception) -> bool:
    """``NoSuchKey`` / 404 を識別する (botocore ClientError 互換)。"""
    response = getattr(err, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            return True
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status == 404:
            return True
    return False
