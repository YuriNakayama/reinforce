"""Pre-terminate log rescue: pull `/var/log/onstart.log` from a stalled pod
via SSH and upload it to S3 before cleanup terminates the pod.

Used by `dev/runpod train --watch` and `dev/runpod watch` when watcher returns
``outcome="stalled"``. The bash trap (`cleanup_destroy`) cannot run if the pod
is killed externally by the SDK terminate path, so we must capture the live
log first. Failure here is non-fatal — cleanup still proceeds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from gpu.runpod.execution.progress import upload_rescue_log
from gpu.runpod.runpod.ssh import (
    SshUnavailable,
    get_pod_ssh_endpoint,
    grab_onstart_log,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RescueResult:
    """1 回の rescue 試行の結末。"""

    uploaded: bool
    s3_uri: str | None
    error: str | None
    bytes_uploaded: int


def rescue_onstart_log(
    sdk: Any,
    pod_id: str,
    run_id: str,
    *,
    aws_profile: str | None = None,
    ssh_timeout_seconds: float = 30.0,
    ssh_endpoint_resolver: Any = get_pod_ssh_endpoint,
    log_grabber: Any = grab_onstart_log,
    uploader: Any = upload_rescue_log,
) -> RescueResult:
    """Pull onstart.log over SSH and upload to S3 as a rescue snapshot.

    Each external dependency (SSH endpoint resolution, log grab, S3 upload)
    is injected to keep this unit testable. Defaults wire up the real
    implementations. Never raises — failures are recorded in ``error``.
    """
    try:
        endpoint = ssh_endpoint_resolver(sdk, pod_id)
    except SshUnavailable as exc:
        return RescueResult(
            uploaded=False,
            s3_uri=None,
            error=f"ssh unavailable: {exc}",
            bytes_uploaded=0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("rescue: ssh endpoint resolution failed")
        return RescueResult(
            uploaded=False,
            s3_uri=None,
            error=f"ssh endpoint error: {exc}",
            bytes_uploaded=0,
        )

    try:
        text = log_grabber(endpoint, timeout_seconds=ssh_timeout_seconds)
    except SshUnavailable as exc:
        return RescueResult(
            uploaded=False,
            s3_uri=None,
            error=f"log grab failed: {exc}",
            bytes_uploaded=0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("rescue: unexpected log grab failure")
        return RescueResult(
            uploaded=False,
            s3_uri=None,
            error=f"log grab error: {exc}",
            bytes_uploaded=0,
        )

    try:
        s3_uri = uploader(run_id, text, profile=aws_profile)
    except Exception as exc:  # noqa: BLE001
        logger.exception("rescue: S3 upload failed")
        return RescueResult(
            uploaded=False,
            s3_uri=None,
            error=f"s3 upload error: {exc}",
            bytes_uploaded=0,
        )

    return RescueResult(
        uploaded=True,
        s3_uri=s3_uri,
        error=None,
        bytes_uploaded=len(text.encode("utf-8")),
    )


__all__ = ["RescueResult", "rescue_onstart_log"]
