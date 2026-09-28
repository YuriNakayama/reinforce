"""Pod cleanup helpers — SIGTERM-safe wrappers around the RunPod SDK.

The onstart bash trap already pushes ``90_cleanup_exit_<code>`` markers and
uploads ``onstart.log`` to S3. Cleanup from the *Python* side is a defense
in depth for the cases the bash trap cannot handle:

  * scheduling failure (pod never reaches RUNNING, no bash to trap)
  * image pull stuck (bash never starts)
  * external SIGKILL on the bash process (trap may not fire)
  * watcher detects EXITED — make sure billable seconds stop

`terminate_pod` is idempotent: a 404/already-removed pod is treated as
success. This module never raises into callers — failures are logged but
swallowed so that retry logic can decide whether to relaunch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanupResult:
    pod_id: str
    terminated: bool
    error: str | None


def terminate_pod(sdk: Any, pod_id: str) -> CleanupResult:
    """Terminate a pod via the RunPod SDK. Idempotent.

    Returns ``CleanupResult.terminated=True`` on a successful or already-removed
    pod. ``error`` is set when termination raised an unexpected exception.
    """
    if not pod_id:
        return CleanupResult(pod_id="", terminated=False, error="empty pod_id")
    try:
        sdk.terminate_pod(pod_id)
    except Exception as exc:  # noqa: BLE001 — never crash callers
        msg = str(exc)
        if "not found" in msg.lower() or "404" in msg:
            logger.info("pod %s already removed", pod_id)
            return CleanupResult(pod_id=pod_id, terminated=True, error=None)
        logger.exception("failed to terminate pod %s", pod_id)
        return CleanupResult(pod_id=pod_id, terminated=False, error=msg)
    logger.info("pod %s terminated", pod_id)
    return CleanupResult(pod_id=pod_id, terminated=True, error=None)


def stop_pod(sdk: Any, pod_id: str) -> CleanupResult:
    """Stop (but do not remove) a pod. Used when we want to keep the disk
    around for diagnostic SSH; ``terminate_pod`` should still be called
    afterwards so that billable seconds end.
    """
    if not pod_id:
        return CleanupResult(pod_id="", terminated=False, error="empty pod_id")
    try:
        sdk.stop_pod(pod_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("failed to stop pod %s", pod_id)
        return CleanupResult(pod_id=pod_id, terminated=False, error=str(exc))
    return CleanupResult(pod_id=pod_id, terminated=True, error=None)
