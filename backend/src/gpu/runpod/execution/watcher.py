"""学習 pod の終了監視 + デスクトップ通知。

`dev/runpod train --watch` および `dev/runpod watch <run_id>` の本体。pod 状態
と S3 marker をポーリングし、以下の終了状態を検出する:

- **success**: `99_done` marker が S3 に出現 (onstart の最終ステップ通過)
- **failure**: pod が SDK から消えた / `90_cleanup_exit_<N>` で N != 0 が観測
- **timeout**: max_wait_seconds を超えても上記が起きない

検出後 `runpod_io.notify.notify` で desktop 通知を発火し、結果サマリを返す。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from gpu.runpod.execution import progress as progress_mod
from gpu.runpod.notify import notify
from gpu.runpod.runpod import pod_state as pod_state_mod

DEFAULT_POLL_INTERVAL = 30.0  # seconds
DEFAULT_MAX_WAIT = 7200  # 2h, onstart の自殺タイムアウトと一致
DEFAULT_STALL_THRESHOLD = 900  # 15 分 marker 更新なしで stalled 判定

SUCCESS_STEP = "99_done"
CLEANUP_PREFIX = "90_cleanup_exit_"
DONE_STATUSES: frozenset[str] = frozenset({"EXITED", "TERMINATED"})


@dataclass(frozen=True)
class WatchResult:
    """1 回の watch ループの結末。"""

    outcome: str  # "success" / "failure" / "timeout"
    last_step: str | None
    elapsed_seconds: float
    detail: str

    @property
    def is_success(self) -> bool:
        return self.outcome == "success"


def _classify_markers(steps: list[str]) -> tuple[str | None, str | None]:
    """marker 列から (latest_step, terminal_outcome) を取り出す。

    terminal_outcome: "success" / "failure" / None (まだ進行中)。
    """
    latest = steps[-1] if steps else None
    terminal: str | None = None
    for step in steps:
        if step == SUCCESS_STEP:
            terminal = "success"
        elif step.startswith(CLEANUP_PREFIX):
            try:
                exit_code = int(step[len(CLEANUP_PREFIX) :])
            except ValueError:
                continue
            terminal = "success" if exit_code == 0 else "failure"
    return latest, terminal


def watch_pod(
    run_id: str,
    pod_id: str,
    *,
    aws_profile: str | None = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    max_wait_seconds: float = DEFAULT_MAX_WAIT,
    stall_threshold_seconds: float = DEFAULT_STALL_THRESHOLD,
    notifier: Any = notify,
    sleep: Any = time.sleep,
    clock: Any = time.monotonic,
    pod_state_module: Any = pod_state_mod,
    progress_module: Any = progress_mod,
) -> WatchResult:
    """終了するまで poll し、結果を返して通知も発火する。

    依存 (`pod_state_module` / `progress_module` / `notifier` / `sleep` / `clock`)
    はテスト時の差し替え用。本番では default。

    `stall_threshold_seconds`: marker 列の最新ステップが N 秒以上更新されず、
    かつ pod は RUNNING のときに `outcome="stalled"` で抜ける。0 以下で無効。
    """
    started = clock()
    last_step: str | None = None
    last_step_changed_at: float = started

    while True:
        elapsed = clock() - started
        if elapsed > max_wait_seconds:
            detail = f"watch timed out after {elapsed:.0f}s (last_step={last_step!r})"
            notifier(
                f"RunPod {run_id}: TIMEOUT",
                f"pod={pod_id} last_step={last_step or '(none)'}",
            )
            return WatchResult(
                outcome="timeout",
                last_step=last_step,
                elapsed_seconds=elapsed,
                detail=detail,
            )

        # S3 markers が一番信頼できる進捗源
        try:
            markers = progress_module.list_markers(run_id, profile=aws_profile)
        except Exception as exc:  # noqa: BLE001
            markers = []
            marker_err = str(exc)
        else:
            marker_err = ""
        steps = [m.step for m in markers]
        new_last_step, terminal = _classify_markers(steps)
        if new_last_step != last_step:
            last_step = new_last_step
            last_step_changed_at = clock()

        # SDK 側 pod 状態 (terminate されたら EXITED に)
        try:
            pod = pod_state_module.get_pod(pod_id)
        except Exception:  # noqa: BLE001
            pod = None
        pod_status = pod.desired_status if pod is not None else "GONE"

        if terminal == "success":
            detail = f"99_done marker present (pod_status={pod_status})"
            notifier(
                f"RunPod {run_id}: SUCCESS",
                f"pod={pod_id} elapsed={elapsed:.0f}s",
            )
            return WatchResult(
                outcome="success",
                last_step=last_step,
                elapsed_seconds=elapsed,
                detail=detail,
            )
        if terminal == "failure":
            detail = (
                f"cleanup_exit non-zero observed (last_step={last_step}); "
                f"pod_status={pod_status}"
            )
            notifier(
                f"RunPod {run_id}: FAILED",
                f"pod={pod_id} last_step={last_step}",
            )
            return WatchResult(
                outcome="failure",
                last_step=last_step,
                elapsed_seconds=elapsed,
                detail=detail,
            )

        # stalled: pod は RUNNING だが marker が一定時間更新されていない
        if (
            stall_threshold_seconds > 0
            and pod is not None
            and pod_status not in DONE_STATUSES
            and (clock() - last_step_changed_at) > stall_threshold_seconds
        ):
            stale_for = clock() - last_step_changed_at
            detail = (
                f"no marker progress for {stale_for:.0f}s "
                f"(threshold {stall_threshold_seconds:.0f}s, "
                f"last_step={last_step!r}, pod_status={pod_status})"
            )
            notifier(
                f"RunPod {run_id}: STALLED",
                f"pod={pod_id} last_step={last_step or '(none)'} "
                f"({stale_for:.0f}s no progress)",
            )
            return WatchResult(
                outcome="stalled",
                last_step=last_step,
                elapsed_seconds=elapsed,
                detail=detail,
            )

        # pod が消えていて success marker も無い → 失敗 (terminate 済 + 99_done 未到達)
        if pod is None or pod_status in DONE_STATUSES:
            detail = (
                f"pod gone before success marker (pod_status={pod_status}, "
                f"last_step={last_step!r}, marker_err={marker_err!r})"
            )
            notifier(
                f"RunPod {run_id}: FAILED",
                f"pod={pod_id} pod gone, last_step={last_step or '(none)'}",
            )
            return WatchResult(
                outcome="failure",
                last_step=last_step,
                elapsed_seconds=elapsed,
                detail=detail,
            )

        sleep(poll_interval)


__all__ = [
    "DEFAULT_MAX_WAIT",
    "DEFAULT_POLL_INTERVAL",
    "DEFAULT_STALL_THRESHOLD",
    "WatchResult",
    "watch_pod",
]
