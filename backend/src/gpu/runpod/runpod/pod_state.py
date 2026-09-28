"""RunPod SDK の get_pod / get_pods を reinforce 用に薄くラップする。

`runpod_sdk.get_pods()` は dict のリストを返すが、フィールド名が GraphQL
そのままで読みづらい。本モジュールは `PodState` dataclass に正規化し、
`dev/runpod ps` / `dev/runpod status` から共通利用する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import runpod as runpod_sdk

ACTIVE_STATUSES: frozenset[str] = frozenset({"RUNNING", "STARTING", "PROVISIONING"})


@dataclass(frozen=True)
class PodState:
    """RunPod pod の現在状態を正規化した snapshot。"""

    pod_id: str
    name: str
    desired_status: str
    cost_per_hr: float
    uptime_seconds: int
    gpu_display_name: str
    image_name: str

    @property
    def is_active(self) -> bool:
        return self.desired_status in ACTIVE_STATUSES

    @property
    def estimated_cost_usd(self) -> float:
        return self.cost_per_hr * self.uptime_seconds / 3600.0


def _to_state(payload: dict[str, Any]) -> PodState:
    machine = payload.get("machine") or {}
    return PodState(
        pod_id=str(payload.get("id", "")),
        name=str(payload.get("name", "")),
        desired_status=str(payload.get("desiredStatus", "UNKNOWN")),
        cost_per_hr=float(payload.get("costPerHr") or 0.0),
        uptime_seconds=int(payload.get("uptimeSeconds") or 0),
        gpu_display_name=str(machine.get("gpuDisplayName") or ""),
        image_name=str(payload.get("imageName") or ""),
    )


def list_pods() -> list[PodState]:
    """所有する pod 全件を取得し PodState のリストで返す。"""
    raw = runpod_sdk.get_pods()
    if not raw:
        return []
    return [_to_state(p) for p in raw]


def get_pod(pod_id: str) -> PodState | None:
    """pod_id 単体を取得。未存在の場合 None。"""
    raw = runpod_sdk.get_pod(pod_id)
    if not raw:
        return None
    return _to_state(raw)


def format_uptime(seconds: int) -> str:
    """秒数を人間可読な `1h23m` / `45m` / `12s` に整形。"""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours}h{minutes:02d}m"
