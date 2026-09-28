"""Run metadata for the RunPod basis.

`gpu.common.run_meta` の `RunMetadata` (schema_version=1) に `runpod_pod_id` /
`runpod_offer_snapshot` の Optional フィールドが追加済みなので、本モジュールは
re-export と RunPod 用 helper だけを担う。
"""

from __future__ import annotations

from typing import Any

from gpu.common.run_meta import (
    RUN_ID_PATTERN,
    SCHEMA_VERSION,
    RunMetadata,
    RunStatus,
    generate_run_id,
    hash_case_config,
    hash_params,
    read_run_json,
    update_run_json,
    write_run_json,
)

__all__ = [
    "RUN_ID_PATTERN",
    "SCHEMA_VERSION",
    "RunMetadata",
    "RunStatus",
    "build_runpod_offer_snapshot",
    "generate_run_id",
    "hash_case_config",
    "hash_params",
    "read_run_json",
    "update_run_json",
    "write_run_json",
]


def build_runpod_offer_snapshot(
    *,
    gpu_type_id: str,
    display_name: str,
    memory_gb: int,
    cloud_type: str,
    secure_cloud: bool,
    community_cloud: bool,
    dph_total: float,
    data_center_id: str | None = None,
) -> dict[str, Any]:
    """`run.json.runpod_offer_snapshot` に格納する dict を組み立てる。"""
    return {
        "gpu_type_id": gpu_type_id,
        "display_name": display_name,
        "memory_gb": memory_gb,
        "cloud_type": cloud_type,
        "secure_cloud": secure_cloud,
        "community_cloud": community_cloud,
        "dph_total": dph_total,
        "data_center_id": data_center_id,
    }
