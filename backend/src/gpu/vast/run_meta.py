"""Run metadata for the Vast.ai basis.

`gpu.common.run_meta` が全基盤共通の `RunMetadata` (schema_version=1、
`vast_instance_id` / `vast_offer_snapshot` を含む) と run_id / config hash /
JSON I/O を持つので、本モジュールは re-export だけを担う。
"""

from __future__ import annotations

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
    "generate_run_id",
    "hash_case_config",
    "hash_params",
    "read_run_json",
    "update_run_json",
    "write_run_json",
]
