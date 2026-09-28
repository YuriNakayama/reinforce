"""Kaggle Kernel の status enum + parse helper。"""

from __future__ import annotations

from enum import Enum


class KernelStatus(str, Enum):
    """Kaggle kernels_status の status 文字列を正規化したもの。"""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    CANCEL_ACKNOWLEDGED = "cancel_acknowledged"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        """status が終端状態 (これ以上 polling しても進まない) かを返す。"""
        return self in (
            KernelStatus.COMPLETE,
            KernelStatus.ERROR,
            KernelStatus.CANCEL_ACKNOWLEDGED,
        )


def parse_status(raw: object) -> KernelStatus:
    """kernels_status の戻り値から KernelStatus を決定する。

    Kaggle API は dict / 専用 SDK Enum (``KernelWorkerStatus``) / 文字列 を返す
    実装の揺れがあるため、複数経路を許容する。
    """
    if isinstance(raw, KernelStatus):
        return raw
    if isinstance(raw, dict):
        value = raw.get("status") or raw.get("state")
        if value is not None:
            return _from_value(value)
    if hasattr(raw, "status"):
        return _from_value(raw.status)
    return _from_value(raw)


def _from_value(value: object) -> KernelStatus:
    """str / Enum / その他から KernelStatus を引き当てる。"""
    if isinstance(value, KernelStatus):
        return value
    if isinstance(value, str):
        return _from_str(value)
    # SDK Enum (kagglesdk.kernels...KernelWorkerStatus.RUNNING etc.) は
    # ``.name`` を持つ。
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return _from_str(name)
    return KernelStatus.UNKNOWN


def _from_str(value: str) -> KernelStatus:
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    # ``KernelWorkerStatus.RUNNING`` のような repr が渡された場合に
    # 最後のドット以降だけ使う。
    if "." in normalized:
        normalized = normalized.rsplit(".", 1)[-1]
    try:
        return KernelStatus(normalized)
    except ValueError:
        return KernelStatus.UNKNOWN
