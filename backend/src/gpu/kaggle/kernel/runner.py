"""Kaggle Kernel の push + status polling。

KaggleApi の最小 surface のみ依存。テストでは Protocol 互換 mock を渡す。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from gpu.kaggle.kernel.state import KernelStatus, parse_status

logger = logging.getLogger(__name__)


class _KaggleApiLike(Protocol):
    def authenticate(self) -> None: ...

    def kernels_push_cli(
        self, folder: str, timeout: int | None, acc: int | None
    ) -> object: ...

    def kernels_status(self, kernel: str) -> object: ...


@dataclass(frozen=True)
class KernelPushResult:
    """kernels_push の戻り値正規化。"""

    slug: str
    version: int | None
    url: str | None
    raw: object


class KernelRunFailed(RuntimeError):
    """Kernel が ERROR / CANCEL_ACKNOWLEDGED で終わったときに raise。"""

    def __init__(self, slug: str, status: KernelStatus, message: str | None) -> None:
        super().__init__(
            f"kernel {slug} terminated with status={status.value} message={message!r}"
        )
        self.slug = slug
        self.status = status
        self.message = message


class KernelPollTimeout(TimeoutError):
    """`poll_status` の max_wait に達した。"""


def push_kernel(api: _KaggleApiLike, kernel_dir: Path) -> KernelPushResult:
    """``kernels_push_cli`` を呼んで戻り値を正規化する。

    Kaggle SDK 1.6+ では ``kernels_push_cli`` が ``timeout`` / ``acc`` (accelerator)
    の追加引数を要求する。kernel-metadata.json 経由で実体は決まるため None で渡す。
    """
    raw = api.kernels_push_cli(str(kernel_dir), None, None)
    slug = _extract_str(raw, ("ref", "slug", "kernelSlug")) or _slug_from_metadata(
        kernel_dir
    )
    version = _extract_int(raw, ("versionNumber", "version"))
    url = _extract_str(raw, ("url",))
    logger.info("kernel pushed: slug=%s version=%s url=%s", slug, version, url)
    return KernelPushResult(slug=slug, version=version, url=url, raw=raw)


def poll_status(
    api: _KaggleApiLike,
    slug: str,
    *,
    interval: float = 60.0,
    timeout: float = 36000.0,
    sleep: object = time.sleep,
) -> tuple[KernelStatus, str | None]:
    """status が終端 (COMPLETE / ERROR / CANCEL_ACKNOWLEDGED) になるまで polling する。

    ``sleep`` はテストで差し替え可能 (no-op 関数を渡す)。
    終端で ``(status, failureMessage)`` を返す。
    """
    if interval <= 0:
        raise ValueError(f"interval must be positive: {interval}")
    if timeout <= 0:
        raise ValueError(f"timeout must be positive: {timeout}")
    elapsed = 0.0
    last_status = KernelStatus.UNKNOWN
    while elapsed <= timeout:
        raw = api.kernels_status(slug)
        status = parse_status(raw)
        message = _extract_str(raw, ("failureMessage", "errorMessage", "message"))
        if status != last_status:
            logger.info("kernel %s status=%s message=%s", slug, status.value, message)
            last_status = status
        if status.is_terminal:
            return status, message
        sleep(interval)  # type: ignore[operator]
        elapsed += interval
    raise KernelPollTimeout(
        f"kernel {slug} did not reach terminal state within {timeout}s "
        f"(last status={last_status.value})"
    )


def _extract_str(raw: object, keys: tuple[str, ...]) -> str | None:
    if isinstance(raw, dict):
        for k in keys:
            v = raw.get(k)
            if isinstance(v, str) and v:
                return v
    for k in keys:
        v = getattr(raw, k, None)
        if isinstance(v, str) and v:
            return v
    return None


def _extract_int(raw: object, keys: tuple[str, ...]) -> int | None:
    if isinstance(raw, dict):
        for k in keys:
            v = raw.get(k)
            if isinstance(v, int):
                return v
    for k in keys:
        v = getattr(raw, k, None)
        if isinstance(v, int):
            return v
    return None


def _slug_from_metadata(kernel_dir: Path) -> str:
    """kernel-metadata.json の id を読む。

    ``kernels_push`` の戻り値が空のときの fallback。
    """
    import json

    meta = kernel_dir / "kernel-metadata.json"
    if not meta.is_file():
        raise FileNotFoundError(f"kernel-metadata.json not found under {kernel_dir}")
    data = json.loads(meta.read_text(encoding="utf-8"))
    slug = data.get("id")
    if not isinstance(slug, str) or "/" not in slug:
        raise ValueError(f"kernel-metadata.json id missing or invalid: {slug!r}")
    return slug
