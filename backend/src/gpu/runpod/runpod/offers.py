"""runpod SDK の get_gpus + get_gpu をラップし、対話的に offer を選択する。

RunPod は Vast.ai と異なり offer を 1 件単位の DSL クエリでは取れない。代わりに
- `runpod.get_gpus()` で GPU type 一覧 (id / displayName / memoryInGb)
- `runpod.get_gpu(id)` で 1 種類分の価格 (securePrice / communityPrice / spot 各種)
の 2 段階で取得する。本モジュールはこの 2 段階呼び出しを 1 つの `Offer`
ストリームに合成し、cloud_type filter を適用する。
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table

# Default GPU candidates: 24GB+ クラスを Secure / Community 両対応で並べる。
DEFAULT_GPU_NAMES: tuple[str, ...] = (
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA A100 80GB PCIe",
)

CloudType = str  # "SECURE" / "COMMUNITY" / "ALL"


@dataclass(frozen=True)
class Offer:
    """1 つの GPU type × cloud_type 組合せ。Vast.ai の Offer に相当。"""

    gpu_type_id: str
    display_name: str
    memory_gb: int
    secure_cloud: bool
    community_cloud: bool
    secure_price: float | None
    community_price: float | None
    secure_spot_price: float | None
    community_spot_price: float | None
    cloud_type: str  # "SECURE" / "COMMUNITY"
    dph_total: float  # cloud_type に応じた選択価格

    def to_snapshot(self) -> dict[str, Any]:
        """run.json.runpod_offer_snapshot に保存する plain dict 形式。"""
        return {
            "gpu_type_id": self.gpu_type_id,
            "display_name": self.display_name,
            "memory_gb": self.memory_gb,
            "secure_cloud": self.secure_cloud,
            "community_cloud": self.community_cloud,
            "secure_price": self.secure_price,
            "community_price": self.community_price,
            "secure_spot_price": self.secure_spot_price,
            "community_spot_price": self.community_spot_price,
            "cloud_type": self.cloud_type,
            "dph_total": self.dph_total,
        }


def _to_float(value: Any) -> float | None:
    """SDK の応答に numeric な値があれば float、なければ None。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_offer(
    detail: dict[str, Any],
    *,
    cloud_type: str,
    dph_total: float,
) -> Offer:
    return Offer(
        gpu_type_id=str(detail.get("id", "")),
        display_name=str(detail.get("displayName", "")),
        memory_gb=int(detail.get("memoryInGb", 0) or 0),
        secure_cloud=bool(detail.get("secureCloud", False)),
        community_cloud=bool(detail.get("communityCloud", False)),
        secure_price=_to_float(detail.get("securePrice")),
        community_price=_to_float(detail.get("communityPrice")),
        secure_spot_price=_to_float(detail.get("secureSpotPrice")),
        community_spot_price=_to_float(detail.get("communitySpotPrice")),
        cloud_type=cloud_type,
        dph_total=dph_total,
    )


def search_offers(
    sdk: Any,
    *,
    gpu_names: Sequence[str] = DEFAULT_GPU_NAMES,
    cloud_type: CloudType = "SECURE",
    max_dph: float = 2.0,
    min_memory_gb: int = 16,
    limit: int = 10,
) -> list[Offer]:
    """`get_gpus` → 各 id を `get_gpu` で詳細取得 → cloud_type filter → dph asc。

    `cloud_type="ALL"` のときは Secure と Community それぞれで Offer を生成し
    両方候補にする (1 GPU から最大 2 Offer)。
    `sdk` は `runpod` SDK (module 自体) または mock を渡せる。
    """
    if cloud_type not in ("SECURE", "COMMUNITY", "ALL"):
        raise ValueError(f"cloud_type must be SECURE/COMMUNITY/ALL, got {cloud_type!r}")

    target_ids = set(gpu_names)
    raw_gpus = sdk.get_gpus()
    if not isinstance(raw_gpus, list):
        raise RuntimeError(
            f"unexpected get_gpus response type: {type(raw_gpus).__name__}"
        )

    candidates: list[Offer] = []
    for entry in raw_gpus:
        if not isinstance(entry, dict):
            continue
        gpu_id = str(entry.get("id", ""))
        if gpu_id not in target_ids:
            continue
        detail = sdk.get_gpu(gpu_id)
        if not isinstance(detail, dict):
            continue
        memory_gb = int(detail.get("memoryInGb", 0) or 0)
        if memory_gb < min_memory_gb:
            continue
        secure_cloud = bool(detail.get("secureCloud", False))
        community_cloud = bool(detail.get("communityCloud", False))
        secure_price = _to_float(detail.get("securePrice"))
        community_price = _to_float(detail.get("communityPrice"))

        if (
            cloud_type in ("SECURE", "ALL")
            and secure_cloud
            and secure_price is not None
            and secure_price <= max_dph
        ):
            candidates.append(
                _build_offer(detail, cloud_type="SECURE", dph_total=secure_price)
            )
        if (
            cloud_type in ("COMMUNITY", "ALL")
            and community_cloud
            and community_price is not None
            and community_price <= max_dph
        ):
            candidates.append(
                _build_offer(detail, cloud_type="COMMUNITY", dph_total=community_price)
            )

    candidates.sort(key=lambda o: o.dph_total)
    return candidates[:limit]


def format_table(offers: Sequence[Offer]) -> Table:
    """rich.Table で offer 一覧を整形する。"""
    table = Table(title="RunPod offers (cheapest first)")
    table.add_column("#", justify="right")
    table.add_column("gpu_type_id")
    table.add_column("display_name")
    table.add_column("mem_gb", justify="right")
    table.add_column("cloud")
    table.add_column("dph", justify="right")
    table.add_column("spot")
    for idx, offer in enumerate(offers, start=1):
        spot = (
            offer.secure_spot_price
            if offer.cloud_type == "SECURE"
            else offer.community_spot_price
        )
        table.add_row(
            str(idx),
            offer.gpu_type_id,
            offer.display_name,
            str(offer.memory_gb),
            offer.cloud_type,
            f"${offer.dph_total:.3f}",
            f"${spot:.3f}" if spot is not None else "-",
        )
    return table


def pick_offer(offers: Sequence[Offer], *, console: Console | None = None) -> Offer:
    """Rich Prompt で番号入力させ、対応する Offer を返す。

    stdin が tty でない場合 (CI / 自動化スクリプト経由) は対話せず先頭 (=最安)
    を採用する。`IntPrompt.ask` は non-tty でも default を返す実装ではなく
    `EOFError` 経由で Abort してしまうため、ここで明示的に分岐する。
    """
    if not offers:
        raise RuntimeError("no offers to pick from")
    console = console or Console()
    console.print(format_table(offers))
    if not sys.stdin.isatty():
        console.print(
            "[dim]non-interactive stdin detected; auto-picking offer #1[/dim]"
        )
        return offers[0]
    while True:
        choice = IntPrompt.ask(
            "Pick offer #",
            console=console,
            default=1,
        )
        if 1 <= choice <= len(offers):
            return offers[choice - 1]
        console.print(f"[red]out of range: 1..{len(offers)}[/red]")
