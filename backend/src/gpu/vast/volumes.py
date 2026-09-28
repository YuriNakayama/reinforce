"""Vast.ai network volume helpers (永続化ボリューム検索 / 作成 / 一覧).

uv wheel cache や DVC cache を Vast インスタンス間で永続化するための薄いラッパ.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class VolumeOffer:
    """Vast.ai の volume search 結果 1 件 (storage offer)."""

    id: int
    machine_id: int
    disk_space_gb: float
    storage_per_hour: float
    geolocation: str | None
    reliability: float

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> VolumeOffer:
        return cls(
            id=int(raw.get("ask_contract_id") or raw.get("id", 0)),
            machine_id=int(raw.get("machine_id", 0)),
            disk_space_gb=float(raw.get("disk_space", 0.0)),
            storage_per_hour=float(raw.get("storage_cost", 0.0)),
            geolocation=raw.get("geolocation"),
            reliability=float(raw.get("reliability2", raw.get("reliability", 0.0))),
        )


@dataclass(frozen=True)
class Volume:
    """既存の volume contract."""

    id: int
    name: str
    size_gb: float
    mount_path: str | None
    machine_id: int | None
    storage_per_hour: float

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> Volume:
        # Vast.ai SDK の show_volumes() レスポンスは volume name を 'label' に
        # 入れる. 互換のため 'name' / 'volume_name' / 'label' を順に見る.
        name = raw.get("name") or raw.get("volume_name") or raw.get("label") or ""
        return cls(
            id=int(raw.get("id", 0)),
            name=str(name),
            size_gb=float(raw.get("disk_space", raw.get("size", 0.0))),
            mount_path=raw.get("mount_path"),
            machine_id=raw.get("machine_id"),
            storage_per_hour=float(raw.get("storage_cost", 0.0)),
        )


def search_volume_offers(
    sdk: Any,
    *,
    min_size_gb: float = 15.0,
    max_storage_per_month_per_tb: float = 1.00,
    min_reliability: float = 0.99,
    limit: int = 10,
    network: bool = False,
) -> list[VolumeOffer]:
    """Volume offer を検索. storage_cost ($/TB/month) 昇順で limit 件返す.

    `network=True` (デフォルト) なら network volume offer を検索する.
    Network volume は machine 非依存で attach 可能 (host 切替後も再利用できる).
    Local volume (network=False) は machine_id 固定なので host 切替時に再利用不可.
    """
    query: dict[str, Any] = {
        "disk_space": {"gte": float(min_size_gb)},
        "storage_cost": {"lt": float(max_storage_per_month_per_tb)},
        "reliability2": {"gte": float(min_reliability)},
    }
    search_fn = sdk.search_network_volumes if network else sdk.search_volumes
    raw_offers = search_fn(
        query=query,
        order=[["storage_cost", "asc"]],
        limit=limit,
        storage=min_size_gb,
    )
    offers = [VolumeOffer.from_raw(r) for r in raw_offers]
    offers.sort(key=lambda o: o.storage_per_hour)
    return offers


def render_volume_offers(
    offers: list[VolumeOffer], console: Console | None = None
) -> None:
    """Volume offers を rich table で表示."""
    if console is None:
        console = Console()
    table = Table(title="Vast.ai volume offers (cheapest first)")
    table.add_column("#", justify="right")
    table.add_column("offer_id")
    table.add_column("machine_id")
    table.add_column("size_gb", justify="right")
    table.add_column("$/TB/mo", justify="right")
    table.add_column("est. $/mo", justify="right")
    table.add_column("reliability", justify="right")
    table.add_column("region")
    for i, o in enumerate(offers, 1):
        # Estimated monthly cost for 15GB allocation
        est_15gb = o.storage_per_hour * 15.0 / 1024.0
        table.add_row(
            str(i),
            str(o.id),
            str(o.machine_id),
            f"{o.disk_space_gb:.0f}",
            f"${o.storage_per_hour:.4f}",
            f"${est_15gb:.4f}",
            f"{o.reliability:.3f}",
            o.geolocation or "-",
        )
    console.print(table)


def pick_volume_offer(
    offers: list[VolumeOffer], *, default_index: int = 1, console: Console | None = None
) -> VolumeOffer:
    """Stdin で offer 番号を選択. 空入力 / EOF なら default_index を選ぶ."""
    if console is None:
        console = Console()
    if not offers:
        raise ValueError("no offers available")
    render_volume_offers(offers, console=console)
    try:
        raw = input(f"Pick volume offer # ({default_index}): ").strip()
    except EOFError:
        raw = ""
    if not raw:
        idx = default_index
    else:
        idx = int(raw)
    if idx < 1 or idx > len(offers):
        raise ValueError(f"invalid choice {idx}; expected 1..{len(offers)}")
    return offers[idx - 1]


_VALID_VOLUME_NAME = re.compile(r"^[A-Za-z0-9_]{1,64}$")


def validate_volume_name(name: str) -> None:
    """Vast.ai の制約: alphanumeric + underscore, max 64 chars."""
    if not _VALID_VOLUME_NAME.match(name):
        raise ValueError(
            f"invalid volume name {name!r}: Vast.ai requires "
            "alphanumeric + underscore only, max 64 chars"
        )


def create_volume(
    sdk: Any, *, offer_id: int, size_gb: float, name: str, network: bool = True
) -> int:
    """Volume を新規作成し volume_id を返す.

    `network=True` (デフォルト) なら machine 非依存の network volume を作成する.
    これにより host を切り替えても同じ volume を再利用できる.
    """
    validate_volume_name(name)
    create_fn = sdk.create_network_volume if network else sdk.create_volume
    response = create_fn(id=offer_id, size=size_gb, name=name)
    if not isinstance(response, Mapping):
        raise RuntimeError(
            f"unexpected create_volume response: {type(response).__name__}"
        )
    # 一部の Vast API endpoint (network_volumes) は id を返さず success フラグと
    # name/size のみを返すので、その場合は list_volumes で name 一致を引いて id を
    # 解決する.
    new_id = response.get("new_contract") or response.get("id")
    if new_id is None:
        if not response.get("success"):
            raise RuntimeError(f"create_volume failed: response={dict(response)}")
        # show_network_disks / show_volumes から name 一致を探す
        for v in list_volumes(sdk):
            if v.name == name:
                return int(v.id)
        raise RuntimeError(
            f"create_volume succeeded but volume not found by name={name!r}: "
            f"response={dict(response)}"
        )
    return int(new_id)


def list_volumes(sdk: Any) -> list[Volume]:
    """所有する volumes を一覧 (network volume も同じ show_volumes に含まれる).

    `show_network_disks()` は Host 専用 API (Customer は 401) なので使わない.
    Customer 視点では show_volumes() が network/local 両方を返す.
    """
    raw = sdk.show_volumes()
    if not isinstance(raw, list):
        return []
    return [Volume.from_raw(r) for r in raw]


def add_network_disk_to_machine(
    sdk: Any, *, disk_id: int, machine_id: int, mount_point: str
) -> dict[str, Any]:
    """Network volume を指定 machine に mount する (post-attach 用)."""
    response = sdk.add_network_disk(
        machines=[machine_id], mount_point=mount_point, disk_id=disk_id
    )
    if isinstance(response, Mapping):
        return dict(response)
    return {"raw": response}


def find_volume_by_name(volumes: list[Volume], name: str) -> Volume | None:
    """name 一致 (前方一致 OK) の volume を返す. 複数あれば最新 id."""
    matches = [v for v in volumes if v.name == name]
    if not matches:
        return None
    matches.sort(key=lambda v: v.id, reverse=True)
    return matches[0]


__all__ = [
    "Volume",
    "VolumeOffer",
    "add_network_disk_to_machine",
    "create_volume",
    "find_volume_by_name",
    "list_volumes",
    "pick_volume_offer",
    "render_volume_offers",
    "search_volume_offers",
]
