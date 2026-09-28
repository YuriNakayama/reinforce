"""RunPod network volume helpers (GraphQL 直叩き).

Pod 作成時のみ attach 可能、Secure Cloud 専用、特定 DC のみ提供 — これらの
制約は RunPod 公式仕様。Pod 起動後の attach/detach は不可で、Pod を削除して
再作成する必要がある (Vast.ai と同設計)。

公式 SDK には volume CRUD ヘルパが無いため、`runpod.api.graphql.run_graphql_query`
を直接使う。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import requests
from rich.console import Console
from rich.table import Table

# RunPod 公式 docs から拾った代表 DC 一覧。SDK / API 側に list endpoint が
# 安定して提供されていないため、本基盤では静的リストとして保持する。volume
# 作成時の入力 hint として使い、誤入力を弾くことが目的。
# Network volume 対応 DC (RunPod REST API のエラーメッセージで返ってきた最新値)。
# DC は時々追加/廃止されるため、もし create_volume が "not found" を返した場合は
# このリストを最新化すること。
KNOWN_DATA_CENTERS: tuple[str, ...] = (
    "AP-JP-1",
    "CA-MTL-3",
    "CA-MTL-4",
    "EU-CZ-1",
    "EU-NL-1",
    "EU-RO-1",
    "EU-SE-1",
    "EUR-IS-1",
    "EUR-IS-3",
    "EUR-NO-1",
    "US-CA-2",
    "US-GA-2",
    "US-IL-1",
    "US-KS-2",
    "US-MO-1",
    "US-MO-2",
    "US-NC-2",
    "US-NE-1",
)

# storage 価格は仕様で固定 ($0.07/GB/月 for first 1TB)。表示用の参考値。
DEFAULT_STORAGE_USD_PER_GB_PER_MONTH = 0.07


@dataclass(frozen=True)
class Volume:
    """RunPod の既存 network volume contract."""

    id: str
    name: str
    size_gb: int
    data_center_id: str

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> Volume:
        return cls(
            id=str(raw.get("id", "")),
            name=str(raw.get("name", "")),
            size_gb=int(raw.get("size", 0)),
            data_center_id=str(raw.get("dataCenterId", "")),
        )


@dataclass(frozen=True)
class VolumeOffer:
    """利用可能な data center とサイズの目安 (実際の availability は API 側)。"""

    data_center_id: str
    storage_usd_per_gb_per_month: float
    suggested_size_gb: int


def list_volumes(sdk: Any) -> list[Volume]:
    """所有 network volumes を一覧する。

    SDK 経由でも GraphQL 経由でも同じ `myself.networkVolumes` field を引く。
    `sdk` には `runpod.api.graphql` モジュールを渡す前提 (`run_graphql_query`
    が attribute として呼べる)。
    """
    query = """
    query MyVolumes {
      myself {
        networkVolumes {
          id
          name
          size
          dataCenterId
        }
      }
    }
    """
    response = sdk.run_graphql_query(query)
    if not isinstance(response, Mapping):
        raise RuntimeError(
            f"unexpected run_graphql_query response: {type(response).__name__}"
        )
    data = response.get("data") or {}
    myself = data.get("myself") or {}
    raw_volumes = myself.get("networkVolumes") or []
    if not isinstance(raw_volumes, list):
        return []
    return [Volume.from_raw(r) for r in raw_volumes if isinstance(r, Mapping)]


def search_volume_offers(
    *,
    min_size_gb: int = 15,
    data_center_id: str | None = None,
) -> list[VolumeOffer]:
    """利用可能な DC を返す静的リスト (RunPod の API は動的 search を提供しない)。

    `data_center_id` 明示があれば一致するもののみに絞り込む。
    """
    centers: list[str] = list(KNOWN_DATA_CENTERS)
    if data_center_id is not None:
        centers = [c for c in centers if c == data_center_id]
    return [
        VolumeOffer(
            data_center_id=c,
            storage_usd_per_gb_per_month=DEFAULT_STORAGE_USD_PER_GB_PER_MONTH,
            suggested_size_gb=int(min_size_gb),
        )
        for c in centers
    ]


def render_volume_offers(
    offers: list[VolumeOffer], console: Console | None = None
) -> None:
    """volume offer を rich table で表示する。"""
    if console is None:
        console = Console()
    table = Table(title="RunPod volume data centers")
    table.add_column("#", justify="right")
    table.add_column("data_center")
    table.add_column("size_gb", justify="right")
    table.add_column("$/GB/month", justify="right")
    for i, o in enumerate(offers, 1):
        table.add_row(
            str(i),
            o.data_center_id,
            str(o.suggested_size_gb),
            f"${o.storage_usd_per_gb_per_month:.4f}",
        )
    console.print(table)


def pick_volume_offer(
    offers: list[VolumeOffer], *, default_index: int = 1
) -> VolumeOffer:
    """stdin で番号入力。空入力 / EOF なら default_index を選ぶ。"""
    if not offers:
        raise ValueError("no offers available")
    render_volume_offers(offers)
    try:
        raw = input(f"Pick volume DC # ({default_index}): ").strip()
    except EOFError:
        raw = ""
    if not raw:
        idx = default_index
    else:
        idx = int(raw)
    if idx < 1 or idx > len(offers):
        raise ValueError(f"invalid choice {idx}; expected 1..{len(offers)}")
    return offers[idx - 1]


_VALID_VOLUME_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_volume_name(name: str) -> None:
    """RunPod 側の制約 (現時点で公式 docs には明示なし) を保守的に置く。

    alphanumeric + `_` / `-`、64 chars 以内。
    """
    if not _VALID_VOLUME_NAME.match(name):
        raise ValueError(
            f"invalid volume name {name!r}: alphanumeric / underscore / hyphen "
            "only, max 64 chars"
        )


def create_volume(
    sdk: Any,
    *,
    name: str,
    size_gb: int,
    data_center_id: str,
    api_key: str | None = None,
) -> str:
    """新規 network volume を作成し、id を返す。REST API 経由。

    `runpod.api.graphql.run_graphql_query` は variables 引数を受け付けない
    (`(query, api_key=None)` のみ) ため、CreateNetworkVolume mutation を
    SDK から直接叩けない。代わりに REST API `/v1/networkvolumes` を使う。
    `api_key` が None なら runpod_sdk.api_key を読む。
    """
    import runpod as runpod_sdk

    validate_volume_name(name)
    if data_center_id not in KNOWN_DATA_CENTERS:
        raise ValueError(
            f"unknown data_center_id {data_center_id!r}; known: {KNOWN_DATA_CENTERS}"
        )

    effective_key = api_key or runpod_sdk.api_key
    if not effective_key:
        raise RuntimeError("RUNPOD_API_KEY required for create_volume")

    response = requests.post(
        "https://rest.runpod.io/v1/networkvolumes",
        headers={
            "Authorization": f"Bearer {effective_key}",
            "Content-Type": "application/json",
        },
        json={
            "name": name,
            "size": int(size_gb),
            "dataCenterId": data_center_id,
        },
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"create_volume failed: HTTP {response.status_code} {response.text[:300]}"
        )
    data = response.json()
    if not isinstance(data, Mapping):
        raise RuntimeError(f"unexpected response type: {type(data).__name__}")
    new_id = data.get("id")
    if new_id is None:
        raise RuntimeError(f"createNetworkVolume returned no id: response={data}")
    return str(new_id)


def find_volume_by_name(volumes: list[Volume], name: str) -> Volume | None:
    """name 完全一致の volume を返す。複数あれば最後 (最新作成想定)。"""
    matches = [v for v in volumes if v.name == name]
    if not matches:
        return None
    return matches[-1]


__all__ = [
    "KNOWN_DATA_CENTERS",
    "Volume",
    "VolumeOffer",
    "create_volume",
    "find_volume_by_name",
    "list_volumes",
    "pick_volume_offer",
    "render_volume_offers",
    "search_volume_offers",
    "validate_volume_name",
]
