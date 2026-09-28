"""vastai SDK の search_offers をラップし、対話的に offer を選択する。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table

DEFAULT_GPU_NAMES: tuple[str, ...] = (
    "RTX_3090",
    "RTX_4090",
    "RTX_A6000",
    "A100",
)


@dataclass(frozen=True)
class Offer:
    """Vast.ai search-offers の 1 件分。表示・選択に必要なフィールドだけ保持する。"""

    id: int
    gpu_name: str
    num_gpus: int
    dph_total: float
    reliability: float
    geolocation: str | None
    cuda_max_good: float | None
    inet_down: float | None
    verified: bool

    def to_snapshot(self) -> dict[str, Any]:
        """run.json に保存するための plain dict 形式。"""
        return {
            "id": self.id,
            "gpu_name": self.gpu_name,
            "num_gpus": self.num_gpus,
            "dph_total": self.dph_total,
            "reliability": self.reliability,
            "geolocation": self.geolocation,
            "cuda_max_good": self.cuda_max_good,
            "inet_down": self.inet_down,
            "verified": self.verified,
        }


def _build_query(
    *,
    gpu_names: Sequence[str],
    max_dph: float,
    min_reliability: float,
    min_cuda: float,
) -> str:
    """vastai CLI 互換の query 文字列を組み立てる。

    複数 GPU 名は `gpu_name in [X,Y]` 構文を使う必要がある。
    `gpu_name=X gpu_name=Y` の並列指定は AND として処理されて 0 件になる。
    """
    if len(gpu_names) == 1:
        gpu_clause = f"gpu_name={gpu_names[0]}"
    else:
        gpu_clause = f"gpu_name in [{','.join(gpu_names)}]"
    parts = [
        gpu_clause,
        "num_gpus=1",
        f"reliability>={min_reliability}",
        f"cuda_max_good>={min_cuda}",
        f"dph_total<{max_dph}",
        "rentable=true",
        "verified=true",
    ]
    return " ".join(p for p in parts if p)


def _parse_offer(raw: dict[str, Any]) -> Offer:
    return Offer(
        id=int(raw["id"]),
        gpu_name=str(raw.get("gpu_name", "unknown")),
        num_gpus=int(raw.get("num_gpus", 1)),
        dph_total=float(raw.get("dph_total", 0.0)),
        reliability=float(raw.get("reliability2", raw.get("reliability", 0.0))),
        geolocation=raw.get("geolocation"),
        cuda_max_good=(
            float(raw["cuda_max_good"])
            if raw.get("cuda_max_good") is not None
            else None
        ),
        inet_down=(
            float(raw["inet_down"]) if raw.get("inet_down") is not None else None
        ),
        verified=bool(raw.get("verified", False)),
    )


def search_offers(
    sdk: Any,
    *,
    gpu_names: Sequence[str] = DEFAULT_GPU_NAMES,
    max_dph: float = 1.0,
    min_reliability: float = 0.99,
    min_cuda: float = 12.0,
    limit: int = 10,
    machine_id: int | None = None,
) -> list[Offer]:
    """vastai SDK 経由で offers を検索し、dph_total 昇順で limit 件返す。

    `sdk` は `VastAI()` インスタンス (テストでは mock を渡せる)。
    `machine_id` を指定すると、その machine 上の offer のみに絞り込む
    (Local volume を attach する場合に必要).
    """
    query = _build_query(
        gpu_names=gpu_names,
        max_dph=max_dph,
        min_reliability=min_reliability,
        min_cuda=min_cuda,
    )
    if machine_id is not None:
        query = f"{query} machine_id={int(machine_id)}"
    raw_results = sdk.search_offers(
        query=query, type="on-demand", order="dph_total", limit=limit
    )
    if not isinstance(raw_results, list):
        raise RuntimeError(
            f"unexpected search_offers response type: {type(raw_results).__name__}"
        )
    offers = [_parse_offer(r) for r in raw_results]
    offers.sort(key=lambda o: o.dph_total)
    return offers[:limit]


def format_table(offers: Sequence[Offer]) -> Table:
    """rich.Table で offer 一覧を整形する。"""
    table = Table(title="Vast.ai offers (cheapest first)")
    table.add_column("#", justify="right")
    table.add_column("offer_id", justify="right")
    table.add_column("gpu")
    table.add_column("num", justify="right")
    table.add_column("dph", justify="right")
    table.add_column("reliability", justify="right")
    table.add_column("cuda")
    table.add_column("region")
    for idx, offer in enumerate(offers, start=1):
        table.add_row(
            str(idx),
            str(offer.id),
            offer.gpu_name,
            str(offer.num_gpus),
            f"${offer.dph_total:.3f}",
            f"{offer.reliability:.3f}",
            f"{offer.cuda_max_good:.1f}" if offer.cuda_max_good is not None else "?",
            offer.geolocation or "?",
        )
    return table


def pick_offer(offers: Sequence[Offer], *, console: Console | None = None) -> Offer:
    """rich Prompt で番号入力させ、対応する Offer を返す。"""
    if not offers:
        raise RuntimeError("no offers to pick from")
    console = console or Console()
    console.print(format_table(offers))
    while True:
        choice = IntPrompt.ask(
            "Pick offer #",
            console=console,
            default=1,
        )
        if 1 <= choice <= len(offers):
            return offers[choice - 1]
        console.print(f"[red]out of range: 1..{len(offers)}[/red]")
