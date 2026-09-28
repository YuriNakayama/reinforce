"""Town shop demand, replayed from the engine's RNG stream.

Shops unlock at end-of-day from ``random.Random((seed * 1_000_003) ^ day)``.
Reproducing that on an accelerator is not practical (Mersenne Twister with a
data-dependent ``_randbelow`` rejection loop), so the schedule is generated
host-side and handed to the step function as a dense ``(days, N_PRODUCTS)``
demand table.

**The draw count is board-dependent.** ``_spawn_weeds`` shares the same RNG and
runs first, drawing once per *empty* tile per player — so the number of draws
consumed before ``rng.choice`` shrinks as weeds and crops occupy tiles.
Measured on a two-player PASS episode: 50 draws on day 0, 49 once a weed
appears, 48 after another. Feeding a wrong count silently selects a different
shop, which is why :func:`unlocked_shops_by_day` takes the per-day empty-tile
counts rather than assuming a constant.

Consequence for this port: the shop schedule can only be pre-tabulated when the
per-day empty-tile counts are known in advance. For the scripted policies used
in the benchmark that holds; for arbitrary play it does not, and the honest
options are to run the schedule host-side in lockstep with the batch or to
disable shops in both engines when comparing. See ``benchmark.py``.
"""

from __future__ import annotations

import random

import jax.numpy as jnp

from .state import N_PRODUCTS, PRODUCT_NAMES

#: SHOPS from the engine. Order matters: ``rng.choice(sorted(SHOPS))``.
SHOPS: dict[str, tuple[str, ...]] = {
    "BAKERY": ("EGG", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "YARN_STORE": ("WOOL",),
}

MAX_SHOP_INSTANCES = 8
SHOP_UNLOCK_INTERVAL = 3
SHOP_SELL_INTERVAL = 4
TOWN_CENTER_SELL_INTERVAL = 24

_PRODUCT_INDEX = {name: i for i, name in enumerate(PRODUCT_NAMES)}


def unlocked_shops_by_day(
    seed: int,
    days: int,
    *,
    board_size: int = 10,
    n_players: int = 2,
    empty_tiles_per_day: int = 25,
) -> list[list[str]]:
    """Replay the engine's unlock draw for each day boundary.

    Returns, for each day ``d``, the shop list in effect *during* day ``d``.
    The engine appends at the end of day ``d`` for ``next_day = d + 1``, so a
    shop drawn at the close of day 2 is active from day 3 onward.

    ``_spawn_weeds`` shares this RNG and runs first, drawing once per **empty**
    tile per player. Those draws must be consumed in the same order or
    ``rng.choice`` lands on a different shop.

    ``empty_tiles_per_day`` is that per-player count. It defaults to the 25
    unlocked NW tiles, which is exact for a policy that never leaves a tile
    planted overnight (measured: 50 draws per end-of-day for two players). A
    policy that holds crops across midnight, buys land, or lets weeds stand
    changes the count, and the schedule then diverges — hence
    :func:`shop_demand_table` takes the count as a parameter.
    """
    active: list[str] = []
    per_day: list[list[str]] = []
    names = sorted(SHOPS)
    del board_size  # kept for call-site clarity; the count is passed explicitly

    for day in range(days):
        per_day.append(list(active))
        rng = random.Random((seed * 1_000_003) ^ day)

        # Advance the stream exactly as _spawn_weeds does before the draw.
        for _ in range(n_players * empty_tiles_per_day):
            rng.random()

        next_day = day + 1
        if next_day > 0 and next_day % SHOP_UNLOCK_INTERVAL == 0:
            if len(active) < MAX_SHOP_INSTANCES:
                active.append(rng.choice(names))

    return per_day


def shop_demand_table(
    seed: int, days: int, *, empty_tiles_per_day: int = 25
) -> jnp.ndarray:
    """Per-day shop consumption per product, as ``(days, N_PRODUCTS)`` int32.

    A single-product shop consumes 2 per tick; multi-product shops consume 1 of
    each. This is the amount removed on every ``SHOP_SELL_INTERVAL`` turn.
    """
    per_day = unlocked_shops_by_day(seed, days, empty_tiles_per_day=empty_tiles_per_day)
    table = [[0] * N_PRODUCTS for _ in range(days)]

    for day, shops in enumerate(per_day):
        for shop in shops:
            products = SHOPS[shop]
            multiplier = 2 if len(products) == 1 else 1
            for item in products:
                table[day][_PRODUCT_INDEX[item]] += multiplier

    return jnp.array(table, dtype=jnp.int32)
