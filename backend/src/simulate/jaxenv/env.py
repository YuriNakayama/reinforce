"""Batched, jit-compiled kaggriculture step (template carried over; see README.md).

Every rule is expressed as a masked array update so one instruction stream
drives all ``B`` environments. The action encoding is deliberately narrow — the
ops a wheat-loop policy actually uses — because a faithful port of *every* op
(land purchase, hands, animals, fertilizer) is a much larger surface and the
point here is to measure whether GPU batching pays off at all.

Supported unit ops (see :data:`OP_NAMES`): PASS, NORTH, SOUTH, EAST, WEST,
PLANT, WATER, HARVEST, DIG.

Supported market ops: none, SELL, BUY_SEED — one order per turn, which is what
the wheat loop issues.

Scope limits, stated plainly so results are not over-read:

- one farmer per player, no hired hands
- NW quadrant only; ``BUY_LAND`` is not modelled
- no animals, no fertilizer, no town shops
- weed spawning is omitted (it needs per-tile RNG; ``weedSpawnChance`` is 0.005)

Within that subset the dynamics are checked against the official engine.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .state import (
    BOARD_SIZE,
    CROP_FIRST_YIELD_DAY,
    CROP_INTERVAL,
    CROP_MAX_YIELD,
    CROP_MAX_YIELD_DAY,
    CROP_NONE,
    CROP_ONGOING,
    CROP_SEED_COST,
    KIND_EMPTY,
    KIND_LOCKED,
    KIND_PLANT,
    KIND_WEED,
    N_PRODUCTS,
    SHED_CAPACITY,
    TURNS_PER_DAY,
    EnvState,
    market_price,
)

OP_PASS = 0
OP_NORTH = 1
OP_SOUTH = 2
OP_EAST = 3
OP_WEST = 4
OP_PLANT = 5
OP_WATER = 6
OP_HARVEST = 7
OP_DIG = 8

OP_NAMES = (
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PLANT",
    "WATER",
    "HARVEST",
    "DIG",
)

MARKET_NONE = 0
MARKET_SELL = 1
MARKET_BUY_SEED = 2

#: (dx, dy) per op code; y grows downward.
_DX = jnp.array([0, 0, 0, 1, -1, 0, 0, 0, 0], dtype=jnp.int32)
_DY = jnp.array([0, -1, 1, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)


def _gather_tile(plane: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Read ``plane[b, p, y, x]`` for every (b, p). Shapes: (B,2,H,W) -> (B,2)."""
    b, p = x.shape
    bi = jnp.arange(b)[:, None]
    pi = jnp.arange(p)[None, :]
    return plane[bi, pi, y, x]


def _scatter_tile(
    plane: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, value: jnp.ndarray
) -> jnp.ndarray:
    """Write ``value`` into ``plane[b, p, y, x]`` for every (b, p)."""
    b, p = x.shape
    bi = jnp.arange(b)[:, None]
    pi = jnp.arange(p)[None, :]
    return plane.at[bi, pi, y, x].set(value)


def _apply_moves(state: EnvState, op: jnp.ndarray) -> EnvState:
    """Move units, clamping at the board edge (off-board moves are no-ops)."""
    nx = state.farmer_x + _DX[op]
    ny = state.farmer_y + _DY[op]
    in_bounds = (nx >= 0) & (nx < BOARD_SIZE) & (ny >= 0) & (ny < BOARD_SIZE)
    return state._replace(
        farmer_x=jnp.where(in_bounds, nx, state.farmer_x),
        farmer_y=jnp.where(in_bounds, ny, state.farmer_y),
    )


def _apply_plant(
    state: EnvState, op: jnp.ndarray, crop_arg: jnp.ndarray, day: jnp.ndarray
) -> EnvState:
    """PLANT on an empty owned tile, consuming one seed."""
    x, y = state.farmer_x, state.farmer_y
    kind_here = _gather_tile(state.kind, x, y)

    seed_held = jnp.take_along_axis(state.seeds, crop_arg[..., None], axis=-1)[..., 0]
    ok = (op == OP_PLANT) & (kind_here == KIND_EMPTY) & (seed_held > 0)

    # consecutive_unwatered starts at 1: the planting day already counts as
    # unwatered, so a new plant must be watered the same day or it weeds over.
    day_b = jnp.broadcast_to(day[:, None], ok.shape)
    ongoing = CROP_ONGOING[crop_arg]
    lifespan = jnp.where(
        ongoing == 1,
        -1,
        (day_b + CROP_MAX_YIELD_DAY[crop_arg] + 1) * TURNS_PER_DAY,
    )

    new_kind = jnp.where(ok, KIND_PLANT, kind_here)
    seeds = state.seeds - jnp.where(ok, 1, 0)[..., None] * jax.nn.one_hot(
        crop_arg, state.seeds.shape[-1], dtype=jnp.int32
    )

    return state._replace(
        kind=_scatter_tile(state.kind, x, y, new_kind),
        crop=_scatter_tile(
            state.crop, x, y, jnp.where(ok, crop_arg, _gather_tile(state.crop, x, y))
        ),
        planted_day=_scatter_tile(
            state.planted_day,
            x,
            y,
            jnp.where(ok, day_b, _gather_tile(state.planted_day, x, y)),
        ),
        watered_today=_scatter_tile(
            state.watered_today,
            x,
            y,
            jnp.where(ok, 0, _gather_tile(state.watered_today, x, y)),
        ),
        consecutive_unwatered=_scatter_tile(
            state.consecutive_unwatered,
            x,
            y,
            jnp.where(ok, 1, _gather_tile(state.consecutive_unwatered, x, y)),
        ),
        yield_units=_scatter_tile(
            state.yield_units,
            x,
            y,
            jnp.where(
                ok, jnp.where(ongoing == 1, 0, 1), _gather_tile(state.yield_units, x, y)
            ),
        ),
        max_lifespan_step=_scatter_tile(
            state.max_lifespan_step,
            x,
            y,
            jnp.where(ok, lifespan, _gather_tile(state.max_lifespan_step, x, y)),
        ),
        seeds=seeds,
    )


def _apply_water(state: EnvState, op: jnp.ndarray, day: jnp.ndarray) -> EnvState:
    """WATER a plant, adding a yield unit inside the bonus window."""
    x, y = state.farmer_x, state.farmer_y
    kind_here = _gather_tile(state.kind, x, y)
    watered = _gather_tile(state.watered_today, x, y)
    crop_here = _gather_tile(state.crop, x, y)
    planted = _gather_tile(state.planted_day, x, y)
    yields = _gather_tile(state.yield_units, x, y)

    ok = (op == OP_WATER) & (kind_here == KIND_PLANT) & (watered == 0)

    crop_idx = jnp.maximum(crop_here, 0)
    day_b = jnp.broadcast_to(day[:, None], ok.shape)
    age = day_b - planted
    max_yield_day = CROP_MAX_YIELD_DAY[crop_idx]
    window_start = (max_yield_day + 1) // 2
    in_window = (age >= window_start) & (age <= max_yield_day)
    # Only non-ongoing crops gain yield from watering.
    gains = ok & in_window & (CROP_ONGOING[crop_idx] == 0)

    new_yields = jnp.where(
        gains, jnp.minimum(CROP_MAX_YIELD[crop_idx], yields + 1), yields
    )

    return state._replace(
        watered_today=_scatter_tile(
            state.watered_today, x, y, jnp.where(ok, 1, watered)
        ),
        yield_units=_scatter_tile(state.yield_units, x, y, new_yields),
    )


def _apply_harvest(state: EnvState, op: jnp.ndarray, day: jnp.ndarray) -> EnvState:
    """HARVEST a mature plant into the carried inventory, clearing the tile."""
    x, y = state.farmer_x, state.farmer_y
    kind_here = _gather_tile(state.kind, x, y)
    crop_here = _gather_tile(state.crop, x, y)
    planted = _gather_tile(state.planted_day, x, y)
    yields = _gather_tile(state.yield_units, x, y)

    crop_idx = jnp.maximum(crop_here, 0)
    day_b = jnp.broadcast_to(day[:, None], kind_here.shape)
    mature = (day_b - planted) >= CROP_FIRST_YIELD_DAY[crop_idx]
    ok = (op == OP_HARVEST) & (kind_here == KIND_PLANT) & (yields > 0) & mature

    picked = jnp.where(ok, yields, 0)
    gain = picked[..., None] * jax.nn.one_hot(crop_idx, N_PRODUCTS, dtype=jnp.int32)

    ongoing = CROP_ONGOING[crop_idx] == 1
    clears = ok & ~ongoing

    return state._replace(
        kind=_scatter_tile(state.kind, x, y, jnp.where(clears, KIND_EMPTY, kind_here)),
        crop=_scatter_tile(state.crop, x, y, jnp.where(clears, CROP_NONE, crop_here)),
        yield_units=_scatter_tile(state.yield_units, x, y, jnp.where(ok, 0, yields)),
        carried=state.carried + gain,
    )


def _apply_dig(state: EnvState, op: jnp.ndarray) -> EnvState:
    """DIG clears a plant or weed, leaving an empty tile."""
    x, y = state.farmer_x, state.farmer_y
    kind_here = _gather_tile(state.kind, x, y)
    ok = (op == OP_DIG) & ((kind_here == KIND_PLANT) | (kind_here == KIND_WEED))
    return state._replace(
        kind=_scatter_tile(state.kind, x, y, jnp.where(ok, KIND_EMPTY, kind_here)),
        crop=_scatter_tile(
            state.crop, x, y, jnp.where(ok, CROP_NONE, _gather_tile(state.crop, x, y))
        ),
        yield_units=_scatter_tile(
            state.yield_units,
            x,
            y,
            jnp.where(ok, 0, _gather_tile(state.yield_units, x, y)),
        ),
    )


def _process_market(
    state: EnvState,
    market_op: jnp.ndarray,
    market_item: jnp.ndarray,
    market_qty: jnp.ndarray,
) -> EnvState:
    """Settle one order per player in per-unit lockstep.

    The engine quotes both players against the same pre-commit inventory and
    then commits both, one unit at a time. That serial dependence is real, so
    this runs a fixed-length ``fori_loop`` over units rather than trying to
    vectorise the sequence away. The bound is the max quantity any policy can
    request, which keeps the trace shape static.
    """
    max_units = 32

    def unit_step(_: int, carry: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, ...]:
        market_inv, money, shed, seeds, remaining = carry

        prices = market_price(market_inv)  # (B, N_PRODUCTS)
        item_price = jnp.take_along_axis(prices, market_item, axis=-1)  # (B,2)

        active = remaining > 0
        selling = active & (market_op == MARKET_SELL)
        buying = active & (market_op == MARKET_BUY_SEED)

        held = jnp.take_along_axis(shed, market_item[..., None], axis=-1)[..., 0]
        can_sell = selling & (held > 0)

        seed_cost = CROP_SEED_COST[jnp.clip(market_item, 0, 4)]
        can_buy = buying & (money >= seed_cost)

        # Sell: shed decreases, money grows, and market inventory grows unless
        # the sale happened at the $1 floor.
        sell_hot = jax.nn.one_hot(market_item, N_PRODUCTS, dtype=jnp.int32)
        shed = shed - jnp.where(can_sell, 1, 0)[..., None] * sell_hot
        money = money + jnp.where(can_sell, item_price, 0).astype(jnp.float32)
        above_floor = item_price > 1
        inv_delta = jnp.where(can_sell & above_floor, 1, 0)[..., None] * sell_hot
        market_inv = market_inv + inv_delta.sum(axis=1)

        # Buy seed: money decreases, seed count grows. Seeds never touch the
        # market inventory (they are minted, not drawn from the book).
        seed_hot = jax.nn.one_hot(
            jnp.clip(market_item, 0, 4), seeds.shape[-1], dtype=jnp.int32
        )
        money = money - jnp.where(can_buy, seed_cost, 0).astype(jnp.float32)
        seeds = seeds + jnp.where(can_buy, 1, 0)[..., None] * seed_hot

        committed = can_sell | can_buy
        remaining = jnp.where(committed, remaining - 1, 0)
        return market_inv, money, shed, seeds, remaining

    remaining0 = jnp.where(market_op == MARKET_NONE, 0, market_qty)
    market_inv, money, shed, seeds, _ = jax.lax.fori_loop(
        0,
        max_units,
        unit_step,
        (state.market_inv, state.money, state.shed, state.seeds, remaining0),
    )
    return state._replace(market_inv=market_inv, money=money, shed=shed, seeds=seeds)


#: Town centre consumes one of every product except fertilizer.
_TOWN_CENTER_MASK = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=jnp.int32)

#: ``townCenterSellInterval`` default. Read from the engine, not the abstract:
#: the centre consumes exactly one unit and this engine version applies no
#: day-based multiplier, despite what the kaggriculture competition summary said.
_TOWN_CENTER_STEP_INTERVAL = 24


def _town_consume(state: EnvState, step: jnp.ndarray) -> EnvState:
    """Town centre demand. Shops are not modelled (they unlock randomly)."""
    fires = (step % _TOWN_CENTER_STEP_INTERVAL) == 0
    amount = jnp.where(fires, 1, 0)[:, None] * _TOWN_CENTER_MASK
    return state._replace(market_inv=state.market_inv - amount)


def _decay_plants(state: EnvState, step: jnp.ndarray) -> EnvState:
    """Past its lifespan a plant loses a yield unit every other step."""
    mls = state.max_lifespan_step
    step_b = step[:, None, None, None]
    expired = (mls >= 0) & (step_b >= mls) & (((step_b - mls) % 2) == 0)
    is_plant = state.kind == KIND_PLANT

    hit = expired & is_plant
    new_yields = jnp.where(hit, state.yield_units - 1, state.yield_units)
    weeded = hit & (new_yields <= 0)

    return state._replace(
        kind=jnp.where(weeded, KIND_WEED, state.kind),
        crop=jnp.where(weeded, CROP_NONE, state.crop),
        yield_units=jnp.where(weeded, 0, new_yields),
    )


def _end_of_day(state: EnvState, day: jnp.ndarray) -> EnvState:
    """Daily refresh: watering accounting, ongoing production, shed drop."""
    is_plant = state.kind == KIND_PLANT
    was_watered = state.watered_today == 1

    unwatered = jnp.where(was_watered, 0, state.consecutive_unwatered + 1)
    # Two consecutive dry days turns a plant into a weed.
    weeded = is_plant & (unwatered >= 2)

    crop_idx = jnp.maximum(state.crop, 0)
    next_day = (day + 1)[:, None, None, None]
    ongoing = CROP_ONGOING[crop_idx] == 1
    interval = jnp.maximum(CROP_INTERVAL[crop_idx], 1)
    since_first = next_day - state.planted_day - CROP_FIRST_YIELD_DAY[crop_idx]
    produces = (
        is_plant
        & ~weeded
        & ongoing
        & (since_first >= 0)
        & ((since_first % interval) == 0)
    )
    count = since_first // interval + 1
    produces = produces & (count <= CROP_MAX_YIELD[crop_idx])

    yields = jnp.where(
        produces,
        jnp.minimum(CROP_MAX_YIELD[crop_idx], state.yield_units + 1),
        state.yield_units,
    )

    # Carried items drop into the shed, subject to capacity; overflow is lost.
    room = jnp.maximum(0, SHED_CAPACITY - state.shed.sum(axis=-1, keepdims=True))
    carried_total = state.carried.sum(axis=-1, keepdims=True)
    scale = jnp.where(carried_total > 0, jnp.minimum(room, carried_total), 0)
    # Deposit in product order until the room runs out.
    cumulative = jnp.cumsum(state.carried, axis=-1)
    prior = cumulative - state.carried
    deposit = jnp.clip(scale - prior, 0, state.carried)

    half = BOARD_SIZE // 2
    spawn = half - 1

    return state._replace(
        kind=jnp.where(weeded, KIND_WEED, state.kind),
        crop=jnp.where(weeded, CROP_NONE, state.crop),
        yield_units=jnp.where(weeded, 0, yields),
        consecutive_unwatered=jnp.where(is_plant, unwatered, 0),
        watered_today=jnp.zeros_like(state.watered_today),
        shed=state.shed + deposit,
        carried=jnp.zeros_like(state.carried),
        farmer_x=jnp.full_like(state.farmer_x, spawn),
        farmer_y=jnp.full_like(state.farmer_y, spawn),
    )


@jax.jit
def step(
    state: EnvState,
    op: jnp.ndarray,
    crop_arg: jnp.ndarray,
    market_op: jnp.ndarray,
    market_item: jnp.ndarray,
    market_qty: jnp.ndarray,
) -> EnvState:
    """Advance every environment one turn.

    Ordering mirrors the engine: unit actions, then the market, then plant
    decay, then end-of-day when the turn closes a day.
    """
    day = state.step // TURNS_PER_DAY

    # A locked tile blocks tile ops but not movement, so mask ops by ownership.
    owned = _gather_tile(state.kind, state.farmer_x, state.farmer_y) != KIND_LOCKED
    tile_op = jnp.where(owned, op, OP_PASS)

    state = _apply_moves(state, op)
    state = _apply_plant(state, tile_op, crop_arg, day)
    state = _apply_water(state, tile_op, day)
    state = _apply_harvest(state, tile_op, day)
    state = _apply_dig(state, tile_op)

    state = _process_market(state, market_op, market_item, market_qty)
    state = _town_consume(state, state.step)
    state = _decay_plants(state, state.step)

    closes_day = ((state.step + 1) % TURNS_PER_DAY) == 0
    eod = _end_of_day(state, day)
    state = jax.tree_util.tree_map(
        lambda new, old: jnp.where(
            closes_day.reshape((-1,) + (1,) * (new.ndim - 1)), new, old
        ),
        eod,
        state,
    )

    return state._replace(step=state.step + 1)
