"""Batched kaggriculture state as flat JAX arrays (template; see README.md).

The official engine holds state as nested Python dicts with per-tile ``dict``
objects. That shape cannot go on an accelerator, so this module re-expresses the
same information as a struct-of-arrays with a leading batch axis ``B``: every
field is a dense array and all ``B`` environments advance in lockstep under one
:func:`jax.jit`-compiled step.

Layout choices that matter:

- **Tiles are planes, not objects.** A tile's ``kind`` / ``crop`` /
  ``planted_day`` / ``watered_today`` / ``consecutive_unwatered`` /
  ``yield_units`` / ``max_lifespan_step`` / ``fertilized_until_day`` each become
  their own ``(B, H, W)`` plane. Branching per tile becomes ``jnp.where``.
- **No Python control flow on traced values.** Every rule is a masked update, so
  the same instruction stream runs for all environments regardless of what any
  individual board contains.
- **int32 throughout** for counts and days; float32 only for money, matching the
  engine's integer-rounded prices.

This is a reimplementation, not a wrapper, so it carries divergence risk that
the ``fast`` engine does not. Equivalence against the official engine is
asserted in ``tests/`` and is the only thing that makes this module usable.

``jax`` needs no entry in ``pyproject.toml``: ``kaggle-environments`` already
requires it directly, so it is present wherever the engine is. Note that
installing ``jax-metal`` into the project venv is actively harmful — it takes
over ``jax.devices()`` for every test, and it runs this workload roughly 100x
*slower* than CPU (see ``docs/plans/simulator/jax-gpu.md``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

#: Tile kind codes. EMPTY must be 0 so a zeroed board is empty.
KIND_EMPTY = 0
KIND_PLANT = 1
KIND_WEED = 2
KIND_LOCKED = 3
KIND_COOP = 4
KIND_PASTURE = 5

#: Crop codes, ordered to match CROP_* tables below.
CROP_NONE = -1
CROP_WHEAT = 0
CROP_CARROT = 1
CROP_TOMATO = 2
CROP_STRAWBERRY = 3
CROP_MELON = 4

CROP_NAMES = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")

#: Per-crop constants, indexed by crop code. Mirrors ``CROPS`` in the engine.
CROP_SEED_COST = jnp.array([10, 20, 50, 100, 80], dtype=jnp.int32)
CROP_FIRST_YIELD_DAY = jnp.array([2, 2, 8, 10, 10], dtype=jnp.int32)
CROP_MAX_YIELD_DAY = jnp.array([4, 3, 8, 10, 12], dtype=jnp.int32)
CROP_INTERVAL = jnp.array([0, 0, 1, 2, 0], dtype=jnp.int32)
CROP_MAX_YIELD = jnp.array([6, 4, 4, 4, 6], dtype=jnp.int32)
CROP_ONGOING = jnp.array([0, 0, 1, 1, 0], dtype=jnp.int32)

#: Product codes. Crops occupy 0..4 so a crop code doubles as a product code.
PRODUCT_NAMES = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
N_PRODUCTS = len(PRODUCT_NAMES)

#: Market shape function codes.
SHAPE_LINEAR = 0
SHAPE_SQ = 1
SHAPE_SQRT = 2
SHAPE_LOG = 3
SHAPE_LOG10 = 4

_SHAPE_CODE = {
    "linear": SHAPE_LINEAR,
    "sq": SHAPE_SQ,
    "sqrt": SHAPE_SQRT,
    "log": SHAPE_LOG,
    "log10": SHAPE_LOG10,
}

#: MARKET_PARAMS from the engine, transposed into parallel arrays.
_RAW_MARKET = (
    # base,   T,  below_func, below_target, above_func, above_target
    (25, 400, "sqrt", 0.80, "log", 0.20),  # WHEAT
    (35, 450, "log", 0.20, "sqrt", 0.70),  # CARROT
    (60, 200, "linear", 0.40, "sqrt", 0.60),  # TOMATO
    (120, 100, "sqrt", 0.70, "linear", 1.60),  # STRAWBERRY
    (250, 300, "log", 0.20, "sq", 3.60),  # MELON
    (50, 332, "linear", 0.40, "log", 0.20),  # EGG
    (160, 122, "sqrt", 0.60, "linear", 1.60),  # MILK
    (200, 105, "log", 0.20, "sq", 3.20),  # WOOL
    (100, 200, "linear", 0.40, "linear", 0.40),  # FERTILIZER
)

MARKET_BASE = jnp.array([r[0] for r in _RAW_MARKET], dtype=jnp.float32)
MARKET_T = jnp.array([r[1] for r in _RAW_MARKET], dtype=jnp.float32)
MARKET_BELOW_FUNC = jnp.array([_SHAPE_CODE[r[2]] for r in _RAW_MARKET], dtype=jnp.int32)
MARKET_BELOW_TARGET = jnp.array([r[3] for r in _RAW_MARKET], dtype=jnp.float32)
MARKET_ABOVE_FUNC = jnp.array([_SHAPE_CODE[r[4]] for r in _RAW_MARKET], dtype=jnp.int32)
MARKET_ABOVE_TARGET = jnp.array([r[5] for r in _RAW_MARKET], dtype=jnp.float32)

MARKET_I0 = 10_000
PRICE_FLOOR = 1

BOARD_SIZE = 10
STARTING_MONEY = 3000
SHED_CAPACITY = 100
TURNS_PER_DAY = 24


class EnvState(NamedTuple):
    """Batched state. Leading axis is the environment index ``B``.

    Player-indexed fields carry a ``2`` axis directly after the batch axis.
    """

    step: jnp.ndarray  # (B,) int32
    money: jnp.ndarray  # (B, 2) float32
    # Tile planes, per player: (B, 2, H, W)
    kind: jnp.ndarray
    crop: jnp.ndarray
    planted_day: jnp.ndarray
    watered_today: jnp.ndarray
    consecutive_unwatered: jnp.ndarray
    yield_units: jnp.ndarray
    max_lifespan_step: jnp.ndarray
    # Unit positions: (B, 2) for the farmer; hands are not modelled yet.
    farmer_x: jnp.ndarray
    farmer_y: jnp.ndarray
    # Inventories: (B, 2, N_PRODUCTS) int32
    shed: jnp.ndarray
    carried: jnp.ndarray
    seeds: jnp.ndarray  # (B, 2, 5) int32
    # Shared market inventory: (B, N_PRODUCTS) int32
    market_inv: jnp.ndarray

    @property
    def batch_size(self) -> int:
        return int(self.step.shape[0])

    @property
    def day(self) -> jnp.ndarray:
        return self.step // TURNS_PER_DAY

    @property
    def hour(self) -> jnp.ndarray:
        return self.step % TURNS_PER_DAY


def _shape_fn(code: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Vectorised ``_shape``: select by code rather than branching."""
    x = jnp.maximum(x, 0.0)
    # Guard the inputs of sqrt/log so no branch produces NaN/inf before select.
    safe = jnp.maximum(x, 1e-12)
    options = jnp.stack(
        [
            x,
            x * x,
            jnp.sqrt(safe),
            jnp.log1p(x),
            jnp.log10(1.0 + x),
        ]
    )
    return jnp.take_along_axis(options, code[None, ...], axis=0)[0]


def market_price(inventory: jnp.ndarray) -> jnp.ndarray:
    """Price for every product from its inventory.

    ``inventory`` is ``(..., N_PRODUCTS)``; the result has the same shape.
    Mirrors the engine exactly, including ``int(round(price))`` and the floor.
    """
    inv = inventory.astype(jnp.float32)
    below = inv < MARKET_I0
    delta = jnp.where(below, MARKET_I0 - inv, inv - MARKET_I0)

    func = jnp.where(below, MARKET_BELOW_FUNC, MARKET_ABOVE_FUNC)
    target = jnp.where(below, MARKET_BELOW_TARGET, MARKET_ABOVE_TARGET)

    codes = jnp.broadcast_to(func, delta.shape)
    denom = _shape_fn(codes, jnp.broadcast_to(MARKET_T, delta.shape))
    amp = target * MARKET_BASE / denom
    magnitude = amp * _shape_fn(codes, delta)

    price = jnp.where(below, MARKET_BASE + magnitude, MARKET_BASE - magnitude)
    # The engine rounds half away from zero via Python's round-then-int on a
    # positive value; jnp.round is banker's rounding, so add the tie offset.
    rounded = jnp.floor(price + 0.5)
    return jnp.maximum(PRICE_FLOOR, rounded).astype(jnp.int32)


def initial_state(batch_size: int) -> EnvState:
    """Fresh batched state matching the engine's initial board.

    Only the NW quadrant starts unlocked, the farmer spawns at the engine's
    default position, and the market sits at equilibrium.
    """
    b, h, w = batch_size, BOARD_SIZE, BOARD_SIZE
    half = BOARD_SIZE // 2

    ys, xs = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    locked = (xs >= half) | (ys >= half)
    kind = jnp.where(locked, KIND_LOCKED, KIND_EMPTY).astype(jnp.int32)
    kind = jnp.broadcast_to(kind, (b, 2, h, w))

    zeros_tile = jnp.zeros((b, 2, h, w), dtype=jnp.int32)
    spawn = half - 1

    return EnvState(
        step=jnp.zeros((b,), dtype=jnp.int32),
        money=jnp.full((b, 2), STARTING_MONEY, dtype=jnp.float32),
        kind=kind,
        crop=jnp.full((b, 2, h, w), CROP_NONE, dtype=jnp.int32),
        planted_day=zeros_tile,
        watered_today=zeros_tile,
        consecutive_unwatered=zeros_tile,
        yield_units=zeros_tile,
        max_lifespan_step=jnp.full((b, 2, h, w), -1, dtype=jnp.int32),
        farmer_x=jnp.full((b, 2), spawn, dtype=jnp.int32),
        farmer_y=jnp.full((b, 2), spawn, dtype=jnp.int32),
        shed=jnp.zeros((b, 2, N_PRODUCTS), dtype=jnp.int32),
        carried=jnp.zeros((b, 2, N_PRODUCTS), dtype=jnp.int32),
        seeds=jnp.zeros((b, 2, len(CROP_NAMES)), dtype=jnp.int32),
        market_inv=jnp.full((b, N_PRODUCTS), MARKET_I0, dtype=jnp.int32),
    )


def tree_flatten_check(state: EnvState) -> None:
    """Assert every leaf carries the same batch size (cheap sanity guard)."""
    sizes = {leaf.shape[0] for leaf in jax.tree_util.tree_leaves(state)}
    if len(sizes) != 1:
        raise ValueError(f"inconsistent batch sizes across state leaves: {sizes}")
