"""Build-castles modifier: players BUILD castles instead of capturing them.

Rules:
  - A third action kind: [2, row, col, _, _] builds a castle at (row, col).
    The pass field value 2 keeps the (2, 5) action shape, so existing agents,
    replays and the stdio protocol are unaffected (they never emit 2).
  - Cost: max(30, 56 - d), where d is the BFS walking distance from the
    builder's OWN general to the cell (mountains impassable). Distances are
    static for a whole game (generals and mountains never move), so compute
    them once per game with compute_build_distances() and pass them to step().
  - The cell must be owned by the builder, plain land (no general, no castle),
    and hold at least `cost` army. Cost is deducted; the remainder stays on
    the new castle — building can leave the cell at 0 army, snipeable by a
    single unit.
  - An invalid build is consumed as a pass, mirroring how the base game
    treats invalid moves as no-ops.
  - Builds resolve BEFORE either player's move each tick, then the base
    game.step runs with the build actions rewritten to passes. The base game
    is never modified. A built castle is an ordinary owned city: +1 army on
    even ticks, just like a captured one.

Maps for this mode carry no neutral castles: GeneralsEnv(build_castles=True)
strips them from generated grids via strip_neutral_cities().
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from generals.core import game

BUILD = 2          # action pass-field value meaning "build at (row, col)"
MIN_COST = 30
COST_BASE = 56
UNREACHABLE = jnp.int32(2**15)  # BFS sentinel; far larger than any path length


def strip_neutral_cities(grid: jnp.ndarray) -> jnp.ndarray:
    """Remove neutral castles from a generated grid (values > 2 are cities)."""
    return jnp.where(grid > 2, 0, grid)


def _distance_field(passable: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
    """BFS distance from `pos` to every cell over passable terrain.

    Min-plus relaxation to a fixed point: each sweep lets distances flow to the
    4-neighbours, so it converges in (longest shortest path) sweeps. Cells cut
    off by mountains keep UNREACHABLE (they can never be owned, so their build
    cost is irrelevant).
    """
    dist = jnp.full(passable.shape, UNREACHABLE, dtype=jnp.int32)
    dist = dist.at[pos[0], pos[1]].set(0)

    def neighbour_min(d):
        up = jnp.roll(d, -1, axis=0).at[-1, :].set(UNREACHABLE)
        down = jnp.roll(d, 1, axis=0).at[0, :].set(UNREACHABLE)
        left = jnp.roll(d, -1, axis=1).at[:, -1].set(UNREACHABLE)
        right = jnp.roll(d, 1, axis=1).at[:, 0].set(UNREACHABLE)
        return jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right))

    def body(carry):
        d, _ = carry
        relaxed = jnp.minimum(d, neighbour_min(d) + 1)
        relaxed = jnp.where(passable, relaxed, UNREACHABLE)
        relaxed = relaxed.at[pos[0], pos[1]].set(0)
        return relaxed, jnp.any(relaxed != d)

    dist, _ = lax.while_loop(lambda c: c[1], body, (dist, jnp.bool_(True)))
    return dist


@jax.jit
def compute_build_distances(state: game.GameState) -> jnp.ndarray:
    """(2, H, W) BFS distance from each player's general. Compute ONCE per game."""
    return jnp.stack([
        _distance_field(state.passable, state.general_positions[0]),
        _distance_field(state.passable, state.general_positions[1]),
    ])


def build_costs(build_dists: jnp.ndarray) -> jnp.ndarray:
    """Castle cost per cell: max(30, 56 - distance from own general)."""
    return jnp.maximum(MIN_COST, COST_BASE - build_dists).astype(jnp.int32)


def _apply_one(state: game.GameState, player_idx: int, action: jnp.ndarray,
               build_dists: jnp.ndarray) -> game.GameState:
    H, W = state.armies.shape
    is_build = action[0] == BUILD
    r, c = action[1], action[2]
    in_bounds = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    rs = jnp.clip(r, 0, H - 1)
    cs = jnp.clip(c, 0, W - 1)

    owns = state.ownership[player_idx, rs, cs]  # implies passable, not enemy/neutral
    plain = ~state.generals[rs, cs] & ~state.cities[rs, cs]
    cost = jnp.maximum(MIN_COST, COST_BASE - build_dists[player_idx, rs, cs])
    affords = state.armies[rs, cs] >= cost
    alive = state.winner < 0
    valid = is_build & in_bounds & owns & plain & affords & alive

    return state._replace(
        armies=jnp.where(valid, state.armies.at[rs, cs].add(-cost), state.armies),
        cities=jnp.where(valid, state.cities.at[rs, cs].set(True), state.cities),
    )


@jax.jit
def apply_build_actions(state: game.GameState, actions: jnp.ndarray,
                        build_dists: jnp.ndarray) -> tuple[game.GameState, jnp.ndarray]:
    """Resolve both players' build actions; rewrite them to passes.

    Builds target the builder's own cells, so the two players can never
    conflict and resolution order doesn't matter. Every action with
    pass-field == BUILD (valid or not) comes back as a plain pass, so the
    base game never sees the value 2.
    """
    state = _apply_one(state, 0, actions[0], build_dists)
    state = _apply_one(state, 1, actions[1], build_dists)

    pass_action = jnp.array([1, 0, 0, 0, 0], dtype=actions.dtype)
    is_build = (actions[:, 0] == BUILD)[:, None]
    return state, jnp.where(is_build, pass_action[None, :], actions)


@jax.jit
def step(state: game.GameState, actions: jnp.ndarray,
         build_dists: jnp.ndarray) -> tuple[game.GameState, game.GameInfo]:
    """Drop-in replacement for game.step under the build-castles modifier."""
    state, actions = apply_build_actions(state, actions, build_dists)
    return game.step(state, actions)
