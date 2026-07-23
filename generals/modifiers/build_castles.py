"""Build-castles modifier: players BUILD castles instead of capturing them.

Rules:
  - A third action kind: [2, row, col, _, _] builds a castle at (row, col).
    The pass field value 2 keeps the (2, 5) action shape, so existing agents,
    replays and the stdio protocol are unaffected (they never emit 2).
  - Cost: 35 + sum over your existing structures (your general + every castle
    you own) of max(0, 10 - 2 * manhattan_distance(cell, structure)). Building
    in fresh territory costs 35; crowding your own structures gets expensive
    (adjacent to your general: 43). Structures 5+ cells away add nothing.
    Enemy structures never affect your price; a captured castle counts as
    yours from then on. Prices are dynamic — every castle you gain raises
    them nearby — so costs are computed from the live state each step.
  - The cell must be owned by the builder, plain land (no general, no castle),
    and hold at least `cost` army. Cost is deducted; the remainder stays on
    the new castle — building can leave the cell at 0 army, snipeable by a
    single unit.
  - An invalid build is consumed as a pass, mirroring how the base game
    treats invalid moves as no-ops.
  - Builds resolve BEFORE either player's move each tick, then the base
    game.step runs with the build actions rewritten to passes. The base game
    is never modified. A built castle is an ordinary owned castle: +1 army on
    even ticks, just like a captured one.

Maps for this mode carry no neutral castles: GeneralsEnv(build_castles=True)
strips them from generated grids via strip_neutral_castles().
"""
import jax
import jax.numpy as jnp

from generals.core import game

BUILD = 2            # action pass-field value meaning "build at (row, col)"
BASE_COST = 35
PROXIMITY_PENALTY = 10   # surcharge at distance 0, fading by...
PROXIMITY_DECAY = 2      # ...this much per manhattan step (zero from d=5)
_RADIUS = (PROXIMITY_PENALTY - 1) // PROXIMITY_DECAY  # farthest d with a surcharge


def strip_neutral_castles(grid: jnp.ndarray) -> jnp.ndarray:
    """Remove neutral castles from a generated grid (values > 2 are castles)."""
    return jnp.where(grid > 2, 0, grid)


def build_cost_grid(state: game.GameState, player_idx: int) -> jnp.ndarray:
    """(H, W) castle price per cell for `player_idx`, from the live state.

    35 everywhere, plus max(0, 10 - 2d) for each own structure at manhattan
    distance d. Computed as a sum of shifted copies of the structure mask —
    the surcharge kernel only spans d <= _RADIUS, so this is ~40 cheap adds.
    """
    H, W = state.armies.shape
    own = state.ownership[player_idx]
    structures = ((state.castles | state.generals) & own).astype(jnp.int32)
    padded = jnp.pad(structures, _RADIUS)

    cost = jnp.full((H, W), BASE_COST, dtype=jnp.int32)
    for di in range(-_RADIUS, _RADIUS + 1):
        for dj in range(-_RADIUS, _RADIUS + 1):
            surcharge = PROXIMITY_PENALTY - PROXIMITY_DECAY * (abs(di) + abs(dj))
            if surcharge > 0:
                shifted = padded[_RADIUS + di:_RADIUS + di + H, _RADIUS + dj:_RADIUS + dj + W]
                cost = cost + surcharge * shifted
    return cost


def _apply_one(state: game.GameState, player_idx: int, action: jnp.ndarray) -> game.GameState:
    H, W = state.armies.shape
    is_build = action[0] == BUILD
    r, c = action[1], action[2]
    in_bounds = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    rs = jnp.clip(r, 0, H - 1)
    cs = jnp.clip(c, 0, W - 1)

    owns = state.ownership[player_idx, rs, cs]  # implies passable, not enemy/neutral
    plain = ~state.generals[rs, cs] & ~state.castles[rs, cs]
    cost = build_cost_grid(state, player_idx)[rs, cs]
    affords = state.armies[rs, cs] >= cost
    alive = state.winner < 0
    valid = is_build & in_bounds & owns & plain & affords & alive

    return state._replace(
        armies=jnp.where(valid, state.armies.at[rs, cs].add(-cost), state.armies),
        castles=jnp.where(valid, state.castles.at[rs, cs].set(True), state.castles),
    )


@jax.jit
def apply_build_actions(state: game.GameState,
                        actions: jnp.ndarray) -> tuple[game.GameState, jnp.ndarray]:
    """Resolve both players' build actions; rewrite them to passes.

    Builds target the builder's own cells and prices depend only on the
    builder's own structures, so the two players can never interact and
    resolution order doesn't matter. Every action with pass-field == BUILD
    (valid or not) comes back as a plain pass, so the base game never sees
    the value 2.
    """
    state = _apply_one(state, 0, actions[0])
    state = _apply_one(state, 1, actions[1])

    pass_action = jnp.array([1, 0, 0, 0, 0], dtype=actions.dtype)
    is_build = (actions[:, 0] == BUILD)[:, None]
    return state, jnp.where(is_build, pass_action[None, :], actions)


@jax.jit
def step(state: game.GameState, actions: jnp.ndarray) -> tuple[game.GameState, game.GameInfo]:
    """Drop-in replacement for game.step under the build-castles modifier."""
    state, actions = apply_build_actions(state, actions)
    return game.step(state, actions)
