"""
JAX game logic for Generals.io.

This module contains the core game mechanics including state management,
action execution, and observation generation. All functions are JIT-compiled
for maximum performance.

The engine is parameterized by the number of players N (a JIT-static
dimension) and a team assignment. Free-for-all is N teams of one
(teams = arange(N), the default); team modes put several players on one
team_id (2v2 is teams = [0, 0, 1, 1]). With the defaults the game is the
classic 1v1: ownership is (2, H, W), actions are (2, 5), winner is 0 or 1.

Key functions:
    - create_initial_state: Create a new game from a grid (and optional teams)
    - step: Execute one game step with actions from all players
    - get_observation: Get a player's view with (team-shared) fog of war
"""
from functools import partial
from typing import Tuple, NamedTuple, Protocol, Any

import jax
import jax.numpy as jnp
from jax import lax

from generals.core.observation import Observation


class Game(Protocol):
    """Protocol for game objects used by the GUI."""
    agents: list[str]
    channels: Any
    grid_dims: tuple[int, int]
    general_positions: dict[str, Any]
    time: int

    def get_infos(self) -> dict[str, dict[str, Any]]:
        """Return player stats."""
        ...


class GameState(NamedTuple):
    """
    Immutable game state containing all information about the game.

    Attributes:
        armies: (H, W) array of army counts per cell.
        ownership: (N, H, W) boolean arrays, ownership[i] is player i's cells.
        ownership_neutral: (H, W) boolean mask of neutral (unowned) cells.
        generals: (H, W) boolean mask of live general positions. A captured
            general turns into a castle and leaves this mask.
        castles: (H, W) boolean mask of castle positions.
        mountains: (H, W) boolean mask of mountain positions.
        passable: (H, W) boolean mask of passable cells (not mountains).
        general_positions: (N, 2) array of [row, col] where each general started.
        teams: (N,) int32 array, teams[i] is the team id of player i. Every
            team id is a player index in the default free-for-all.
        eliminated: (N,) bool array, True once player i's general has been
            captured (their territory is gone and their actions are ignored).
        time: Scalar, current game timestep.
        winner: Scalar, -1 if game ongoing, otherwise the team id of the
            last team standing (== the player index in 1v1 / free-for-all).
        pool_idx: Scalar, index into the pre-generated state pool for auto-reset.
    """

    armies: jnp.ndarray
    ownership: jnp.ndarray
    ownership_neutral: jnp.ndarray
    generals: jnp.ndarray
    castles: jnp.ndarray
    mountains: jnp.ndarray
    passable: jnp.ndarray
    general_positions: jnp.ndarray
    teams: jnp.ndarray
    eliminated: jnp.ndarray
    time: jnp.ndarray
    winner: jnp.ndarray
    pool_idx: jnp.ndarray

    @property
    def cities(self):
        """Deprecated alias — castles were renamed from cities."""
        return self.castles

    @property
    def num_players(self) -> int:
        return self.ownership.shape[-3]


class GameInfo(NamedTuple):
    """
    Game statistics returned after each step.

    Attributes:
        army: (N,) array of total army counts per player.
        land: (N,) array of total land counts per player.
        is_done: Boolean, True if game has ended.
        winner: -1 if ongoing, otherwise the winning team id.
        time: Current game timestep.
    """

    army: jnp.ndarray
    land: jnp.ndarray
    is_done: jnp.ndarray
    winner: jnp.ndarray
    time: jnp.ndarray


# Direction offsets: UP, DOWN, LEFT, RIGHT
DIRECTIONS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


def create_initial_state(grid: jnp.ndarray, teams=None, num_players: int | None = None) -> GameState:
    """
    Create initial game state from a numeric grid.

    Args:
        grid: 2D array with cell values:
            - -2: Mountain (impassable)
            - 0: Empty cell
            - k in 1..N: Player (k-1)'s general
            - > N (typically 20-50): Castle with that army value
        teams: Optional (N,) array of team ids, one per player. Defaults to
            free-for-all, teams = arange(N).
        num_players: N when `teams` is not given. Defaults to 2.

    Returns:
        GameState ready for gameplay. A player whose general is missing from
        the grid starts out eliminated.
    """
    if teams is None:
        N = 2 if num_players is None else int(num_players)
        teams = jnp.arange(N, dtype=jnp.int32)
    else:
        teams = jnp.asarray(teams, dtype=jnp.int32)
        N = int(teams.shape[0])
        if num_players is not None and int(num_players) != N:
            raise ValueError(f"num_players={num_players} does not match teams of length {N}")

    ownership = jnp.stack([grid == (i + 1) for i in range(N)])
    generals = jnp.any(ownership, axis=0)

    mountains = grid == -2
    passable = grid != -2
    castles = grid > N

    ownership_neutral = passable & ~generals

    armies = jnp.where(generals, 1, 0).astype(jnp.int32)
    armies = jnp.where(castles, grid, armies)

    general_positions = jnp.stack([
        jnp.argwhere(ownership[i], size=1, fill_value=-1)[0] for i in range(N)
    ])

    return GameState(
        armies=armies,
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        generals=generals,
        castles=castles,
        mountains=mountains,
        passable=passable,
        general_positions=general_positions,
        teams=teams,
        eliminated=~jnp.any(ownership, axis=(1, 2)),
        time=jnp.int32(0),
        winner=jnp.int32(-1),
        pool_idx=jnp.int32(0),
    )


@jax.jit
def get_visibility(ownership: jnp.ndarray) -> jnp.ndarray:
    """Compute visibility mask (3x3 around owned cells)."""
    H, W = ownership.shape
    ownership_float = ownership.astype(jnp.float32)
    padded = jnp.pad(ownership_float, 1, mode="constant", constant_values=0)

    stacked = jnp.stack(
        [
            padded[0:H, 0:W],
            padded[0:H, 1 : W + 1],
            padded[0:H, 2 : W + 2],
            padded[1 : H + 1, 0:W],
            padded[1 : H + 1, 1 : W + 1],
            padded[1 : H + 1, 2 : W + 2],
            padded[2 : H + 2, 0:W],
            padded[2 : H + 2, 1 : W + 1],
            padded[2 : H + 2, 2 : W + 2],
        ],
        axis=0,
    )

    return jnp.max(stacked, axis=0) > 0


@partial(jax.jit, static_argnames=("spoils",))
def execute_action(state: GameState, player_idx: int, action: jnp.ndarray, spoils: bool = True) -> GameState:
    """Execute a single player's action.

    Args:
        spoils: With the default True a general capture is settled in full —
            the captured player's territory passes to the capturer (see
            eliminate_player). With False only the move itself is resolved:
            the tile changes hands and the general becomes a castle, but the
            territory, elimination and win are left unsettled. Modifiers use
            this to judge the other players' moves on the board as the move
            left it, before the spoils confiscated anyone's army.
    """
    pass_turn, si, sj, direction, split_army = action

    return lax.cond(
        pass_turn == 1,
        lambda s: s,
        lambda s: _execute_move(s, player_idx, si, sj, direction, split_army, spoils),
        state,
    )


def _execute_move(state: GameState, player_idx: int, si: int, sj: int, direction: int, split_army: int,
                  spoils: bool = True) -> GameState:
    """Execute move logic."""
    H, W = state.armies.shape

    in_bounds = (si >= 0) & (si < H) & (sj >= 0) & (sj < W)

    di = si + DIRECTIONS[direction, 0]
    dj = sj + DIRECTIONS[direction, 1]
    dest_in_bounds = (di >= 0) & (di < H) & (dj >= 0) & (dj < W)

    owns_source = state.ownership[player_idx, si, sj]
    source_army = state.armies[si, sj]

    army_to_move = lax.cond(split_army == 1, lambda a: a // 2, lambda a: a - 1, source_army)
    army_to_move = jnp.maximum(0, jnp.minimum(army_to_move, source_army - 1))

    # An eliminated player owns nothing, so owns_source already fails; the
    # explicit check keeps hand-built states honest too.
    valid_move = (in_bounds & dest_in_bounds & owns_source & (army_to_move > 0)
                  & state.passable[di, dj] & ~state.eliminated[player_idx])

    return lax.cond(
        valid_move,
        lambda s: _apply_move(s, player_idx, si, sj, di, dj, army_to_move, spoils),
        lambda s: s,
        state,
    )


def _apply_move(state: GameState, player_idx: int, si: int, sj: int, di: int, dj: int, army_to_move: int,
                spoils: bool = True) -> GameState:
    """Apply a validated move.

    Three outcomes, decided by who holds the destination:
      - Friendly (the mover or a teammate): armies pool on the destination and
        it becomes the mover's cell.
      - Enemy or neutral: an attack; the larger force keeps the difference and
        a won attack flips the cell to the mover.
      - Enemy general: a won attack is a capture. The tile keeps the attacker's
        surplus and turns into a castle; then (spoils) eliminate_player hands
        the rest of the captured player's territory to the mover with every
        army halved, rounded up.
    """
    armies = state.armies
    ownership = state.ownership
    ownership_neutral = state.ownership_neutral
    N = ownership.shape[0]
    players = jnp.arange(N)

    mover = players == player_idx                                    # (N,) one-hot
    target_owners = ownership[:, di, dj]                             # (N,)
    same_team = state.teams == state.teams[player_idx]               # (N,)
    friendly = jnp.any(target_owners & same_team)
    target_army = armies[di, dj]

    attacker_wins = army_to_move > target_army
    takes_cell = friendly | attacker_wins
    dest_army = jnp.where(friendly, target_army + army_to_move, jnp.abs(target_army - army_to_move))

    armies = armies.at[di, dj].set(dest_army).at[si, sj].add(-army_to_move)
    ownership = ownership.at[:, di, dj].set(jnp.where(takes_cell, mover, target_owners))
    ownership_neutral = ownership_neutral.at[di, dj].set(ownership_neutral[di, dj] & ~takes_cell)

    # A general tile is always owned, so a won attack on one is a capture.
    captured = attacker_wins & ~friendly & state.generals[di, dj] & jnp.any(target_owners)
    captured_idx = jnp.argmax(target_owners)
    generals = state.generals.at[di, dj].set(state.generals[di, dj] & ~captured)
    castles = state.castles.at[di, dj].set(state.castles[di, dj] | captured)

    state = state._replace(
        armies=armies,
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        generals=generals,
        castles=castles,
    )
    if spoils:
        state = lax.cond(
            captured,
            lambda s: eliminate_player(s, captured_idx, player_idx),
            lambda s: s,
            state,
        )
    return state


def eliminate_player(state: GameState, captured_idx, capturer_idx) -> GameState:
    """Settle a general capture: `captured_idx` drops out of the game.

    Every cell the captured player still holds passes to `capturer_idx` with
    its army halved (rounded up, so a lone unit survives); any general the
    captured player still holds becomes a castle; the player is marked
    eliminated. If that leaves only the capturer's team alive, the winner is
    set to that team. Idempotent, so it is safe to apply to an already
    eliminated player.
    """
    N = state.ownership.shape[0]
    players = jnp.arange(N)
    cells = state.ownership[captured_idx]                            # (H, W)
    capturer = players == capturer_idx                               # (N,)

    armies = jnp.where(cells, (state.armies + 1) // 2, state.armies)
    ownership = jnp.where(cells[None], capturer[:, None, None], state.ownership)
    castles = state.castles | (state.generals & cells)
    generals = state.generals & ~cells

    eliminated = state.eliminated | (players == captured_idx)
    capturer_team = state.teams[capturer_idx]
    last_team_standing = jnp.all(eliminated | (state.teams == capturer_team))
    winner = jnp.where((state.winner < 0) & last_team_standing, capturer_team, state.winner)

    return state._replace(
        armies=armies,
        ownership=ownership,
        generals=generals,
        castles=castles,
        eliminated=eliminated,
        winner=winner,
    )


@jax.jit
def global_update(state: GameState) -> GameState:
    """Perform army increments (every 2 turns for structures, every 50 for all)."""
    time = state.time
    armies = state.armies
    owned = jnp.any(state.ownership, axis=0).astype(jnp.int32)

    increment_all = time % 50 == 0
    armies = lax.cond(
        increment_all,
        lambda a: a + owned,
        lambda a: a,
        armies,
    )

    # Generals/castles grow every 2 ticks, on EVEN ticks. generals.io has a spawn
    # frame + a first frame before production starts, so the first increment lands
    # on tick 2, not tick 1. Matching this phase is required to replay real games
    # move-for-move (verified: 0 illegal moves across scraped generals.io replays;
    # the odd-tick phase desynced the general's army by one tick and lost games early).
    increment_structures = (time % 2 == 0)
    structure_mask = (state.generals | state.castles).astype(jnp.int32)
    armies = lax.cond(
        increment_structures,
        lambda a: a + structure_mask * owned,
        lambda a: a,
        armies,
    )

    return state._replace(armies=armies)


def _determine_move_order(state: GameState, actions: jnp.ndarray) -> jnp.ndarray:
    """Order in which this turn's moves resolve: an (N,) array of player indices.

    Priority is chasing > reinforcing > SMALLER army, ties by player index,
    passes last. Chasing: the move lands on the source of another player's
    move. Reinforcing: the move lands on a cell the mover's team holds.
    Smaller army first: on a contested cell the bigger force resolves last and
    ends up holding it (larger-first let the smaller force snipe a neutral
    castle the bigger one had just paid for), and a deathtouch head-on clash
    goes to the attacker, keeping the endgame a forced finish.

    For two players this is exactly the old first-mover rule; it is computed
    with pairwise comparisons rather than a sort, so it costs a few (N, N)
    boolean ops.
    """
    N = actions.shape[0]
    H, W = state.armies.shape
    idx = jnp.arange(N)

    passes = actions[:, 0] != 0
    si, sj, direction = actions[:, 1], actions[:, 2], actions[:, 3]
    di = si + DIRECTIONS[direction, 0]
    dj = sj + DIRECTIONS[direction, 1]

    # chasing[i]: i's destination is the source of some other player's move
    onto_source = (di[:, None] == si[None, :]) & (dj[:, None] == sj[None, :])
    chasing = jnp.any(onto_source & ~passes[None, :] & ~jnp.eye(N, dtype=bool), axis=1)

    # reinforcing[i]: i's destination is held by i's team. Out-of-bounds
    # destinations are clipped; such a move is invalid and never executes, so
    # its slot in the order is irrelevant.
    ci, cj = jnp.clip(di, 0, H - 1), jnp.clip(dj, 0, W - 1)
    dest_owners = state.ownership[:, ci, cj]                          # (owner, mover)
    same_team = state.teams[:, None] == state.teams[None, :]          # (owner, mover)
    reinforcing = jnp.any(dest_owners & same_team, axis=0)

    army = state.armies[jnp.clip(si, 0, H - 1), jnp.clip(sj, 0, W - 1)]

    c = chasing & ~passes
    r = reinforcing & ~passes
    a = jnp.where(passes, jnp.iinfo(jnp.int32).max, army)

    # ahead[j, i]: j resolves before i (lexicographic: c desc, r desc, a asc, index asc)
    cj_, ci_ = c[:, None], c[None, :]
    rj_, ri_ = r[:, None], r[None, :]
    aj_, ai_ = a[:, None], a[None, :]
    by_index = idx[:, None] < idx[None, :]
    ahead = (cj_ & ~ci_) | ((cj_ == ci_) & ((rj_ & ~ri_) | ((rj_ == ri_) & ((aj_ < ai_) | ((aj_ == ai_) & by_index)))))

    rank = jnp.sum(ahead, axis=0)                                     # players ahead of each i
    return jnp.argmax(rank[None, :] == idx[:, None], axis=1)          # slot k -> player with rank k


@jax.jit
def step(state: GameState, actions: jnp.ndarray) -> tuple[GameState, GameInfo]:
    """Execute one game step with actions from all players.

    Args:
        state: Current game state.
        actions: (N, 5) array, one [pass, row, col, direction, split] per player.

    Moves resolve one after another in _determine_move_order's order, each on
    the board the previous one left — a capture confiscates the captured
    player's territory immediately, so their later move (and every move after
    the game is decided) finds no army to command.
    """
    N = actions.shape[0]
    if N != state.ownership.shape[0]:
        raise ValueError(f"got actions for {N} players but the state has {state.ownership.shape[0]}")
    done_before = state.winner >= 0

    order = _determine_move_order(state, actions)
    for k in range(N):
        player = order[k]
        state = execute_action(state, player, actions[player])

    state = lax.cond(done_before, lambda s: s, lambda s: s._replace(time=s.time + 1), state)

    state = lax.cond(
        state.winner >= 0,
        lambda s: s,
        lambda s: global_update(s),
        state,
    )

    return state, get_info(state)


@jax.jit
def get_info(state: GameState) -> GameInfo:
    """Compute game statistics."""
    armies = state.armies
    ownership = state.ownership

    return GameInfo(
        army=jnp.sum(armies[None] * ownership, axis=(1, 2)),
        land=jnp.sum(ownership, axis=(1, 2)),
        is_done=state.winner >= 0,
        winner=state.winner,
        time=state.time,
    )


def team_cells(state: GameState, player_idx) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """(own, allied, enemy) cell masks for `player_idx`. Allied excludes self;
    in 1v1 / free-for-all it is empty and enemy is everyone else's land."""
    same_team = state.teams == state.teams[player_idx]
    own = state.ownership[player_idx]
    team = jnp.any(state.ownership & same_team[:, None, None], axis=0)
    enemy = jnp.any(state.ownership & ~same_team[:, None, None], axis=0)
    return own, team & ~own, enemy


def _observe(state: GameState, player_idx, fog: bool) -> Observation:
    N = state.ownership.shape[0]
    players = jnp.arange(N)
    same_team = state.teams == state.teams[player_idx]
    teammate = same_team & (players != player_idx)

    own_cells, allied_cells, enemy_cells = team_cells(state, player_idx)
    if fog:
        # Sight is shared within a team: 3x3 around any cell the team holds.
        visible = get_visibility(own_cells | allied_cells)
    else:
        visible = jnp.ones_like(own_cells)
    invisible = ~visible

    info = get_info(state)
    structures = state.mountains | state.castles

    return Observation(
        armies=state.armies * visible,
        generals=state.generals * visible,
        castles=state.castles * visible,
        mountains=state.mountains * visible,
        neutral_cells=state.ownership_neutral * visible,
        owned_cells=own_cells * visible,
        opponent_cells=enemy_cells * visible,
        fog_cells=invisible & ~structures,
        structures_in_fog=invisible & structures,
        owned_land_count=info.land[player_idx],
        owned_army_count=info.army[player_idx],
        opponent_land_count=jnp.sum(jnp.where(same_team, 0, info.land)),
        opponent_army_count=jnp.sum(jnp.where(same_team, 0, info.army)),
        timestep=state.time,
        allied_cells=allied_cells * visible,
        allied_land_count=jnp.sum(jnp.where(teammate, info.land, 0)),
        allied_army_count=jnp.sum(jnp.where(teammate, info.army, 0)),
    )


@jax.jit
def get_observation(state: GameState, player_idx: int) -> Observation:
    """Get player observation with fog of war applied.

    Fog is team-shared: the player sees the 3x3 neighbourhood of every cell
    their team holds. `opponent_*` covers every enemy team together and
    `allied_*` the teammates (empty in 1v1 / free-for-all).
    """
    return _observe(state, player_idx, fog=True)


@jax.jit
def get_full_observation(state: GameState, player_idx: int) -> Observation:
    """Get a player's observation with NO fog of war — everything is visible.

    Used for perfect-information variants of the competition. fog_cells and
    structures_in_fog masks are all-zeros; every other field reflects the
    true game state.
    """
    return _observe(state, player_idx, fog=False)


@jax.jit
def batch_step(states: GameState, actions: jnp.ndarray) -> Tuple[GameState, GameInfo]:
    """Vectorized step for multiple environments."""
    return jax.vmap(step)(states, actions)
