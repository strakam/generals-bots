"""
JAX game logic for Generals.io.

This module contains the core game mechanics including state management,
action execution, and observation generation. All functions are JIT-compiled
for maximum performance.

The engine is parameterized by the number of players N (a JIT-static dimension).
Free-for-all is N teams of one (teams = arange(N)); team modes assign multiple
players to the same team_id (e.g. 2v2 is teams = [0, 0, 1, 1]).

Key functions:
    - create_initial_state: Create a new game from a grid and a teams array
    - step: Execute one game step with actions from all players
    - get_observation: Get a player's view with fog of war applied
"""
import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, NamedTuple, Protocol, Any

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
        generals: (H, W) boolean mask of *active* general positions. When a
            general is captured it converts to a city and is removed from this mask.
        cities: (H, W) boolean mask of city positions.
        mountains: (H, W) boolean mask of mountain positions.
        passable: (H, W) boolean mask of passable cells (not mountains).
        general_positions: (N, 2) array of [row, col] for each general at start of game.
        teams: (N,) int32 array, teams[i] is team_id of player i.
        eliminated: (N,) bool array, True if player i has been eliminated (general captured).
        time: Scalar, current game timestep.
        winner: Scalar, -1 if game ongoing, team_id of winning team otherwise.
        pool_idx: Scalar, index into the pre-generated state pool for auto-reset.
    """

    armies: jnp.ndarray
    ownership: jnp.ndarray
    ownership_neutral: jnp.ndarray
    generals: jnp.ndarray
    cities: jnp.ndarray
    mountains: jnp.ndarray
    passable: jnp.ndarray
    general_positions: jnp.ndarray
    teams: jnp.ndarray
    eliminated: jnp.ndarray
    time: jnp.ndarray
    winner: jnp.ndarray
    pool_idx: jnp.ndarray


class GameInfo(NamedTuple):
    """
    Game statistics returned after each step.

    Attributes:
        army: (N,) array of total army counts per player.
        land: (N,) array of total land counts per player.
        is_done: Boolean, True if game has ended.
        winner: -1 if ongoing, team_id of winning team otherwise.
        time: Current game timestep.
    """

    army: jnp.ndarray
    land: jnp.ndarray
    is_done: jnp.ndarray
    winner: jnp.ndarray
    time: jnp.ndarray


# Direction offsets: UP, DOWN, LEFT, RIGHT
DIRECTIONS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


def create_initial_state(grid: jnp.ndarray, teams: jnp.ndarray) -> GameState:
    """
    Create initial game state from a numeric grid and teams assignment.

    Args:
        grid: 2D array with cell values:
            - -2: Mountain (impassable)
            - 0: Empty cell
            - k (1..N): Player (k-1)'s general
            - 40-50: City with that army value
        teams: (N,) int32 array, teams[i] is team_id of player i.

    Returns:
        GameState ready for gameplay.
    """
    H, W = grid.shape
    N = teams.shape[0]

    # Per-player general masks: player i's general has grid value (i+1)
    player_general_masks = jnp.stack([grid == (i + 1) for i in range(N)])  # (N, H, W)
    generals = jnp.any(player_general_masks, axis=0)

    mountains = grid == -2
    passable = grid != -2
    cities = grid > N  # values above the general range encode cities (40-50)

    ownership = player_general_masks  # (N, H, W)
    ownership_neutral = passable & ~generals

    armies = jnp.where(generals, 1, 0).astype(jnp.int32)
    armies = jnp.where(cities, grid, armies)

    general_positions = jnp.stack([
        jnp.argwhere(player_general_masks[i], size=1, fill_value=-1)[0]
        for i in range(N)
    ])

    return GameState(
        armies=armies,
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        generals=generals,
        cities=cities,
        mountains=mountains,
        passable=passable,
        general_positions=general_positions,
        teams=teams,
        eliminated=jnp.zeros(N, dtype=bool),
        time=jnp.int32(0),
        winner=jnp.int32(-1),
        pool_idx=jnp.int32(0),
    )


@jax.jit
def get_visibility(ownership_mask: jnp.ndarray) -> jnp.ndarray:
    """Compute visibility mask (3x3 around owned cells). Input is a (H, W) bool mask."""
    H, W = ownership_mask.shape
    ownership_float = ownership_mask.astype(jnp.float32)
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


def execute_action(state: GameState, player_idx, action: jnp.ndarray) -> GameState:
    """Execute a single player's action."""
    pass_turn, si, sj, direction, split_army = action

    return lax.cond(
        pass_turn == 1,
        lambda s: s,
        lambda s: _execute_move(s, player_idx, si, sj, direction, split_army),
        state,
    )


def _execute_move(state: GameState, player_idx, si, sj, direction, split_army) -> GameState:
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

    valid_move = in_bounds & dest_in_bounds & owns_source & (army_to_move > 0) & state.passable[di, dj]

    return lax.cond(
        valid_move,
        lambda s: _apply_move(s, player_idx, si, sj, di, dj, army_to_move),
        lambda s: s,
        state,
    )


def _apply_move(state: GameState, player_idx, si, sj, di, dj, army_to_move) -> GameState:
    """
    Apply a validated move.

    Three branches by destination ownership:
      - Friendly (same-team owner, including self): armies add; ownership transfers to
        the mover if the destination belonged to a teammate.
      - Enemy/neutral non-general capture: standard attack (subtract, flip ownership
        if attacker wins).
      - Enemy general capture: the entire captured player's territory transfers to
        the capturer with per-cell armies halved; the general tile becomes a city;
        the captured player is marked eliminated. The game does NOT end here — the
        win condition is evaluated by step() (last team standing).
    """
    armies = state.armies
    ownership = state.ownership
    ownership_neutral = state.ownership_neutral
    eliminated = state.eliminated
    generals = state.generals
    cities = state.cities
    teams = state.teams

    player_team = teams[player_idx]

    target_owner_mask = ownership[:, di, dj]
    same_team_owners = target_owner_mask & (teams == player_team)
    is_friendly = jnp.any(same_team_owners)
    is_self_owned = ownership[player_idx, di, dj]

    def set_dest_owner_to_mover(o):
        o = o.at[:, di, dj].set(False)
        o = o.at[player_idx, di, dj].set(True)
        return o

    # ---- Friendly merge ----
    armies_friendly = armies.at[di, dj].add(army_to_move).at[si, sj].add(-army_to_move)
    ownership_friendly = lax.cond(is_self_owned, lambda o: o, set_dest_owner_to_mover, ownership)

    # ---- Standard attack (no general capture) ----
    target_army = armies[di, dj]
    attacker_wins = army_to_move > target_army
    remaining_army = jnp.abs(target_army - army_to_move)

    armies_attack = armies.at[di, dj].set(remaining_army).at[si, sj].add(-army_to_move)
    ownership_attack = lax.cond(attacker_wins, set_dest_owner_to_mover, lambda o: o, ownership)
    ownership_neutral_attack = lax.cond(
        attacker_wins & ownership_neutral[di, dj],
        lambda o: o.at[di, dj].set(False),
        lambda o: o,
        ownership_neutral,
    )

    is_general = state.generals[di, dj]
    general_captured = attacker_wins & is_general & ~is_friendly

    # ---- General-capture: sweep captured player's territory to capturer (half armies) ----
    captured_player_idx = jnp.argmax(target_owner_mask)
    captured_cells = ownership[captured_player_idx]

    armies_capture = jnp.where(captured_cells, armies // 2, armies)
    armies_capture = armies_capture.at[si, sj].add(-army_to_move)

    ownership_capture = ownership.at[player_idx].set(ownership[player_idx] | captured_cells)
    ownership_capture = ownership_capture.at[captured_player_idx].set(jnp.zeros_like(captured_cells))

    generals_capture = generals.at[di, dj].set(False)
    cities_capture = cities.at[di, dj].set(True)
    eliminated_capture = eliminated.at[captured_player_idx].set(True)

    # ---- Branch selection ----
    armies = lax.cond(
        is_friendly,
        lambda: armies_friendly,
        lambda: lax.cond(general_captured, lambda: armies_capture, lambda: armies_attack),
    )
    ownership = lax.cond(
        is_friendly,
        lambda: ownership_friendly,
        lambda: lax.cond(general_captured, lambda: ownership_capture, lambda: ownership_attack),
    )
    ownership_neutral = lax.cond(
        is_friendly,
        lambda: ownership_neutral,
        lambda: lax.cond(general_captured, lambda: ownership_neutral, lambda: ownership_neutral_attack),
    )
    generals = lax.cond(general_captured, lambda: generals_capture, lambda: generals)
    cities = lax.cond(general_captured, lambda: cities_capture, lambda: cities)
    eliminated = lax.cond(general_captured, lambda: eliminated_capture, lambda: eliminated)

    return state._replace(
        armies=armies,
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        generals=generals,
        cities=cities,
        eliminated=eliminated,
    )


def global_update(state: GameState) -> GameState:
    """Perform army increments (every 2 turns for structures, every 50 for all)."""
    time = state.time
    armies = state.armies

    total_owned = jnp.any(state.ownership, axis=0).astype(jnp.int32)  # (H, W)

    increment_all = time % 50 == 0
    armies = lax.cond(
        increment_all,
        lambda a: a + total_owned,
        lambda a: a,
        armies,
    )

    increment_structures = (time % 2 == 1)
    structure_mask = (state.generals | state.cities).astype(jnp.int32)
    armies = lax.cond(
        increment_structures,
        lambda a: a + structure_mask * total_owned,
        lambda a: a,
        armies,
    )

    return state._replace(armies=armies)


def _determine_move_order(state: GameState, actions: jnp.ndarray) -> jnp.ndarray:
    """
    Determine the order in which players' actions are applied this step.

    Returns a (N,) array of player indices in order of execution. Players with
    higher priority (chasing > reinforcing > larger army) go first; pass actions
    go last. Ties are broken by player index (stable sort), so for N=2 with no
    distinguishing priority, player 0 moves first.
    """
    N = actions.shape[0]

    passes = actions[:, 0]
    sources_i = actions[:, 1]
    sources_j = actions[:, 2]
    directions = actions[:, 3]

    dests_i = sources_i + DIRECTIONS[directions, 0]
    dests_j = sources_j + DIRECTIONS[directions, 1]

    # chasing[i] = does any other player's source equal player i's destination
    chasing_matrix = (sources_i[None, :] == dests_i[:, None]) & (sources_j[None, :] == dests_j[:, None])
    chasing_matrix = chasing_matrix & ~jnp.eye(N, dtype=bool)
    chasing = jnp.any(chasing_matrix, axis=1)

    # reinforcing[i] = player i moves onto their own cell
    H, W = state.armies.shape
    safe_di = jnp.clip(dests_i, 0, H - 1)
    safe_dj = jnp.clip(dests_j, 0, W - 1)
    reinforcing = state.ownership[jnp.arange(N), safe_di, safe_dj]

    safe_si = jnp.clip(sources_i, 0, H - 1)
    safe_sj = jnp.clip(sources_j, 0, W - 1)
    source_armies = state.armies[safe_si, safe_sj]

    # Lexicographic priority (chasing, reinforcing, source_army). Pass = lowest priority.
    H_W = jnp.int32(H * W)
    army_cap = source_armies + 1  # ensure non-negative
    priority = (
        chasing.astype(jnp.int32) * (H_W * H_W * 8)
        + reinforcing.astype(jnp.int32) * (H_W * 8)
        + army_cap
    )
    priority = jnp.where(passes == 1, jnp.int32(-1), priority)

    order = jnp.argsort(-priority, stable=True)
    return order


@jax.jit
def step(state: GameState, actions: jnp.ndarray) -> tuple[GameState, GameInfo]:
    """
    Execute one game step with actions from all players.

    Args:
        state: Current game state.
        actions: (N, 5) array, one [pass, row, col, direction, split] per player.
    """
    N = actions.shape[0]
    done_before = state.winner >= 0

    order = _determine_move_order(state, actions)

    for k in range(N):
        idx = order[k]
        state = execute_action(state, idx, actions[idx])

    state = lax.cond(done_before, lambda s: s, lambda s: s._replace(time=s.time + 1), state)

    # Last-team-standing check. Once winner is set we hold it.
    new_winner = _check_winner(state)
    state = state._replace(winner=jnp.where(state.winner >= 0, state.winner, new_winner))

    state = lax.cond(
        state.winner >= 0,
        lambda s: s,
        lambda s: global_update(s),
        state,
    )

    return state, get_info(state)


def _check_winner(state: GameState) -> jnp.ndarray:
    """Return the team_id of the only surviving team, or -1 if more than one team is alive."""
    active = ~state.eliminated  # (N,)
    INT_MAX = jnp.iinfo(jnp.int32).max
    INT_MIN = jnp.iinfo(jnp.int32).min
    min_team = jnp.min(jnp.where(active, state.teams, INT_MAX))
    max_team = jnp.max(jnp.where(active, state.teams, INT_MIN))
    single_team = (min_team == max_team) & jnp.any(active)
    return jnp.where(single_team, min_team, jnp.int32(-1))


@jax.jit
def get_info(state: GameState) -> GameInfo:
    """Compute game statistics."""
    armies = state.armies
    ownership = state.ownership  # (N, H, W)

    army = jnp.sum(armies[None, :, :] * ownership.astype(jnp.int32), axis=(1, 2))  # (N,)
    land = jnp.sum(ownership, axis=(1, 2))  # (N,)

    return GameInfo(
        army=army,
        land=land,
        is_done=state.winner >= 0,
        winner=state.winner,
        time=state.time,
    )


@jax.jit
def get_observation(state: GameState, player_idx) -> Observation:
    """
    Get player observation with fog of war applied.

    For phase 1, "opponent" aggregates every non-self player (allies and enemies alike).
    Phase 4 will split this into separate allied / enemy channels.
    """
    own_ownership = state.ownership[player_idx]
    visible = get_visibility(own_ownership)
    invisible = ~visible

    all_owned = jnp.any(state.ownership, axis=0)
    opponent_cells_all = all_owned & ~own_ownership

    info = get_info(state)

    total_land = jnp.sum(info.land)
    total_army = jnp.sum(info.army)

    return Observation(
        armies=state.armies * visible,
        generals=state.generals * visible,
        cities=state.cities * visible,
        mountains=state.mountains * visible,
        neutral_cells=state.ownership_neutral * visible,
        owned_cells=own_ownership * visible,
        opponent_cells=opponent_cells_all * visible,
        fog_cells=invisible & ~(state.mountains | state.cities),
        structures_in_fog=invisible & (state.mountains | state.cities),
        owned_land_count=info.land[player_idx],
        owned_army_count=info.army[player_idx],
        opponent_land_count=total_land - info.land[player_idx],
        opponent_army_count=total_army - info.army[player_idx],
        timestep=state.time,
    )


@jax.jit
def get_full_observation(state: GameState, player_idx) -> Observation:
    """
    Perfect-info observation. Same content as get_observation but with no fog.
    """
    own_ownership = state.ownership[player_idx]
    all_owned = jnp.any(state.ownership, axis=0)
    opponent_cells_all = all_owned & ~own_ownership

    info = get_info(state)
    total_land = jnp.sum(info.land)
    total_army = jnp.sum(info.army)

    z = jnp.zeros_like(state.ownership[0])

    return Observation(
        armies=state.armies,
        generals=state.generals,
        cities=state.cities,
        mountains=state.mountains,
        neutral_cells=state.ownership_neutral,
        owned_cells=own_ownership,
        opponent_cells=opponent_cells_all,
        fog_cells=z,
        structures_in_fog=z,
        owned_land_count=info.land[player_idx],
        owned_army_count=info.army[player_idx],
        opponent_land_count=total_land - info.land[player_idx],
        opponent_army_count=total_army - info.army[player_idx],
        timestep=state.time,
    )


@jax.jit
def batch_step(states: GameState, actions: jnp.ndarray) -> Tuple[GameState, GameInfo]:
    """Vectorized step for multiple environments."""
    return jax.vmap(step)(states, actions)
