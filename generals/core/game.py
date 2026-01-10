"""
JAX game logic for Generals.io.

This module contains the core game mechanics including state management,
action execution, and observation generation. All functions are JIT-compiled
for maximum performance.

Key functions:
    - create_initial_state: Create a new game from a grid
    - step: Execute one game step with actions from both players
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
        ownership: (2, H, W) boolean arrays, ownership[i] is player i's cells.
        ownership_neutral: (H, W) boolean mask of neutral (unowned) cells.
        generals: (H, W) boolean mask of general positions.
        cities: (H, W) boolean mask of city positions.
        mountains: (H, W) boolean mask of mountain positions.
        passable: (H, W) boolean mask of passable cells (not mountains).
        general_positions: (2, 2) array of [row, col] for each general.
        time: Scalar, current game timestep.
        winner: Scalar, -1 if game ongoing, 0 or 1 if that player won.
    """

    armies: jnp.ndarray
    ownership: jnp.ndarray
    ownership_neutral: jnp.ndarray
    generals: jnp.ndarray
    cities: jnp.ndarray
    mountains: jnp.ndarray
    passable: jnp.ndarray
    general_positions: jnp.ndarray
    time: jnp.ndarray
    winner: jnp.ndarray


class GameInfo(NamedTuple):
    """
    Game statistics returned after each step.

    Attributes:
        army: (2,) array of total army counts per player.
        land: (2,) array of total land counts per player.
        is_done: Boolean, True if game has ended.
        winner: -1 if ongoing, 0 or 1 indicating winner.
        time: Current game timestep.
    """

    army: jnp.ndarray
    land: jnp.ndarray
    is_done: jnp.ndarray
    winner: jnp.ndarray
    time: jnp.ndarray


# Direction offsets: UP, DOWN, LEFT, RIGHT
DIRECTIONS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


def create_initial_state(grid: jnp.ndarray) -> GameState:
    """
    Create initial game state from a numeric grid.

    Args:
        grid: 2D array with cell values:
            - -2: Mountain (impassable)
            - 0: Empty cell
            - 1: Player 0's general
            - 2: Player 1's general
            - 40-50: City with that army value

    Returns:
        GameState ready for gameplay.
    """
    H, W = grid.shape

    is_general_0 = grid == 1
    is_general_1 = grid == 2
    generals = is_general_0 | is_general_1

    mountains = grid == -2
    passable = grid != -2
    cities = (grid >= 40) & (grid <= 50)

    ownership = jnp.stack([is_general_0, is_general_1])
    ownership_neutral = passable & ~is_general_0 & ~is_general_1

    armies = jnp.where(is_general_0 | is_general_1, 1, 0).astype(jnp.int32)
    armies = jnp.where(cities, grid, armies)

    general_pos_0 = jnp.argwhere(is_general_0, size=1, fill_value=-1)[0]
    general_pos_1 = jnp.argwhere(is_general_1, size=1, fill_value=-1)[0]
    general_positions = jnp.stack([general_pos_0, general_pos_1])

    return GameState(
        armies=armies,
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        generals=generals,
        cities=cities,
        mountains=mountains,
        passable=passable,
        general_positions=general_positions,
        time=jnp.int32(0),
        winner=jnp.int32(-1),
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


@jax.jit
def execute_action(state: GameState, player_idx: int, action: jnp.ndarray) -> GameState:
    """Execute a single player's action."""
    pass_turn, si, sj, direction, split_army = action

    return lax.cond(
        pass_turn == 1,
        lambda s: s,
        lambda s: _execute_move(s, player_idx, si, sj, direction, split_army),
        state,
    )


@jax.jit
def _execute_move(state: GameState, player_idx: int, si: int, sj: int, direction: int, split_army: int) -> GameState:
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


@jax.jit
def _apply_move(state: GameState, player_idx: int, si: int, sj: int, di: int, dj: int, army_to_move: int) -> GameState:
    """Apply move to game state."""
    armies = state.armies
    ownership = state.ownership
    ownership_neutral = state.ownership_neutral

    target_owner_0 = ownership[0, di, dj]
    target_owner_1 = ownership[1, di, dj]
    target_neutral = ownership_neutral[di, dj]

    moving_to_own = (player_idx == 0) & target_owner_0 | (player_idx == 1) & target_owner_1

    # Move to own cell
    armies_own = armies.at[di, dj].add(army_to_move)
    armies_own = armies_own.at[si, sj].add(-army_to_move)

    # Attack
    target_army = armies[di, dj]
    attacker_wins = army_to_move > target_army
    remaining_army = jnp.abs(target_army - army_to_move)

    armies_attack = armies.at[di, dj].set(remaining_army)
    armies_attack = armies_attack.at[si, sj].add(-army_to_move)

    ownership_attack = ownership
    ownership_neutral_attack = ownership_neutral

    ownership_attack = lax.cond(attacker_wins, lambda o: o.at[player_idx, di, dj].set(True), lambda o: o, ownership_attack)

    ownership_attack = lax.cond(
        attacker_wins & target_owner_0 & (player_idx == 1), lambda o: o.at[0, di, dj].set(False), lambda o: o, ownership_attack
    )
    ownership_attack = lax.cond(
        attacker_wins & target_owner_1 & (player_idx == 0), lambda o: o.at[1, di, dj].set(False), lambda o: o, ownership_attack
    )
    ownership_neutral_attack = lax.cond(
        attacker_wins & target_neutral, lambda o: o.at[di, dj].set(False), lambda o: o, ownership_neutral_attack
    )

    is_general = state.generals[di, dj]
    general_captured = attacker_wins & is_general & ~moving_to_own

    armies = lax.cond(moving_to_own, lambda: armies_own, lambda: armies_attack)
    ownership = lax.cond(moving_to_own, lambda: ownership, lambda: ownership_attack)
    ownership_neutral = lax.cond(moving_to_own, lambda: ownership_neutral, lambda: ownership_neutral_attack)

    winner = lax.cond(general_captured, lambda: jnp.int32(player_idx), lambda: state.winner)

    return state._replace(armies=armies, ownership=ownership, ownership_neutral=ownership_neutral, winner=winner)


@jax.jit
def global_update(state: GameState) -> GameState:
    """Perform army increments (every 2 turns for structures, every 50 for all)."""
    time = state.time
    armies = state.armies

    increment_all = time % 50 == 0
    armies = lax.cond(
        increment_all,
        lambda a: a + state.ownership[0].astype(jnp.int32) + state.ownership[1].astype(jnp.int32),
        lambda a: a,
        armies,
    )

    increment_structures = (time % 2 == 0) & (time > 0)
    structure_mask = (state.generals | state.cities).astype(jnp.int32)
    armies = lax.cond(
        increment_structures,
        lambda a: a + structure_mask * state.ownership[0].astype(jnp.int32) + structure_mask * state.ownership[1].astype(jnp.int32),
        lambda a: a,
        armies,
    )

    return state._replace(armies=armies)


def _determine_move_order(state: GameState, actions: jnp.ndarray) -> int:
    """Determine which player moves first (chasing > reinforcing > bigger army)."""
    pass_0, row_0, col_0, dir_0, _ = actions[0]
    pass_1, row_1, col_1, dir_1, _ = actions[1]

    only_p0_passes = pass_0 & ~pass_1

    si_0, sj_0 = row_0, col_0
    di_0, dj_0 = si_0 + DIRECTIONS[dir_0][0], sj_0 + DIRECTIONS[dir_0][1]

    si_1, sj_1 = row_1, col_1
    di_1, dj_1 = si_1 + DIRECTIONS[dir_1][0], sj_1 + DIRECTIONS[dir_1][1]

    p0_chasing = (di_0 == si_1) & (dj_0 == sj_1)
    p1_chasing = (di_1 == si_0) & (dj_1 == sj_0)

    p0_reinforcing = state.ownership[0, di_0, dj_0]
    p1_reinforcing = state.ownership[1, di_1, dj_1]

    army_0 = state.armies[si_0, sj_0]
    army_1 = state.armies[si_1, sj_1]

    p1_wins_by_chase = p1_chasing & ~p0_chasing
    tie_on_chase = p0_chasing == p1_chasing
    p1_wins_by_reinforce = tie_on_chase & p1_reinforcing & ~p0_reinforcing
    tie_on_reinforce = p0_reinforcing == p1_reinforcing
    p1_wins_by_army = tie_on_chase & tie_on_reinforce & (army_1 > army_0)

    p1_goes_first = p1_wins_by_chase | p1_wins_by_reinforce | p1_wins_by_army | only_p0_passes

    return lax.cond(p1_goes_first, lambda: 1, lambda: 0)


@jax.jit
def step(state: GameState, actions: jnp.ndarray) -> tuple[GameState, GameInfo]:
    """Execute one game step with actions from both players."""
    done_before = state.winner >= 0

    first_player = _determine_move_order(state, actions)
    second_player = 1 - first_player

    state = execute_action(state, first_player, actions[first_player])
    state = execute_action(state, second_player, actions[second_player])

    state = lax.cond(done_before, lambda s: s, lambda s: s._replace(time=s.time + 1), state)

    state = lax.cond(
        state.winner >= 0,
        lambda s: _transfer_loser_cells_to_winner(s),
        lambda s: global_update(s),
        state,
    )

    return state, get_info(state)


def _transfer_loser_cells_to_winner(state: GameState) -> GameState:
    """Transfer loser's cells to winner."""
    winner_idx = state.winner
    loser_idx = 1 - winner_idx

    new_ownership = state.ownership.at[winner_idx].set(state.ownership[winner_idx] | state.ownership[loser_idx])
    new_ownership = new_ownership.at[loser_idx].set(jnp.zeros_like(state.ownership[loser_idx], dtype=bool))
    new_ownership_neutral = state.ownership_neutral & ~state.ownership[loser_idx]

    return state._replace(ownership=new_ownership, ownership_neutral=new_ownership_neutral)


@jax.jit
def get_info(state: GameState) -> GameInfo:
    """Compute game statistics."""
    armies = state.armies
    ownership = state.ownership

    return GameInfo(
        army=jnp.stack([jnp.sum(armies * ownership[0]), jnp.sum(armies * ownership[1])]),
        land=jnp.stack([jnp.sum(ownership[0]), jnp.sum(ownership[1])]),
        is_done=state.winner >= 0,
        winner=state.winner,
        time=state.time,
    )


@jax.jit
def get_observation(state: GameState, player_idx: int) -> Observation:
    """Get player observation with fog of war applied."""
    visible = get_visibility(state.ownership[player_idx])
    invisible = ~visible
    opponent_idx = 1 - player_idx

    info = get_info(state)

    return Observation(
        armies=state.armies * visible,
        generals=state.generals * visible,
        cities=state.cities * visible,
        mountains=state.mountains * visible,
        neutral_cells=state.ownership_neutral * visible,
        owned_cells=state.ownership[player_idx] * visible,
        opponent_cells=state.ownership[opponent_idx] * visible,
        fog_cells=invisible & ~(state.mountains | state.cities),
        structures_in_fog=invisible & (state.mountains | state.cities),
        owned_land_count=info.land[player_idx],
        owned_army_count=info.army[player_idx],
        opponent_land_count=info.land[opponent_idx],
        opponent_army_count=info.army[opponent_idx],
        timestep=state.time,
    )


@jax.jit
def batch_step(states: GameState, actions: jnp.ndarray) -> Tuple[GameState, GameInfo]:
    """Vectorized step for multiple environments."""
    return jax.vmap(step)(states, actions)
