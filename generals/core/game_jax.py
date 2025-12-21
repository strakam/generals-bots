import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, Tuple, Any, NamedTuple

from generals.core.observation_jax import ObservationJax


class GameState(NamedTuple):
    """Game state NamedTuple for efficient JAX operations."""
    armies: jnp.ndarray  # [H, W] army counts
    ownership: jnp.ndarray  # [2, H, W] player ownership
    ownership_neutral: jnp.ndarray  # [H, W] neutral cells
    generals: jnp.ndarray  # [H, W] general positions (static)
    cities: jnp.ndarray  # [H, W] city positions (static)
    mountains: jnp.ndarray  # [H, W] mountain positions (static)
    passable: jnp.ndarray  # [H, W] passable cells (static)
    general_positions: jnp.ndarray  # [2, 2] (row, col) for each player
    time: jnp.ndarray  # scalar int
    winner: jnp.ndarray  # scalar int (-1 for none, 0 or 1 for player)


class GameInfo(NamedTuple):
    """Game information NamedTuple for efficient JAX operations."""
    army: jnp.ndarray  # [2] array of army counts for both players
    land: jnp.ndarray  # [2] array of land counts for both players
    is_done: jnp.ndarray  # scalar boolean
    winner: jnp.ndarray  # scalar int (-1 for none, 0 or 1 for player)
    time: jnp.ndarray  # scalar int

# Direction constants matching config.py
DIRECTIONS = jnp.array([
    [-1, 0],  # UP
    [1, 0],   # DOWN
    [0, -1],  # LEFT
    [0, 1],   # RIGHT
], dtype=jnp.int32)


def create_initial_state(grid: jnp.ndarray) -> GameState:
    """
    Create initial game state from a numeric JAX grid.
    
    Args:
        grid: Numeric array where:
            -2 = mountain
            0 = passable/neutral
            1 = general player 0
            2 = general player 1
            40-50 = cities (army count)
    
    Returns:
        GameState NamedTuple containing all game state arrays
    """
    H, W = grid.shape
    
    # Decode grid
    is_general_0 = (grid == 1)
    is_general_1 = (grid == 2)
    generals = (is_general_0 | is_general_1)
    
    mountains = (grid == -2)
    passable = (grid != -2)  # Everything except mountains is passable
    
    # Cities: values 40-50
    cities = (grid >= 40) & (grid <= 50)
    
    # Initial ownership
    ownership = jnp.stack([
        is_general_0,
        is_general_1,
    ])
    
    # Neutral cells (passable but not owned)
    ownership_neutral = passable & ~is_general_0 & ~is_general_1
    
    # Initial armies
    # Generals start with 1 army
    armies = jnp.where(is_general_0 | is_general_1, 1, 0).astype(jnp.int32)
    
    # Cities have their encoded army value
    armies = jnp.where(cities, grid, armies)
    
    # Find general positions
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
    """
    Compute visibility mask for a player using convolution (optimized).
    
    Args:
        ownership: [H, W] boolean array of owned cells
    
    Returns:
        [H, W] boolean array of visible cells (3x3 around owned cells)
    """
    # Use efficient max pooling approach with padding
    # Any cell within 1 step of an owned cell is visible
    H, W = ownership.shape
    ownership_float = ownership.astype(jnp.float32)
    
    # Pad with zeros for boundary handling
    padded = jnp.pad(ownership_float, 1, mode='constant', constant_values=0)
    
    # Stack all shifted versions directly (no Python list)
    # This is faster than creating a list and then stacking
    stacked = jnp.stack([
        padded[0:H, 0:W],     # top-left
        padded[0:H, 1:W+1],   # top
        padded[0:H, 2:W+2],   # top-right
        padded[1:H+1, 0:W],   # left
        padded[1:H+1, 1:W+1], # center
        padded[1:H+1, 2:W+2], # right
        padded[2:H+2, 0:W],   # bottom-left
        padded[2:H+2, 1:W+1], # bottom
        padded[2:H+2, 2:W+2], # bottom-right
    ], axis=0)
    
    # Any neighbor owned -> visible
    visible = jnp.max(stacked, axis=0) > 0
    
    return visible


@jax.jit
def execute_action(
    state: GameState,
    player_idx: int,
    action: jnp.ndarray,
) -> GameState:
    """
    Execute a single player's action.
    
    Args:
        state: Current game state
        player_idx: Player index (0 or 1)
        action: [5] array [pass, row, col, direction, split]
    
    Returns:
        Updated state
    """
    pass_turn, si, sj, direction, split_army = action
    
    # Early exit if passing
    state = lax.cond(
        pass_turn == 1,
        lambda s: s,  # Do nothing
        lambda s: _execute_move(s, player_idx, si, sj, direction, split_army),
        state
    )
    
    return state


@jax.jit
def _execute_move(
    state: GameState,
    player_idx: int,
    si: int,
    sj: int,
    direction: int,
    split_army: int,
) -> GameState:
    """Helper to execute actual move logic."""
    H, W = state.armies.shape
    
    # Check bounds
    in_bounds = (si >= 0) & (si < H) & (sj >= 0) & (sj < W)
    
    # Calculate destination
    di = si + DIRECTIONS[direction, 0]
    dj = sj + DIRECTIONS[direction, 1]
    dest_in_bounds = (di >= 0) & (di < H) & (dj >= 0) & (dj < W)
    
    # Check ownership and army requirements (using current state after any previous moves)
    owns_source = state.ownership[player_idx, si, sj]
    source_army = state.armies[si, sj]
    
    # Calculate army to move based on current army (CRITICAL: use current state!)
    army_to_move = lax.cond(
        split_army == 1,
        lambda a: a // 2,
        lambda a: a - 1,
        source_army
    )
    # Cap to available army - matches line 133 in game.py
    army_to_move = jnp.maximum(0, jnp.minimum(army_to_move, source_army - 1))
    
    # Check if move is valid
    valid_move = (
        in_bounds &
        dest_in_bounds &
        owns_source &
        (army_to_move > 0) &
        state.passable[di, dj]
    )
    
    # Execute move if valid
    state = lax.cond(
        valid_move,
        lambda s: _apply_move(s, player_idx, si, sj, di, dj, army_to_move),
        lambda s: s,  # No-op if invalid
        state
    )
    
    return state


@jax.jit
def _apply_move(
    state: GameState,
    player_idx: int,
    si: int,
    sj: int,
    di: int,
    dj: int,
    army_to_move: int,
) -> GameState:
    """Apply the move to the game state."""
    armies = state.armies
    ownership = state.ownership
    ownership_neutral = state.ownership_neutral
    
    # Determine target owner (0, 1, or 2 for neutral)
    target_owner_0 = ownership[0, di, dj]
    target_owner_1 = ownership[1, di, dj]
    target_neutral = ownership_neutral[di, dj]
    
    # If moving to own cell
    moving_to_own = (player_idx == 0) & target_owner_0 | (player_idx == 1) & target_owner_1
    
    # Update for moving to own cell
    armies_own = armies.at[di, dj].add(army_to_move)
    armies_own = armies_own.at[si, sj].add(-army_to_move)
    
    # Update for attacking (neutral or enemy)
    target_army = armies[di, dj]
    attacker_wins = army_to_move > target_army
    remaining_army = jnp.abs(target_army - army_to_move)
    
    armies_attack = armies.at[di, dj].set(remaining_army)
    armies_attack = armies_attack.at[si, sj].add(-army_to_move)
    
    # Update ownership on successful attack
    ownership_attack = ownership
    ownership_neutral_attack = ownership_neutral
    
    # If attacker wins, update ownership
    ownership_attack = lax.cond(
        attacker_wins,
        lambda o: o.at[player_idx, di, dj].set(True),
        lambda o: o,
        ownership_attack
    )
    
    # Remove previous owner if attacker wins
    other_player = 1 - player_idx
    ownership_attack = lax.cond(
        attacker_wins & target_owner_0 & (player_idx == 1),
        lambda o: o.at[0, di, dj].set(False),
        lambda o: o,
        ownership_attack
    )
    ownership_attack = lax.cond(
        attacker_wins & target_owner_1 & (player_idx == 0),
        lambda o: o.at[1, di, dj].set(False),
        lambda o: o,
        ownership_attack
    )
    ownership_neutral_attack = lax.cond(
        attacker_wins & target_neutral,
        lambda o: o.at[di, dj].set(False),
        lambda o: o,
        ownership_neutral_attack
    )
    
    # Check if general captured (can't capture your own general!)
    is_general = state.generals[di, dj]
    general_captured = attacker_wins & is_general & ~moving_to_own
    
    # Choose between own cell move or attack
    armies = lax.cond(moving_to_own, lambda: armies_own, lambda: armies_attack)
    ownership = lax.cond(moving_to_own, lambda: ownership, lambda: ownership_attack)
    ownership_neutral = lax.cond(moving_to_own, lambda: ownership_neutral, lambda: ownership_neutral_attack)
    
    # Update winner if general captured
    winner = lax.cond(
        general_captured,
        lambda: jnp.int32(player_idx),
        lambda: state.winner
    )
    
    # Return updated state using _replace (NamedTuple method)
    return state._replace(
        armies=armies,
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        winner=winner,
    )


@jax.jit
def global_update(state: GameState) -> GameState:
    """
    Perform global game updates (army increments).
    
    Every 50 turns: increment all owned cells by 1
    Every 2 turns: increment generals and cities by 1 (if owned)
    """
    time = state.time
    armies = state.armies
    
    # Every 50 turns, increment all owned cells
    increment_all = (time % 50 == 0)
    armies = lax.cond(
        increment_all,
        lambda a: a + state.ownership[0].astype(jnp.int32) + state.ownership[1].astype(jnp.int32),
        lambda a: a,
        armies
    )
    
    # Every 2 turns, increment generals and cities
    increment_structures = (time % 2 == 0) & (time > 0)
    structure_mask = (state.generals | state.cities).astype(jnp.int32)
    armies = lax.cond(
        increment_structures,
        lambda a: a + structure_mask * state.ownership[0].astype(jnp.int32) + \
                      structure_mask * state.ownership[1].astype(jnp.int32),
        lambda a: a,
        armies
    )
    
    return state._replace(armies=armies)


def _determine_move_order(state: GameState, actions: jnp.ndarray) -> int:
    """
    Determine which player's move should be executed first.
    Based on game.py lines 1235-1280.
    
    Priority (higher priority goes first):
    1. Chasing move (moving to cell where opponent is moving FROM)
    2. Reinforcing move (moving to own cell)
    3. Bigger army being moved
    
    Returns:
        Player index (0 or 1) who should move first
    """
    # Parse actions
    pass_0, row_0, col_0, dir_0, split_0 = actions[0]
    pass_1, row_1, col_1, dir_1, split_1 = actions[1]
    
    # If either player passes, other goes first
    both_pass = pass_0 & pass_1
    only_p0_passes = pass_0 & ~pass_1
    only_p1_passes = ~pass_0 & pass_1
    
    # Compute source and destination cells for both players
    si_0, sj_0 = row_0, col_0
    di_0, dj_0 = si_0 + DIRECTIONS[dir_0][0], sj_0 + DIRECTIONS[dir_0][1]
    
    si_1, sj_1 = row_1, col_1
    di_1, dj_1 = si_1 + DIRECTIONS[dir_1][0], sj_1 + DIRECTIONS[dir_1][1]
    
    # Check if moves are chasing (p0 moves to where p1 is moving from, or vice versa)
    p0_chasing = (di_0 == si_1) & (dj_0 == sj_1)  # p0 moves to p1's source
    p1_chasing = (di_1 == si_0) & (dj_1 == sj_0)  # p1 moves to p0's source
    
    # Check if moves are reinforcing (moving to own cell)
    p0_reinforcing = state.ownership[0, di_0, dj_0]
    p1_reinforcing = state.ownership[1, di_1, dj_1]
    
    # Calculate army sizes being moved
    army_0 = state.armies[si_0, sj_0]
    army_1 = state.armies[si_1, sj_1]
    
    # Priority calculation (matching game.py logic)
    # If one is chasing and other isn't, chasing goes first
    p1_wins_by_chase = p1_chasing & ~p0_chasing
    
    # If both/neither chasing, check reinforcing
    tie_on_chase = (p0_chasing == p1_chasing)
    p1_wins_by_reinforce = tie_on_chase & p1_reinforcing & ~p0_reinforcing
    
    # If both/neither reinforcing, bigger army goes first
    tie_on_reinforce = (p0_reinforcing == p1_reinforcing)
    p1_wins_by_army = tie_on_chase & tie_on_reinforce & (army_1 > army_0)
    
    # Determine winner (default to player 0 when fully tied, matching numpy sorted() behavior)
    p1_goes_first = p1_wins_by_chase | p1_wins_by_reinforce | p1_wins_by_army | only_p0_passes
    
    # When fully tied (neither has priority), default to player 0 to match NumPy
    return lax.cond(
        p1_goes_first,
        lambda: 1,
        lambda: 0  # Default to player 0
    )


@jax.jit
def step(
    state: GameState,
    actions: jnp.ndarray,
) -> Tuple[GameState, GameInfo]:
    """
    Execute one game step with actions from both players.
    
    Args:
        state: Current game state
        actions: [2, 5] array of actions for both players
                 Each action is [pass, row, col, direction, split]
    
    Returns:
        (new_state, info) tuple where info is GameInfo NamedTuple
    """
    done_before = state.winner >= 0
    
    # Determine move order based on priority (matching game.py lines 1235-1280)
    # Priority: 1) chasing moves, 2) reinforcing moves, 3) bigger army
    first_player = _determine_move_order(state, actions)
    second_player = 1 - first_player
    
    # Execute actions sequentially based on computed priority
    state = execute_action(state, first_player, actions[first_player])
    state = execute_action(state, second_player, actions[second_player])
    
    # Increment time only if game wasn't done before actions
    state = lax.cond(
        done_before,
        lambda s: s,  # Don't increment time if already done
        lambda s: s._replace(time=s.time + 1),
        state
    )
    
    # Global updates (if game not done)
    state = lax.cond(
        state.winner >= 0,
        lambda s: _transfer_loser_cells_to_winner(s),  # Transfer cells if game done
        lambda s: global_update(s),  # Otherwise do global update
        state
    )
    
    # Compute info
    info = get_info(state)
    
    return state, info


def _transfer_loser_cells_to_winner(state: GameState) -> GameState:
    """Transfer all of loser's cells to winner when game ends."""
    winner_idx = state.winner
    loser_idx = 1 - winner_idx
    
    # Transfer ownership
    new_ownership = state.ownership.at[winner_idx].set(
        state.ownership[winner_idx] | state.ownership[loser_idx]
    )
    new_ownership = new_ownership.at[loser_idx].set(
        jnp.zeros_like(state.ownership[loser_idx], dtype=bool)
    )
    
    # Also update neutral ownership (loser's cells are no longer neutral)
    new_ownership_neutral = state.ownership_neutral & ~state.ownership[loser_idx]
    
    return state._replace(
        ownership=new_ownership,
        ownership_neutral=new_ownership_neutral
    )


@jax.jit
def get_info(state: GameState) -> GameInfo:
    """Compute game info (army/land counts, done status)."""
    armies = state.armies
    ownership = state.ownership
    
    army_0 = jnp.sum(armies * ownership[0])
    army_1 = jnp.sum(armies * ownership[1])
    land_0 = jnp.sum(ownership[0])
    land_1 = jnp.sum(ownership[1])
    
    is_done = state.winner >= 0
    
    return GameInfo(
        army=jnp.stack([army_0, army_1]),
        land=jnp.stack([land_0, land_1]),
        is_done=is_done,
        winner=state.winner,
        time=state.time,
    )


@jax.jit
def get_observation(state: GameState, player_idx: int) -> ObservationJax:
    """
    Get observation for a specific player (with fog of war).
    
    Args:
        state: Current game state
        player_idx: Player index (0 or 1)
    
    Returns:
        ObservationJax named tuple with observation arrays
    """
    visible = get_visibility(state.ownership[player_idx])
    invisible = ~visible
    
    opponent_idx = 1 - player_idx
    
    # Apply visibility mask
    armies = state.armies * visible
    mountains = state.mountains * visible
    generals = state.generals * visible
    cities = state.cities * visible
    neutral_cells = state.ownership_neutral * visible
    owned_cells = state.ownership[player_idx] * visible
    opponent_cells = state.ownership[opponent_idx] * visible
    
    # Fog and structures in fog
    structures_in_fog = invisible & (state.mountains | state.cities)
    fog_cells = invisible & ~structures_in_fog
    
    # Compute stats
    info = get_info(state)
    
    return ObservationJax(
        armies=armies,
        generals=generals,
        cities=cities,
        mountains=mountains,
        neutral_cells=neutral_cells,
        owned_cells=owned_cells,
        opponent_cells=opponent_cells,
        fog_cells=fog_cells,
        structures_in_fog=structures_in_fog,
        owned_land_count=info.land[player_idx],
        owned_army_count=info.army[player_idx],
        opponent_land_count=info.land[opponent_idx],
        opponent_army_count=info.army[opponent_idx],
        timestep=state.time,
        priority=jnp.int32(0),  # Backwards compatibility
    )


# Vectorized versions for batched execution
@jax.jit
def batch_step(
    states: GameState,
    actions: jnp.ndarray,
) -> Tuple[GameState, GameInfo]:
    """
    Vectorized step for multiple environments.
    
    Args:
        states: Batched state dict where each array has leading batch dimension
        actions: [B, 2, 5] array of actions
    
    Returns:
        (batched_states, batched_info) where info is batched GameInfo
    """
    return jax.vmap(step)(states, actions)
