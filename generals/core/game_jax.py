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
    Create initial game state from a grid.
    
    Args:
        grid: String array where:
            'A', 'B' = generals for player 0, 1
            '0'-'9', 'x' = cities
            '#' = mountains
            '.' = passable/neutral
    
    Returns:
        GameState NamedTuple containing all game state arrays
    """
    H, W = grid.shape
    
    # Decode grid
    is_general_0 = (grid == ord('A'))
    is_general_1 = (grid == ord('B'))
    generals = (is_general_0 | is_general_1).astype(jnp.bool_)
    
    mountains = (grid == ord('#')).astype(jnp.bool_)
    passable = ~mountains
    
    # Cities: digits 0-9 and 'x' (which represents 50)
    is_digit = (grid >= ord('0')) & (grid <= ord('9'))
    is_x = (grid == ord('x'))
    cities = (is_digit | is_x).astype(jnp.bool_)
    
    # Initial ownership
    ownership = jnp.stack([
        is_general_0.astype(jnp.bool_),
        is_general_1.astype(jnp.bool_),
    ])
    
    # Neutral cells (passable but not owned)
    ownership_neutral = (passable & ~is_general_0 & ~is_general_1).astype(jnp.bool_)
    
    # Initial armies
    armies = jnp.where(is_general_0 | is_general_1, 1, 0).astype(jnp.int32)
    
    # Add city armies (40 + digit value, or 50 for 'x')
    city_costs = jnp.where(is_digit, grid - ord('0'), 0)
    city_costs = jnp.where(is_x, 10, city_costs)
    armies = armies + 40 * cities.astype(jnp.int32) + city_costs
    
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
    
    # Check ownership and army requirements
    owns_source = state.ownership[player_idx, si, sj]
    source_army = state.armies[si, sj]
    
    # Calculate army to move
    army_to_move = lax.cond(
        split_army == 1,
        lambda a: a // 2,
        lambda a: a - 1,
        source_army
    )
    army_to_move = jnp.maximum(army_to_move, 0)
    army_to_move = jnp.minimum(army_to_move, source_army - 1)
    
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
    increment_all = (time % 50 == 0) & (time > 0)
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
    
    # Execute actions in order (player 0, then player 1)
    # TODO: Implement proper priority ordering
    state = execute_action(state, 0, actions[0])
    state = execute_action(state, 1, actions[1])
    
    # Increment time
    state = state._replace(time=state.time + 1)
    
    # Global updates (if game not done)
    state = lax.cond(
        state.winner >= 0,
        lambda s: s,  # Skip updates if done
        lambda s: global_update(s),
        state
    )
    
    # Compute info
    info = get_info(state)
    
    return state, info


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
