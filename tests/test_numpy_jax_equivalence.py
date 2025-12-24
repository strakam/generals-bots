"""
Test equivalence between NumPy (Gymnasium) and JAX environments.

This test verifies that both environments produce identical results when:
- Starting from the same grid
- Receiving the same sequence of actions
- Comparing ALL elements of observations and infos
"""

import jax
import jax.numpy as jnp
import numpy as np

from generals.core.action import Action
from generals.core.grid import GridFactory
from generals.core.game import Game
from generals.core import game_jax
from generals.core.observation import Observation


def grid_to_numeric(grid: np.ndarray) -> np.ndarray:
    """
    Convert Grid's ASCII representation to JAX numeric format.
    
    Mapping:
    - '.' (passable) -> 0
    - '#' (mountain) -> -2
    - 'A' (general player 0) -> 1
    - 'B' (general player 1) -> 2
    - '0'-'9' (cities) -> 40-49 (40 + digit value)
    - 'x' (city) -> 50 (city with value 50)
    
    Args:
        grid: NumPy character array from Grid.grid
    
    Returns:
        Numeric array for JAX
    """
    numeric_grid = np.zeros(grid.shape, dtype=np.int8)
    
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            char = grid[i, j]
            if char == '.':
                numeric_grid[i, j] = 0
            elif char == '#':
                numeric_grid[i, j] = -2
            elif char == 'A':
                numeric_grid[i, j] = 1
            elif char == 'B':
                numeric_grid[i, j] = 2
            elif char == 'x':
                # 'x' is a city with value 50
                numeric_grid[i, j] = 50
            elif char.isdigit():
                # Cities are '0'-'9', convert to 40-49
                numeric_grid[i, j] = 40 + int(char)
            else:
                # Unknown character - treat as passable
                numeric_grid[i, j] = 0
    
    return numeric_grid


def action_to_jax(action: Action) -> jnp.ndarray:
    """Convert Action to JAX format [to_pass, row, col, direction, to_split]."""
    return jnp.array([int(action[i]) for i in range(5)], dtype=jnp.int32)


def compare_observations(numpy_obs: Observation, jax_obs, player_name: str, step: int):
    """
    Compare observations from NumPy Game and JAX game_jax.
    
    jax_obs is an ObservationJax NamedTuple
    
    Returns:
        List of difference messages (empty if all equal)
    """
    differences = []
    
    # Compare scalar fields
    scalar_fields = [
        ('owned_land_count', 'owned_land_count'),
        ('owned_army_count', 'owned_army_count'),
        ('opponent_land_count', 'opponent_land_count'),
        ('opponent_army_count', 'opponent_army_count'),
        ('timestep', 'timestep'),
    ]
    
    for numpy_field, jax_field in scalar_fields:
        numpy_val = getattr(numpy_obs, numpy_field)
        jax_val = int(getattr(jax_obs, jax_field))
        
        if numpy_val != jax_val:
            differences.append(
                f"Step {step}, {player_name}, {numpy_field}: "
                f"NumPy={numpy_val}, JAX={jax_val}"
            )
    
    # Compare array fields
    array_fields = [
        ('armies', 'armies'),
        ('generals', 'generals'),
        ('cities', 'cities'),
        ('mountains', 'mountains'),
        ('neutral_cells', 'neutral_cells'),
        ('owned_cells', 'owned_cells'),
        ('opponent_cells', 'opponent_cells'),
        ('fog_cells', 'fog_cells'),
        ('structures_in_fog', 'structures_in_fog'),
    ]
    
    for numpy_field, jax_field in array_fields:
        numpy_arr = getattr(numpy_obs, numpy_field)
        jax_arr = np.array(getattr(jax_obs, jax_field))
        
        if not np.array_equal(numpy_arr, jax_arr):
            diff_count = np.sum(numpy_arr != jax_arr)
            differences.append(
                f"Step {step}, {player_name}, {numpy_field}: "
                f"{diff_count} cells differ"
            )
    
    return differences


def compare_infos(numpy_info: dict, jax_info, player_name: str, step: int):
    """
    Compare info dicts from NumPy Game and JAX game_jax.
    
    jax_info is a GameInfo NamedTuple
    
    Returns:
        List of difference messages (empty if all equal)
    """
    differences = []
    
    # NumPy game returns info with player-specific fields
    player_idx = 0 if player_name == "agent_0" else 1
    
    # Compare army and land counts
    numpy_army = numpy_info[player_name]['army']
    jax_army = int(jax_info.army[player_idx])
    
    if numpy_army != jax_army:
        differences.append(
            f"Step {step}, {player_name}: Army count differs - "
            f"NumPy={numpy_army}, JAX={jax_army}"
        )
    
    numpy_land = numpy_info[player_name]['land']
    jax_land = int(jax_info.land[player_idx])
    
    if numpy_land != jax_land:
        differences.append(
            f"Step {step}, {player_name}: Land count differs - "
            f"NumPy={numpy_land}, JAX={jax_land}"
        )
    
    return differences


def test_numpy_jax_equivalence(num_games=10, steps_per_game=2000, seed=42):
    """
    Test that NumPy and JAX game implementations produce identical results.
    
    For N games in sequence:
    1. Generate the same grid for both implementations
    2. Apply the same sequence of random actions
    3. Verify ALL observations and infos are equal at each step
    
    Args:
        num_games: Number of games to test
        steps_per_game: Maximum number of steps per game
        seed: Random seed for reproducibility
    """
    grid_factory = GridFactory(
        min_grid_dims=(10, 10),
        max_grid_dims=(10, 10),
        mountain_density=0.15,
        city_density=0.1,
        general_positions=[[1, 1], [8, 8]],
    )
    
    np.random.seed(seed)
    all_differences = []
    
    # JIT compile JAX step function once
    jitted_step = jax.jit(game_jax.step)
    
    for game_idx in range(num_games):
        print(f"Testing game {game_idx + 1}/{num_games}...")
        
        # Generate a single grid for both implementations
        grid = grid_factory.generate()
        
        # Create NumPy game
        numpy_game = Game(grid, ["agent_0", "agent_1"])
        
        # Create JAX state from same grid
        grid_array = grid_to_numeric(grid.grid)
        jax_state = game_jax.create_initial_state(jnp.array(grid_array))
        
        # Get and compare initial observations
        numpy_obs_dict = {
            "agent_0": numpy_game.agent_observation("agent_0"),
            "agent_1": numpy_game.agent_observation("agent_1"),
        }
        numpy_info = numpy_game.get_infos()
        
        jax_obs_0 = game_jax.get_observation(jax_state, 0)
        jax_obs_1 = game_jax.get_observation(jax_state, 1)
        jax_info = game_jax.get_info(jax_state)
        
        # Compare initial state
        all_differences.extend(compare_observations(numpy_obs_dict["agent_0"], jax_obs_0, "agent_0", 0))
        all_differences.extend(compare_observations(numpy_obs_dict["agent_1"], jax_obs_1, "agent_1", 0))
        all_differences.extend(compare_infos(numpy_info, jax_info, "agent_0", 0))
        all_differences.extend(compare_infos(numpy_info, jax_info, "agent_1", 0))
        
        # Run game with same actions
        for step in range(1, steps_per_game + 1):
            # Generate random actions
            actions = []
            for _ in range(2):
                if np.random.random() < 0.3:
                    actions.append(Action(to_pass=True))
                else:
                    actions.append(Action(
                        to_pass=False,
                        row=np.random.randint(0, 10),
                        col=np.random.randint(0, 10),
                        direction=np.random.randint(0, 4),
                        to_split=bool(np.random.randint(0, 2)),
                    ))
            
            # Step NumPy game
            numpy_actions = {"agent_0": actions[0], "agent_1": actions[1]}
            numpy_obs_dict, numpy_info = numpy_game.step(numpy_actions)
            
            # Step JAX game
            jax_actions = jnp.stack([action_to_jax(actions[0]), action_to_jax(actions[1])])
            jax_state, jax_step_info = jitted_step(jax_state, jax_actions)
            
            # Get JAX observations
            jax_obs_0 = game_jax.get_observation(jax_state, 0)
            jax_obs_1 = game_jax.get_observation(jax_state, 1)
            jax_info = game_jax.get_info(jax_state)
            
            # Compare observations and infos
            all_differences.extend(compare_observations(numpy_obs_dict["agent_0"], jax_obs_0, "agent_0", step))
            all_differences.extend(compare_observations(numpy_obs_dict["agent_1"], jax_obs_1, "agent_1", step))
            all_differences.extend(compare_infos(numpy_info, jax_info, "agent_0", step))
            all_differences.extend(compare_infos(numpy_info, jax_info, "agent_1", step))
            
            # Check if game is done
            if numpy_game.is_done() or jax_state.winner >= 0:
                break
    
    # Report results
    if all_differences:
        print("\n" + "=" * 70)
        print("DIFFERENCES FOUND:")
        print("=" * 70)
        for diff in all_differences[:50]:  # Show first 50
            print(f"  {diff}")
        if len(all_differences) > 50:
            print(f"  ... and {len(all_differences) - 50} more")
        print("=" * 70)
        raise AssertionError(
            f"Found {len(all_differences)} differences between NumPy and JAX implementations"
        )
    else:
        print("\n" + "=" * 70)
        print("✅ All tests passed! NumPy and JAX are equivalent.")
        print("=" * 70)


if __name__ == "__main__":
    test_numpy_jax_equivalence(num_games=10, steps_per_game=2000, seed=42)
