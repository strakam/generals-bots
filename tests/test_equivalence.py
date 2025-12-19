"""
Test equivalence between NumPy and JAX implementations.

This test creates random game instances and verifies that both implementations
produce identical results when given the same sequence of actions.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from generals.core.game import Game
from generals.core import game_jax
from generals.core.grid import GridFactory
from generals.core.action import Action


def create_matched_games(grid_factory):
    """
    Create a NumPy game and corresponding JAX state from the same grid.
    
    Returns:
        (numpy_game, jax_state) tuple
    """
    grid = grid_factory.generate()
    
    # Create NumPy game
    numpy_game = Game(grid, ["agent_0", "agent_1"])
    
    # Create JAX state from same grid
    grid_array = np.vectorize(ord)(grid.grid).astype(np.uint8)
    grid_jax = jnp.array(grid_array)
    jax_state = game_jax.create_initial_state(grid_jax)
    
    return numpy_game, jax_state, grid


def action_to_jax(action: Action, player_idx: int) -> jnp.ndarray:
    """Convert NumPy Action to JAX action array for a specific player."""
    return jnp.array([
        int(action[0]),  # pass
        int(action[1]),  # row
        int(action[2]),  # col
        int(action[3]),  # direction
        int(action[4]),  # split
    ], dtype=jnp.int32)


def random_action(height: int, width: int) -> Action:
    """Generate a random action."""
    # 30% chance to pass
    if np.random.random() < 0.3:
        return Action(to_pass=True)
    
    return Action(
        to_pass=False,
        row=np.random.randint(0, height),
        col=np.random.randint(0, width),
        direction=np.random.randint(0, 4),
        to_split=bool(np.random.randint(0, 2)),
    )


def compare_observations(numpy_obs, jax_obs, player_name: str, step: int):
    """
    Compare observations from NumPy and JAX implementations.
    
    Returns:
        List of differences found, empty if observations match.
    """
    differences = []
    
    # Map player name to index for JAX
    player_idx = 0 if player_name == "agent_0" else 1
    
    # Get JAX observation for this player
    jax_player_obs = jax_obs[player_idx]
    
    # Compare scalar values
    scalar_fields = [
        ('owned_land_count', 'owned_land_count'),
        ('owned_army_count', 'owned_army_count'),
        ('opponent_land_count', 'opponent_land_count'),
        ('opponent_army_count', 'opponent_army_count'),
        ('timestep', 'timestep'),
    ]
    
    for numpy_field, jax_field in scalar_fields:
        numpy_val = getattr(numpy_obs, numpy_field)
        jax_val = int(jax_player_obs[jax_field])
        
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
        jax_arr = np.array(jax_player_obs[jax_field])
        
        if not np.array_equal(numpy_arr, jax_arr):
            # Count differences
            diff_count = np.sum(numpy_arr != jax_arr)
            differences.append(
                f"Step {step}, {player_name}, {numpy_field}: "
                f"{diff_count} cells differ"
            )
            
            # Show first few differences
            diff_indices = np.argwhere(numpy_arr != jax_arr)
            if len(diff_indices) > 0:
                for idx in diff_indices[:3]:  # Show first 3
                    i, j = idx
                    differences.append(
                        f"  at ({i},{j}): NumPy={numpy_arr[i,j]}, JAX={jax_arr[i,j]}"
                    )
    
    return differences


def compare_game_state(numpy_game: Game, jax_state: dict, step: int):
    """
    Compare core game state between NumPy and JAX implementations.
    
    Returns:
        List of differences found.
    """
    differences = []
    
    # Compare armies
    numpy_armies = numpy_game.channels.armies
    jax_armies = np.array(jax_state['armies'])
    
    if not np.array_equal(numpy_armies, jax_armies):
        diff_count = np.sum(numpy_armies != jax_armies)
        differences.append(f"Step {step}: Armies differ in {diff_count} cells")
        
        # Show a few differences
        diff_indices = np.argwhere(numpy_armies != jax_armies)
        for idx in diff_indices[:5]:
            i, j = idx
            differences.append(
                f"  Armies at ({i},{j}): NumPy={numpy_armies[i,j]}, JAX={jax_armies[i,j]}"
            )
    
    # Compare ownership
    for player_idx, player_name in enumerate(["agent_0", "agent_1"]):
        numpy_ownership = numpy_game.channels.ownership[player_name]
        jax_ownership = np.array(jax_state['ownership'][player_idx])
        
        if not np.array_equal(numpy_ownership, jax_ownership):
            diff_count = np.sum(numpy_ownership != jax_ownership)
            differences.append(
                f"Step {step}: {player_name} ownership differs in {diff_count} cells"
            )
    
    # Compare winner
    numpy_winner = -1 if numpy_game.winner is None else (0 if numpy_game.winner == "agent_0" else 1)
    jax_winner = int(jax_state['winner'])
    
    if numpy_winner != jax_winner:
        differences.append(
            f"Step {step}: Winner differs - NumPy={numpy_winner}, JAX={jax_winner}"
        )
    
    # Compare time
    if numpy_game.time != int(jax_state['time']):
        differences.append(
            f"Step {step}: Time differs - NumPy={numpy_game.time}, JAX={jax_state['time']}"
        )
    
    return differences


def test_single_game_equivalence():
    """Test that a single game produces identical results."""
    import jax
    
    grid_factory = GridFactory(
        min_grid_dims=(8, 8),
        max_grid_dims=(8, 8),
        mountain_density=0.1,
        city_density=0.1,
        general_positions=[[1, 1], [6, 6]],
    )
    
    numpy_game, jax_state, grid = create_matched_games(grid_factory)
    height, width = numpy_game.grid_dims
    
    # JIT compile
    jitted_step = jax.jit(game_jax.step)
    
    all_differences = []
    
    # Run 50 steps with same random actions
    for step in range(50):
        # Generate same actions for both
        action_0 = random_action(height, width)
        action_1 = random_action(height, width)
        
        # NumPy step
        numpy_actions = {"agent_0": action_0, "agent_1": action_1}
        numpy_obs_dict, numpy_info = numpy_game.step(numpy_actions)
        
        # JAX step
        jax_actions = jnp.stack([
            action_to_jax(action_0, 0),
            action_to_jax(action_1, 1),
        ])
        jax_state, jax_info = jitted_step(jax_state, jax_actions)
        
        # Compare game state
        state_diffs = compare_game_state(numpy_game, jax_state, step)
        all_differences.extend(state_diffs)
        
        # Compare observations
        jax_obs = [
            game_jax.get_observation(jax_state, 0),
            game_jax.get_observation(jax_state, 1),
        ]
        
        for player_name in ["agent_0", "agent_1"]:
            obs_diffs = compare_observations(
                numpy_obs_dict[player_name],
                jax_obs,
                player_name,
                step
            )
            all_differences.extend(obs_diffs)
        
        # Stop if game is done
        if numpy_game.is_done() or jax_state['winner'] >= 0:
            break
    
    if all_differences:
        print("\nDifferences found:")
        for diff in all_differences[:20]:  # Print first 20
            print(f"  {diff}")
        if len(all_differences) > 20:
            print(f"  ... and {len(all_differences) - 20} more")
    
    assert len(all_differences) == 0, f"Found {len(all_differences)} differences between implementations"


def test_100_random_games_equivalence():
    """Test 100 random game instances for equivalence."""
    import jax
    
    grid_factory = GridFactory(
        min_grid_dims=(6, 6),
        max_grid_dims=(10, 10),
        mountain_density=0.15,
        city_density=0.1,
    )
    
    # JIT compile the step function once
    jitted_step = jax.jit(game_jax.step)
    
    total_steps = 0
    games_with_differences = 0
    total_differences = 0
    
    for game_idx in range(100):
        numpy_game, jax_state, grid = create_matched_games(grid_factory)
        height, width = numpy_game.grid_dims
        
        game_differences = []
        
        # Run each game for up to 30 steps
        for step in range(30):
            # Generate same random actions
            action_0 = random_action(height, width)
            action_1 = random_action(height, width)
            
            # NumPy step
            numpy_actions = {"agent_0": action_0, "agent_1": action_1}
            numpy_obs_dict, numpy_info = numpy_game.step(numpy_actions)
            
            # JAX step
            jax_actions = jnp.stack([
                action_to_jax(action_0, 0),
                action_to_jax(action_1, 1),
            ])
            jax_state, jax_info = jitted_step(jax_state, jax_actions)
            
            # Compare state
            state_diffs = compare_game_state(numpy_game, jax_state, step)
            game_differences.extend(state_diffs)
            
            # Stop if done
            if numpy_game.is_done() or jax_state['winner'] >= 0:
                break
            
            total_steps += 1
        
        if game_differences:
            games_with_differences += 1
            total_differences += len(game_differences)
            
            if games_with_differences == 1:  # Print details for first failing game
                print(f"\nGame {game_idx} differences:")
                for diff in game_differences[:10]:
                    print(f"  {diff}")
    
    print(f"\nTested {100} games with {total_steps} total steps")
    print(f"Games with differences: {games_with_differences}")
    print(f"Total differences: {total_differences}")
    
    assert games_with_differences == 0, \
        f"{games_with_differences}/100 games had differences (total: {total_differences})"


def test_edge_cases_equivalence():
    """Test specific edge cases for equivalence."""
    import jax
    
    # JIT compile once
    jitted_step = jax.jit(game_jax.step)
    
    test_cases = [
        {
            "name": "Small grid",
            "grid_dims": (4, 4),
            "positions": [[0, 0], [3, 3]],
        },
        {
            "name": "No mountains",
            "grid_dims": (6, 6),
            "positions": [[1, 1], [4, 4]],
            "mountain_density": 0.0,
        },
        {
            "name": "Many cities",
            "grid_dims": (8, 8),
            "positions": [[1, 1], [6, 6]],
            "city_density": 0.3,
        },
    ]
    
    for test_case in test_cases:
        grid_factory = GridFactory(
            min_grid_dims=test_case["grid_dims"],
            max_grid_dims=test_case["grid_dims"],
            general_positions=test_case["positions"],
            mountain_density=test_case.get("mountain_density", 0.1),
            city_density=test_case.get("city_density", 0.1),
        )
        
        numpy_game, jax_state, grid = create_matched_games(grid_factory)
        height, width = numpy_game.grid_dims
        
        differences = []
        
        # Run 20 steps
        for step in range(20):
            action_0 = random_action(height, width)
            action_1 = random_action(height, width)
            
            numpy_actions = {"agent_0": action_0, "agent_1": action_1}
            numpy_game.step(numpy_actions)
            
            jax_actions = jnp.stack([
                action_to_jax(action_0, 0),
                action_to_jax(action_1, 1),
            ])
            jax_state, _ = jitted_step(jax_state, jax_actions)
            
            differences.extend(compare_game_state(numpy_game, jax_state, step))
            
            if numpy_game.is_done():
                break
        
        assert len(differences) == 0, \
            f"Test case '{test_case['name']}' failed with {len(differences)} differences"


if __name__ == "__main__":
    print("Testing NumPy vs JAX equivalence...\n")
    
    print("Test 1: Single game equivalence")
    test_single_game_equivalence()
    print("✓ Passed\n")
    
    print("Test 2: Edge cases")
    test_edge_cases_equivalence()
    print("✓ Passed\n")
    
    print("Test 3: 100 random games")
    test_100_random_games_equivalence()
    print("✓ Passed\n")
    
    print("All equivalence tests passed!")
