"""
Simple test to verify JAX observation tensor format is correct.

This validates that the 15 channels in the observation tensor
are in the expected order and contain reasonable values.
"""

import numpy as np
import jax

from generals.envs.jax_env import VectorizedJaxEnv


def test_jax_obs_format():
    """Test that JAX observation has correct format and channels."""
    
    # Create JAX env
    jax_env = VectorizedJaxEnv(
        num_envs=2,
        grid_size=(10, 10),
        mountain_density=0.2,
        city_density=0.05,
    )
    
    # Reset
    obs, _ = jax_env.reset(seed=42)
    
    # Get observation tensor
    obs_tensor = np.array(obs.as_tensor())
    print(f"Observation tensor shape: {obs_tensor.shape}")
    print(f"Expected: (num_envs=2, num_players=2, channels=15, height=10, width=10)")
    
    assert obs_tensor.shape == (2, 2, 15, 10, 10), f"Wrong shape: {obs_tensor.shape}"
    
    # Check each channel makes sense
    channel_names = [
        'armies', 'generals', 'cities', 'mountains', 'neutral_cells',
        'owned_cells', 'opponent_cells', 'fog_cells', 'structures_in_fog',
        'owned_land_count', 'owned_army_count', 'opponent_land_count',
        'opponent_army_count', 'timestep', 'priority'
    ]
    
    print("\nChannel validation:")
    for env_idx in range(2):
        for player_idx in range(2):
            print(f"\nEnv {env_idx}, Player {player_idx}:")
            
            # Channel 1: generals - should have exactly 1 cell = 1 (player's general)
            generals = obs_tensor[env_idx, player_idx, 1]
            num_own_general = np.sum(generals == 1)
            print(f"  Generals (own): {num_own_general} cells")
            assert num_own_general == 1, f"Should have 1 own general, found {num_own_general}"
            
            # Channel 3: mountains - should be 0 or 1
            mountains = obs_tensor[env_idx, player_idx, 3]
            assert np.all((mountains == 0) | (mountains == 1)), "Mountains should be 0 or 1"
            num_mountains = np.sum(mountains == 1)
            print(f"  Mountains: {num_mountains} cells")
            
            # Channel 5: owned_cells - should be 0 or 1
            owned = obs_tensor[env_idx, player_idx, 5]
            assert np.all((owned == 0) | (owned == 1)), "Owned cells should be 0 or 1"
            num_owned = np.sum(owned == 1)
            print(f"  Owned cells: {num_owned} cells")
            
            # Channel 6: opponent_cells - should be 0 or 1
            opponent = obs_tensor[env_idx, player_idx, 6]
            assert np.all((opponent == 0) | (opponent == 1)), "Opponent cells should be 0 or 1"
            num_opponent = np.sum(opponent == 1)
            print(f"  Opponent cells: {num_opponent} cells")
            
            # Channel 9: owned_land_count - should be constant across all cells
            owned_count = obs_tensor[env_idx, player_idx, 9]
            assert np.all(owned_count == owned_count[0, 0]), "owned_land_count should be constant"
            print(f"  Owned land count: {owned_count[0, 0]}")
            
            # Channel 13: timestep - should be constant and = 0 at start
            timestep = obs_tensor[env_idx, player_idx, 13]
            assert np.all(timestep == timestep[0, 0]), "Timestep should be constant"
            assert timestep[0, 0] == 0, f"Timestep should be 0 at start, got {timestep[0, 0]}"
            print(f"  Timestep: {timestep[0, 0]}")
    
    # Take some steps
    print("\n\nTaking 10 steps...")
    for step in range(10):
        # Random actions in format (pass, row, col, direction, split)
        # Shape: (2 envs, 2 players, 5)
        actions = np.zeros((2, 2, 5), dtype=np.int32)
        actions[:, :, 0] = np.random.randint(0, 2, size=(2, 2))  # pass
        actions[:, :, 1] = np.random.randint(0, 10, size=(2, 2))  # row
        actions[:, :, 2] = np.random.randint(0, 10, size=(2, 2))  # col
        actions[:, :, 3] = np.random.randint(0, 4, size=(2, 2))   # direction
        actions[:, :, 4] = 0  # split (disabled for simplicity)
        
        obs, rewards, terminated, truncated, info = jax_env.step(actions)
        
        obs_tensor = np.array(obs.as_tensor())
        
        # Check timestep increases
        timestep = obs_tensor[0, 0, 13, 0, 0]
        print(f"Step {step + 1}: timestep={timestep}")
        assert timestep == step + 1, f"Timestep should be {step + 1}, got {timestep}"
    
    print("\n✅ All checks passed! JAX observation format is correct.")


if __name__ == "__main__":
    test_jax_obs_format()
