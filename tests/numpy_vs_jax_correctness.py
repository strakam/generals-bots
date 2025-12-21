"""
Test correctness of JAX environment vs NumPy environment.
Ensures both implementations produce identical results using shared grid factory.
"""

import numpy as np
import jax
import jax.numpy as jnp

from generals.envs import GymnasiumGenerals
from generals.envs.jax_env import VectorizedJaxEnv
from generals import GridFactory
from generals.core.action import Action
from generals.core.grid import Grid


def test_numpy_vs_jax():
    """Compare NumPy and JAX environments using same grid factory."""
    print("\n=== Testing NumPy vs JAX Equivalence ===")
    
    seed = 42
    num_steps = 2000
    
    # Configurable grid size
    grid_width = 6
    grid_height = 6
    
    # Create grid factories with same seed for both environments
    grid_factory_np = GridFactory(
        min_grid_dims=(grid_height, grid_width),
        max_grid_dims=(grid_height, grid_width),
        mountain_density=0.2,
        city_density=0.05,
        general_positions=[[2, 2], [grid_height - 1, grid_width - 1]],
        seed=seed
    )
    
    grid_factory_jax = GridFactory(
        min_grid_dims=(grid_height, grid_width),
        max_grid_dims=(grid_height, grid_width),
        mountain_density=0.2,
        city_density=0.05,
        general_positions=[[2, 2], [grid_height - 1, grid_width - 1]],
        seed=seed  # Same seed ensures same grid generation
    )
    
    # Create both environments
    np_env = GymnasiumGenerals(
        agents=["Player1", "Player2"],
        grid_factory=grid_factory_np,
        truncation=5000,
        pad_observations_to=grid_height,  # Match grid size
    )
    
    jax_env = VectorizedJaxEnv(
        num_envs=1,
        grid_size=(grid_height, grid_width),
        grid_factory=grid_factory_jax,
    )
    
    # Reset both environments
    np_obs_array, np_info = np_env.reset(seed=seed)
    jax_obs, jax_info = jax_env.reset(seed=seed)
    
    success = True
    
    # Run steps with same actions
    for step in range(num_steps):
        # Generate deterministic actions
        np.random.seed(seed + step)
        
        # Get NumPy action masks from info
        np_masks = np_info.get('action_mask', None)  # Shape: (2, num_actions)
        
        # Generate masked random actions for both players
        actions_np = []
        actions_jax_list = []
        
        for player_idx in range(2):
            # Get valid actions for this player using NumPy mask
            if np_masks is not None:
                valid_actions = np.where(np_masks[player_idx])[0]
            else:
                valid_actions = np.arange(grid_height * grid_width * 4 * 2 + 1)
            
            if len(valid_actions) == 0:
                # No valid actions, pass
                action_np = Action(to_pass=True)
                action_jax = [1, 0, 0, 0, 0]
            else:
                # Sample a random valid action
                action_idx = np.random.choice(valid_actions)
                
                if action_idx == 0:  # Pass action
                    action_np = Action(to_pass=True)
                    action_jax = [1, 0, 0, 0, 0]
                else:
                    # Decode action index to (row, col, direction, split)
                    action_idx -= 1  # Remove pass action offset
                    split = action_idx % 2
                    action_idx //= 2
                    direction = action_idx % 4
                    action_idx //= 4
                    col = action_idx % grid_width
                    row = action_idx // grid_width
                    
                    action_np = Action(to_pass=False, row=row, col=col, direction=direction, to_split=bool(split))
                    action_jax = [0, row, col, direction, split]
            
            actions_np.append(action_np)
            actions_jax_list.append(action_jax)
        
        actions_jax = jnp.array([actions_jax_list], dtype=jnp.int32)  # Shape: (1, 2, 5)
        
        # Step both environments
        np_obs_array, np_reward, np_terminated, np_truncated, np_info = np_env.step(actions_np)
        jax_obs, jax_rewards, jax_terminated, jax_truncated, jax_info = jax_env.step(actions_jax)
        
        # Compare observations (NumPy: (2,15,h,w), JAX: (1,2,15,h,w))
        np_obs_tensor = np_obs_array
        jax_obs_tensor = np.array(jax_obs.as_tensor()[0])
        
        print(f"\nStep {step}:")
        
        # Compare observations channel by channel
        channel_names = [
            "armies", "generals", "cities", "mountains", "neutral", "owned",
            "opponent_cells", "fog", "structures_in_fog", "armies_in_fog",
            "opponent_territory", "priority_turn1", "priority_turn2",
            "ownership_p1", "ownership_p2"
        ]
        
        obs_match = True
        for player_idx in range(2):
            for ch_idx in range(15):
                np_channel = np_obs_tensor[player_idx, ch_idx]
                jax_channel = jax_obs_tensor[player_idx, ch_idx]
                
                if not np.allclose(np_channel, jax_channel, rtol=1e-5, atol=1e-5):
                    print(f"  ❌ Player {player_idx}, Channel {ch_idx} ({channel_names[ch_idx]}) differs!")
                    print(f"     Max diff: {np.abs(np_channel - jax_channel).max()}")
                    print(f"     NumPy channel:\n{np_channel.astype(int)}")
                    print(f"     JAX channel:\n{jax_channel.astype(int)}")
                    obs_match = False
                    success = False
        
        if obs_match:
            print(f"  ✓ Observations match")
        else:
            print(f"  ❌ Observations differ!")
        
        # Compare terminated/truncated
        np_terminated_val = bool(np_terminated)
        np_truncated_val = bool(np_truncated)
        jax_terminated_val = bool(jax_terminated[0])
        jax_truncated_val = bool(jax_truncated[0])
        
        if np_terminated_val != jax_terminated_val:
            print(f"  ❌ Terminated differs: numpy={np_terminated_val}, jax={jax_terminated_val}")
            success = False
        else:
            print(f"  ✓ Terminated match: {np_terminated_val}")
            
        if np_truncated_val != jax_truncated_val:
            print(f"  ❌ Truncated differs: numpy={np_truncated_val}, jax={jax_truncated_val}")
            success = False
        else:
            print(f"  ✓ Truncated match: {np_truncated_val}")
        
        # Stop if either environment terminates
        if np_terminated_val or np_truncated_val or jax_terminated_val or jax_truncated_val:
            print(f"\nEnvironment terminated/truncated at step {step}")
            break

        if not success:
            return
    
    if success:
        print("\n✅ All checks passed! NumPy and JAX implementations are equivalent.")
    else:
        print("\n❌ Some checks failed! NumPy and JAX implementations differ.")
    
    return success


if __name__ == "__main__":
    test_numpy_vs_jax()
