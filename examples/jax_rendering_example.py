"""
Example demonstrating rendering with the vectorized JAX environment.

Shows how to visualize one environment while running multiple in parallel.
Uses valid action sampling to ensure players actually move.

Player 0 (Red): Random valid actions
Player 1 (Blue): Expander agent (prioritizes captures)
"""
import time
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.envs import VectorizedJaxEnv
from generals.core.action_jax import sample_valid_action_jax
from generals.agents.expander_agent_jax import expander_agent_jax


def sample_mixed_actions_batch(
    key: jnp.ndarray,
    observations,
) -> jnp.ndarray:
    """
    Sample actions where P0 is random and P1 is expander agent.
    
    Args:
        key: JAX random key
        observations: Batched ObservationJax [num_envs, 2, H, W, ...]
    
    Returns:
        Actions array [num_envs, 2, 5]
    """
    from generals.core.observation_jax import ObservationJax
    
    # Infer num_envs from observations shape
    num_envs = observations.armies.shape[0]
    
    # Split keys for each environment (each env will split for 2 players)
    env_keys = jrandom.split(key, num_envs)
    
    # Vectorize over environments
    def sample_for_env(env_key, env_obs):
        # env_obs is [2, H, W, ...] for both players
        # Extract individual player observations
        # env_obs.armies has shape [2, H, W]
        # We need to create ObservationJax for each player
        
        # Split key for 2 players
        k1, k2 = jrandom.split(env_key, 2)
        
        # Extract player 0 observation
        obs_p0 = ObservationJax(
            armies=env_obs.armies[0],
            generals=env_obs.generals[0],
            cities=env_obs.cities[0],
            mountains=env_obs.mountains[0],
            neutral_cells=env_obs.neutral_cells[0],
            owned_cells=env_obs.owned_cells[0],
            opponent_cells=env_obs.opponent_cells[0],
            fog_cells=env_obs.fog_cells[0],
            structures_in_fog=env_obs.structures_in_fog[0],
            owned_land_count=env_obs.owned_land_count[0],
            owned_army_count=env_obs.owned_army_count[0],
            opponent_land_count=env_obs.opponent_land_count[0],
            opponent_army_count=env_obs.opponent_army_count[0],
            timestep=env_obs.timestep[0],
            priority=env_obs.priority[0],
        )
        
        # Extract player 1 observation
        obs_p1 = ObservationJax(
            armies=env_obs.armies[1],
            generals=env_obs.generals[1],
            cities=env_obs.cities[1],
            mountains=env_obs.mountains[1],
            neutral_cells=env_obs.neutral_cells[1],
            owned_cells=env_obs.owned_cells[1],
            opponent_cells=env_obs.opponent_cells[1],
            fog_cells=env_obs.fog_cells[1],
            structures_in_fog=env_obs.structures_in_fog[1],
            owned_land_count=env_obs.owned_land_count[1],
            owned_army_count=env_obs.owned_army_count[1],
            opponent_land_count=env_obs.opponent_land_count[1],
            opponent_army_count=env_obs.opponent_army_count[1],
            timestep=env_obs.timestep[1],
            priority=env_obs.priority[1],
        )
        
        # Player 0: Random valid action
        action_p0 = sample_valid_action_jax(k1, obs_p0)
        
        # Player 1: Expander agent
        action_p1 = expander_agent_jax(k2, obs_p1)
        
        return jnp.stack([action_p0, action_p1])
    
    # vmap over environments
    actions = jax.vmap(sample_for_env)(env_keys, observations)
    
    return actions


def main():
    """Run environments with rendering."""
    # Configuration
    num_envs = 16  # Use 1 for debugging, increase for parallel training
    grid_size = (10, 10)  # Larger grid = longer games
    max_steps = 2000
    
    print(f"\nJAX Environment Rendering Example")
    print(f"=" * 60)
    print(f"\nConfiguration:")
    print(f"  Number of parallel environments: {num_envs}")
    print(f"  Grid size: {grid_size}")
    print(f"  Rendering environment #0")
    print(f"  Max steps: {max_steps}")
    print(f"\nNote: Games may end quickly (~20-50 steps) because:")
    print(f"  - Player 0 (Red): Random valid moves")
    print(f"  - Player 1 (Blue): Expander agent (smart, aggressive)")
    print(f"  - Expander prioritizes captures and expansion")
    print(f"\nPress ESC or close window to quit")
    print(f"=" * 60)
    
    # Create environment with rendering enabled
    env = VectorizedJaxEnv(
        num_envs=num_envs,
        grid_size=grid_size,
        render_mode='human',  # Enable pygame rendering
        agent_names=['Random (Red)', 'Expander (Blue)'],
        render_env_index=0,  # Render the first environment
        speed_multiplier=0.5,  # Slower for better visibility
    )
    
    # Initialize JAX random key
    rng_key = jrandom.PRNGKey(42)
    
    # Reset
    obs, info = env.reset(seed=42)
    
    print("\nRunning simulation with rendering...")
    print("(The other 15 environments are running in the background)")
    print("Player 0 (Red): Random valid moves")
    print("Player 1 (Blue): Expander agent - watch it expand and capture!\n")
    
    total_steps = 0
    total_resets = 0
    start_time = time.time()
    
    try:
        for step in range(max_steps):
            # Generate actions: P0=random, P1=expander
            rng_key, subkey = jrandom.split(rng_key)
            actions = sample_mixed_actions_batch(subkey, obs)
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Render (shows environment #0)
            try:
                env.render()
            except SystemExit as e:
                print(f"\n{e}")
                break
            
            total_steps += num_envs
            total_resets += jnp.sum(terminated).item()
            
            # Break if the rendered environment (index 0) terminates
            # (Other environments will auto-reset and keep running)
            if terminated[0]:
                print(f"\nGame ended after {step+1} steps!")
                print(f"Winner: Player {info.winner[0]}")
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    elapsed = time.time() - start_time
    
    # Close environment
    env.close()
    
    print(f"\n" + "=" * 60)
    print(f"Statistics:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Total auto-resets: {total_resets}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {total_steps / elapsed:,.0f} steps/sec")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
