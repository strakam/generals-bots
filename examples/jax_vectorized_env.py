"""
Example demonstrating the vectorized JAX environment.

Shows how to use the VectorizedJaxEnv for high-performance training with:
- Automatic environment reset on termination
- JAX random for fast action generation
- Full Gym API compatibility
"""
import time
from typing import Tuple

import jax.numpy as jnp
import jax.random as jrandom

from generals.envs import VectorizedJaxEnv


def random_actions_jax(key: jnp.ndarray, num_envs: int, grid_size: Tuple[int, int]) -> jnp.ndarray:
    """Generate random actions for all environments using JAX random (vectorized)."""
    H, W = grid_size
    
    # Split key for different random operations
    subkeys = jrandom.split(key, 5)
    
    # Generate random values for all actions at once
    pass_vals = jrandom.uniform(subkeys[0], (num_envs, 2)) < 0.3  # 30% chance to pass
    rows = jrandom.randint(subkeys[1], (num_envs, 2), 0, H)
    cols = jrandom.randint(subkeys[2], (num_envs, 2), 0, W)
    directions = jrandom.randint(subkeys[3], (num_envs, 2), 0, 4)
    splits = jrandom.randint(subkeys[4], (num_envs, 2), 0, 2)
    
    # Stack into action arrays
    actions = jnp.stack([
        pass_vals.astype(jnp.int32),
        rows,
        cols,
        directions,
        splits
    ], axis=-1)
    
    return actions


def main():
    """Run a simple training loop demonstration."""
    # Configuration
    num_envs = 128
    grid_size = (20, 20)
    num_episodes = 5
    max_steps_per_episode = 2000
    
    print(f"\nJAX Vectorized Environment Example")
    print(f"=" * 60)
    print(f"\nConfiguration:")
    print(f"  Number of parallel environments: {num_envs}")
    print(f"  Grid size: {grid_size}")
    print(f"  Episodes to run: {num_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    
    # Create environment with seed
    env = VectorizedJaxEnv(num_envs=num_envs, grid_size=grid_size)
    env.seed(42)
    
    # Initialize JAX random key
    rng_key = jrandom.PRNGKey(42)
    
    # Reset and warmup JIT
    obs, info = env.reset()
    print(f"\nObservation type: {type(obs).__name__}")
    print(f"Observation fields: {obs._fields}")
    print(f"Example field shape (armies): {obs.armies.shape}")
    print(f"Info type: {type(info).__name__}")
    
    print("\nWarming up JIT compilation...")
    for _ in range(5):
        rng_key, subkey = jrandom.split(rng_key)
        actions = random_actions_jax(subkey, num_envs, grid_size)
        obs, rewards, terminated, truncated, info = env.step(actions)
    
    total_steps = 0
    total_resets = 0
    times = []
    
    print("\nRunning episodes...")
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = jnp.zeros((num_envs, 2))
        episode_length = jnp.zeros(num_envs)
        
        episode_start = time.time()
        
        for step in range(max_steps_per_episode):
            # Generate random actions using JAX random (FAST!)
            rng_key, subkey = jrandom.split(rng_key)
            actions = random_actions_jax(subkey, num_envs, grid_size)
            
            # Step environment (Gym v0.26+ API with 5 returns)
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Accumulate rewards
            episode_reward += rewards
            episode_length += ~terminated
            total_steps += num_envs
            total_resets += jnp.sum(terminated).item()
            
            # Check if all done
            if jnp.all(terminated | truncated):
                break
        
        episode_time = time.time() - episode_start
        times.append(episode_time)
    
    # Close environment
    env.close()
    
    print(f"\n" + "=" * 60)
    print(f"Performance:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Total auto-resets: {total_resets}")
    print(f"  Time: {sum(times):.2f}s")
    print(f"  Throughput: {total_steps / sum(times):,.0f} steps/sec")
    print(f"=" * 60)


if __name__ == "__main__":
    main()

