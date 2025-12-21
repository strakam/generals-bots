"""
Example demonstrating the vectorized JAX environment.

Shows how to use VectorizedJaxEnv for high-performance RL training:
- Pure JAX grid generation (10-50x faster than NumPy)
- Two modes: 'generalsio' (random size) and 'fixed' (exact size)
- Different grids per environment (better diversity)
- Automatic auto-reset on episode termination
- Vectorized operations for maximum throughput
- GPU-compatible (if JAX GPU installed)
"""
import time

import jax.numpy as jnp
import jax.random as jrandom

from generals.envs import VectorizedJaxEnv


def random_actions_jax(key: jnp.ndarray, num_envs: int, grid_size: tuple[int, int]) -> jnp.ndarray:
    """Generate random actions for all environments using JAX random."""
    H, W = grid_size
    
    # Split key for different random operations
    subkeys = jrandom.split(key, 5)
    
    # Generate random values for all actions at once
    pass_vals = jrandom.uniform(subkeys[0], (num_envs, 2)) < 0.3  # 30% chance to pass
    rows = jrandom.randint(subkeys[1], (num_envs, 2), 0, H)
    cols = jrandom.randint(subkeys[2], (num_envs, 2), 0, W)
    directions = jrandom.randint(subkeys[3], (num_envs, 2), 0, 4)
    splits = jrandom.randint(subkeys[4], (num_envs, 2), 0, 2)
    
    # Stack into action arrays [num_envs, 2, 5]
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
    num_episodes = 5
    max_steps_per_episode = 2000
    
    print("\n" + "=" * 70)
    print("JAX Vectorized Environment Example")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Parallel environments: {num_envs}")
    print(f"  Episodes to run: {num_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    
    # Create environment with generals.io mode (default)
    # This matches the online game:
    # - Grid size: 18-23x18-23 (random, padded to 24x24)
    # - Mountains: 20% density + 2% variation
    # - Castles: 9-11 random count
    # - Different grid per environment!
    print("\nCreating environment in 'generalsio' mode...")
    env = VectorizedJaxEnv(num_envs=num_envs, mode='generalsio')
    
    print(f"  Mode: {env.mode}")
    print(f"  Grid size (padded): {env.grid_size}")
    
    # Initialize JAX random key
    rng_key = jrandom.PRNGKey(42)
    
    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset(seed=42)
    
    print(f"  Observation type: {type(obs).__name__}")
    print(f"  Observation fields: {obs._fields}")
    print(f"  Armies shape: {obs.armies.shape}")  # [num_envs, 2, H, W]
    print(f"  Info type: {type(info).__name__}")
    
    # Warmup JIT compilation
    print("\nWarming up JIT compilation...")
    warmup_start = time.time()
    for _ in range(5):
        rng_key, subkey = jrandom.split(rng_key)
        actions = random_actions_jax(subkey, num_envs, env.grid_size)
        obs, rewards, terminated, truncated, info = env.step(actions)
    warmup_time = time.time() - warmup_start
    print(f"  Warmup completed in {warmup_time:.2f}s")
    
    # Run training loop
    total_steps = 0
    total_resets = 0
    episode_times = []
    
    print("\nRunning episodes...")
    print("-" * 70)
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_start = time.time()
        
        for step in range(max_steps_per_episode):
            # Generate random actions using JAX random
            rng_key, subkey = jrandom.split(rng_key)
            actions = random_actions_jax(subkey, num_envs, env.grid_size)
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            total_steps += num_envs
            total_resets += jnp.sum(terminated).item()
            
            # Check if all environments are done
            if jnp.all(terminated | truncated):
                break
        
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        steps_in_episode = (step + 1) * num_envs
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"{steps_in_episode:,} steps in {episode_time:.2f}s "
              f"({steps_in_episode / episode_time:,.0f} steps/sec)")
    
    # Summary
    total_time = sum(episode_times)
    avg_throughput = total_steps / total_time
    
    print("-" * 70)
    print("\nPerformance Summary:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Total auto-resets: {total_resets}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average throughput: {avg_throughput:,.0f} steps/sec")
    print(f"  Average episode time: {total_time / num_episodes:.2f}s")
    
    # Clean up
    env.close()
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Try 'fixed' mode: VectorizedJaxEnv(num_envs=128, mode='fixed', grid_dims=(15, 15))")
    print("  - Integrate with your RL algorithm (PPO, DQN, etc.)")
    print("  - Enable GPU acceleration (install jax[cuda])")
    print("  - Scale to 1000s of parallel environments!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

