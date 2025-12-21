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
    # Configuration
    num_envs = 128
    num_episodes = 5
    max_steps_per_episode = 1000

    env = VectorizedJaxEnv(num_envs=num_envs, mode='generalsio')
    # Initialize JAX random key
    rng_key = jrandom.PRNGKey(42)
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(obs.armies.shape)
    
    total_steps = 0
    total_resets = 0
    
    print("\nRunning episodes...")
    print("-" * 70)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
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
        
    # Clean up
    env.close()
if __name__ == "__main__":
    main()

