"""Simple performance benchmark for vectorized Generals environment."""
import time
import jax
import jax.numpy as jnp
import jax.random as jrandom
from generals import GeneralsEnv


def benchmark(num_envs: int = 256, num_steps: int = 10_000):
    """
    Benchmark the vectorized environment.

    Args:
        num_envs: Number of parallel environments
        num_steps: Total number of steps to run
    """
    print(f"Creating {num_envs} parallel environments...")
    env = GeneralsEnv(grid_dims=(4, 4), truncation=500, pool_size=1_000)

    # Generate pool once, then create per-env initial states
    key = jrandom.PRNGKey(42)
    key, pool_key = jrandom.split(key)
    env.reset(pool_key)

    init_keys = jrandom.split(key, num_envs)
    states = jax.vmap(env.init_state)(init_keys)

    # Simple pass action for both players: [1, 0, 0, 0, 0]
    pass_action = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
    actions = jnp.tile(pass_action, (num_envs, 2, 1))  # Shape: [num_envs, 2, 5]

    print("Warming up JIT compilation...")
    timesteps, states = jax.vmap(env.step)(states, actions)
    jax.block_until_ready(states)
    print("JIT compilation complete!\n")

    print(f"Running benchmark: {num_envs} envs x {num_steps:,} steps")
    print("=" * 70)

    start_time = time.time()
    print_interval = 1000

    for step in range(num_steps):
        timesteps, states = jax.vmap(env.step)(states, actions)
        jax.block_until_ready(states)

        if (step + 1) % print_interval == 0:
            elapsed = time.time() - start_time
            total_steps = (step + 1) * num_envs
            fps = total_steps / elapsed
            print(f"Step {step+1:7,}/{num_steps:,}: {fps:10,.0f} steps/s  "
                  f"({elapsed:6.1f}s elapsed)")

    elapsed = time.time() - start_time
    total_steps = num_steps * num_envs
    fps = total_steps / elapsed

    print("=" * 70)
    print(f"Total time:      {elapsed:.2f}s")
    print(f"Total steps:     {total_steps:,}")
    print(f"Average FPS:     {fps:,.0f} steps/second")
    print(f"Per environment: {fps / num_envs:.1f} steps/second")


if __name__ == "__main__":
    import sys
    num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
    benchmark(num_envs=num_envs, num_steps=num_steps)
