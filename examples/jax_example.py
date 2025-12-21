"""
Example demonstrating JAX-based game implementation for high-performance training.

This example shows how to use the JAX game implementation for efficient
parallel environment execution, suitable for RL training.
Fully optimized with JIT compilation and JAX random for MAXIMUM SPEED.
"""
import time
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generals.core import game_jax
from generals.core.grid import GridFactory


def create_jax_state_from_factory():
    """Create a JAX game state using the existing grid factory."""
    grid_factory = GridFactory(
        min_grid_dims=(10, 10),
        max_grid_dims=(10, 10),
        mountain_density=0.1,
        city_density=0.1,
        general_positions=[[1, 1], [8, 8]],
    )
    
    # Generate grid using NumPy factory
    grid = grid_factory.generate()
    
    # Convert to JAX-compatible format
    grid_array = np.vectorize(ord)(grid.grid).astype(np.uint8)
    grid_jax = jnp.array(grid_array)
    
    return game_jax.create_initial_state(grid_jax)


def random_actions_jax(key: jnp.ndarray, batch_size: int, grid_dims: Tuple[int, int] = (10, 10)):
    """Generate random actions for both players (uses JAX random)."""
    H, W = grid_dims
    
    # Split key into enough subkeys
    subkeys = jrandom.split(key, 5)
    
    # Generate all random values at once
    pass_vals = jrandom.randint(subkeys[0], (batch_size, 2), 0, 2)
    rows = jrandom.randint(subkeys[1], (batch_size, 2), 0, H)
    cols = jrandom.randint(subkeys[2], (batch_size, 2), 0, W)
    directions = jrandom.randint(subkeys[3], (batch_size, 2), 0, 4)
    splits = jnp.zeros((batch_size, 2), dtype=jnp.int32)  # No splits for simplicity
    
    # Stack into action format [batch, 2_players, 5]
    actions = jnp.stack([pass_vals, rows, cols, directions, splits], axis=-1)
    
    return actions


def example_single_environment():
    """Run a single environment with JAX."""
    print("=" * 60)
    print("Example 1: Single Environment")
    print("=" * 60)
    
    state = create_jax_state_from_factory()
    jitted_step = jax.jit(game_jax.step)
    
    rng_key = jrandom.PRNGKey(0)
    
    # Run for 100 steps
    for i in range(100):
        rng_key, subkey = jrandom.split(rng_key)
        actions = random_actions_jax(subkey, 1)[0]  # Remove batch dimension
        state, info = jitted_step(state, actions)
        
        if info.is_done:
            print(f"\nGame finished at step {i}")
            print(f"Winner: Player {info.winner}")
            break
        
        if i % 20 == 0:
            print(f"Step {i}: P0 army={info.army[0]}, P1 army={info.army[1]}")
    
    # Get final observations
    obs_p0 = game_jax.get_observation(state, 0)
    obs_p1 = game_jax.get_observation(state, 1)
    
    print(f"\nFinal state:")
    print(f"  Player 0: {obs_p0.owned_land_count} land, {obs_p0.owned_army_count} army")
    print(f"  Player 1: {obs_p1.owned_land_count} land, {obs_p1.owned_army_count} army")


def example_batched_environments():
    """Run multiple environments in parallel with JAX."""
    print("\n" + "=" * 60)
    print("Example 2: Batched Environments (64 parallel)")
    print("=" * 60)
    
    num_envs = 64
    rng_key = jrandom.PRNGKey(42)
    
    # Create batched state
    state = create_jax_state_from_factory()
    batched_state = jax.tree.map(
        lambda x: jnp.stack([x] * num_envs),
        state
    )
    
    # JIT compile batched step
    jitted_step = jax.jit(game_jax.batch_step)
    
    # Warmup
    print("Warming up JIT compilation...")
    for _ in range(5):
        rng_key, subkey = jrandom.split(rng_key)
        actions = random_actions_jax(subkey, num_envs)
        batched_state, _ = jitted_step(batched_state, actions)
    
    # Reset and benchmark
    batched_state = jax.tree.map(
        lambda x: jnp.stack([x] * num_envs),
        create_jax_state_from_factory()
    )
    
    print(f"Running {num_envs} environments for 1000 steps...")
    start_time = time.time()
    
    for i in range(1000):
        rng_key, subkey = jrandom.split(rng_key)
        actions = random_actions_jax(subkey, num_envs)
        batched_state, infos = jitted_step(batched_state, actions)
        
        if i % 200 == 0:
            num_done = jnp.sum(infos.is_done).item()
            avg_army_p0 = jnp.mean(infos.army[:, 0]).item()
            avg_army_p1 = jnp.mean(infos.army[:, 1]).item()
            print(f"Step {i}: {num_done} done, avg armies: P0={avg_army_p0:.1f}, P1={avg_army_p1:.1f}")
    
    elapsed = time.time() - start_time
    total_steps = num_envs * 1000
    
    print(f"\nPerformance:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {total_steps / elapsed:,.0f} steps/sec")


def example_training_loop():
    """Demonstrate a simplified RL training loop structure."""
    print("\n" + "=" * 60)
    print("Example 3: RL Training Loop Structure")
    print("=" * 60)
    
    num_envs = 32
    rollout_steps = 128
    rng_key = jrandom.PRNGKey(123)
    
    # Initialize environments
    state = create_jax_state_from_factory()
    batched_state = jax.tree.map(
        lambda x: jnp.stack([x] * num_envs),
        state
    )
    
    jitted_step = jax.jit(game_jax.batch_step)
    
    print(f"Training with {num_envs} parallel environments")
    print(f"Collecting {rollout_steps} steps per rollout")
    
    # Simulate 5 rollouts
    for epoch in range(5):
        observations = []
        actions_taken = []
        infos_collected = []
        
        start = time.time()
        
        # Collect rollout
        for step in range(rollout_steps):
            # In real training, you'd call your policy here
            rng_key, subkey = jrandom.split(rng_key)
            actions = random_actions_jax(subkey, num_envs)
            
            # Step environments
            new_state, infos = jitted_step(batched_state, actions)
            
            # Store data (in real training, you'd compute rewards, etc.)
            infos_collected.append(infos)
            actions_taken.append(actions)
            
            batched_state = new_state
        
        elapsed = time.time() - start
        total_steps = num_envs * rollout_steps
        
        print(f"Epoch {epoch+1}: Collected {total_steps} steps in {elapsed:.2f}s "
              f"({total_steps/elapsed:.0f} steps/sec)")


if __name__ == "__main__":
    print("\nJAX Generals.io Implementation Examples\n")
    
    example_single_environment()
    example_batched_environments()
    example_training_loop()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
