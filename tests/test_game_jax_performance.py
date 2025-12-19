"""Performance comparison between NumPy and JAX implementations."""
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals.core import game_jax
from generals.core.game import Game
from generals.core.grid import GridFactory
from generals.core.action import Action


def create_numpy_game():
    """Create a NumPy-based game."""
    grid_factory = GridFactory(
        min_grid_dims=(10, 10),
        max_grid_dims=(10, 10),
        mountain_density=0.1,
        city_density=0.1,
        general_positions=[[0, 0], [9, 9]],
    )
    grid = grid_factory.generate()
    return Game(grid, ["agent_0", "agent_1"])


def create_jax_game():
    """Create a JAX-based game state."""
    grid_factory = GridFactory(
        min_grid_dims=(10, 10),
        max_grid_dims=(10, 10),
        mountain_density=0.1,
        city_density=0.1,
        general_positions=[[0, 0], [9, 9]],
    )
    grid = grid_factory.generate()
    
    # Convert grid to byte array
    grid_array = np.vectorize(ord)(grid.grid).astype(np.uint8)
    grid_jax = jnp.array(grid_array)
    
    return game_jax.create_initial_state(grid_jax)


def random_numpy_action(game):
    """Generate a random valid action for NumPy game."""
    # Simple random action: pass or random move
    if np.random.random() < 0.3:
        return Action(to_pass=True)
    
    height, width = game.grid_dims
    row = np.random.randint(0, height)
    col = np.random.randint(0, width)
    direction = np.random.randint(0, 4)
    
    return Action(to_pass=False, row=row, col=col, direction=direction)


def random_jax_actions():
    """Generate random actions for both players."""
    return jnp.array([
        [
            np.random.randint(0, 2),  # pass
            np.random.randint(0, 10),  # row
            np.random.randint(0, 10),  # col
            np.random.randint(0, 4),   # direction
            0,                          # split
        ],
        [
            np.random.randint(0, 2),
            np.random.randint(0, 10),
            np.random.randint(0, 10),
            np.random.randint(0, 4),
            0,
        ],
    ], dtype=jnp.int32)


@pytest.mark.skip(reason="Performance test - run manually")
def test_numpy_performance():
    """Benchmark NumPy implementation."""
    num_steps = 1000
    
    game = create_numpy_game()
    
    start_time = time.time()
    
    for _ in range(num_steps):
        actions = {
            "agent_0": random_numpy_action(game),
            "agent_1": random_numpy_action(game),
        }
        game.step(actions)
        
        if game.is_done():
            game = create_numpy_game()
    
    elapsed = time.time() - start_time
    
    print(f"\nNumPy Performance:")
    print(f"  Steps: {num_steps}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Steps/sec: {num_steps / elapsed:.2f}")


@pytest.mark.skip(reason="Performance test - run manually")
def test_jax_performance():
    """Benchmark JAX implementation (single environment)."""
    num_steps = 1000
    
    state = create_jax_game()
    
    # JIT compile
    jitted_step = jax.jit(game_jax.step)
    
    # Warmup
    for _ in range(10):
        actions = random_jax_actions()
        state, _ = jitted_step(state, actions)
    
    state = create_jax_game()
    start_time = time.time()
    
    for _ in range(num_steps):
        actions = random_jax_actions()
        state, info = jitted_step(state, actions)
        
        if info['is_done']:
            state = create_jax_game()
    
    elapsed = time.time() - start_time
    
    print(f"\nJAX Performance (single env):")
    print(f"  Steps: {num_steps}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Steps/sec: {num_steps / elapsed:.2f}")


@pytest.mark.skip(reason="Performance test - run manually")
def test_jax_batched_performance():
    """Benchmark JAX implementation (batched environments)."""
    num_envs = 256
    steps_per_env = 1000
    total_steps = num_envs * steps_per_env
    
    # Create batched states
    state = create_jax_game()
    batched_state = jax.tree.map(
        lambda x: jnp.stack([x] * num_envs),
        state
    )
    
    # JIT compile batched step
    jitted_batch_step = jax.jit(game_jax.batch_step)
    
    # Warmup
    for _ in range(10):
        actions = jnp.stack([random_jax_actions() for _ in range(num_envs)])
        batched_state, _ = jitted_batch_step(batched_state, actions)
    
    # Reset
    batched_state = jax.tree.map(
        lambda x: jnp.stack([x] * num_envs),
        create_jax_game()
    )
    
    start_time = time.time()
    
    for _ in range(steps_per_env):
        actions = jnp.stack([random_jax_actions() for _ in range(num_envs)])
        batched_state, infos = jitted_batch_step(batched_state, actions)
    
    elapsed = time.time() - start_time
    
    print(f"\nJAX Batched Performance:")
    print(f"  Num environments: {num_envs}")
    print(f"  Steps per env: {steps_per_env}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Steps/sec: {total_steps / elapsed:,.2f}")
    print(f"  Time per 1M steps: {elapsed / total_steps * 1_000_000:.2f}s")


def test_jax_quick_benchmark():
    """Quick benchmark that runs in CI."""
    num_steps = 100
    
    state = create_jax_game()
    jitted_step = jax.jit(game_jax.step)
    
    # Warmup
    for _ in range(5):
        actions = random_jax_actions()
        state, _ = jitted_step(state, actions)
    
    state = create_jax_game()
    start_time = time.time()
    
    for _ in range(num_steps):
        actions = random_jax_actions()
        state, info = jitted_step(state, actions)
        
        if info['is_done']:
            state = create_jax_game()
    
    elapsed = time.time() - start_time
    
    # Just verify it runs without error and is reasonably fast
    assert elapsed < 5.0  # Should complete 100 steps in < 5 seconds
    print(f"\nQuick benchmark: {num_steps} steps in {elapsed:.3f}s ({num_steps/elapsed:.1f} steps/sec)")


if __name__ == "__main__":
    # Run performance tests manually
    print("Running performance benchmarks...")
    
    print("\n" + "="*60)
    test_numpy_performance()
    
    print("\n" + "="*60)
    test_jax_performance()
    
    print("\n" + "="*60)
    test_jax_batched_performance()
