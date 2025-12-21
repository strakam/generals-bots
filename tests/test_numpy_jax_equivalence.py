"""
Simple smoke test: Verify JAX environment can be created and run.

This is a basic sanity check that the JAX environment works with the new interface.
For comprehensive correctness testing, see numpy_vs_jax_correctness.py
"""

import jax.numpy as jnp
import numpy as np

from generals.core.grid_jax import GridFactoryJax
from generals.envs.jax_env import VectorizedJaxEnv


def test_jax_env_creation_and_basic_usage():
    """Test that JAX environment can be created and used."""
    
    # Create JAX grid factory
    grid_factory = GridFactoryJax(
        grid_dims=(10, 10),
        mountain_density=0.1,
        num_castles_range=(1, 3),
        min_generals_distance=5,
        castle_val_range=(40, 51),
        pad_to=10,
    )
    
    # Create JAX environment
    env = VectorizedJaxEnv(
        num_envs=4,
        grid_factory=grid_factory,
        render_mode=None,
    )
    
    assert env.grid_size == (10, 10), f"Expected grid_size (10, 10), got {env.grid_size}"
    assert env.num_envs == 4, f"Expected 4 envs, got {env.num_envs}"
    
    # Reset
    obs, info = env.reset(seed=42)
    
    # Check observation structure
    assert hasattr(obs, 'armies'), "Observation should have 'armies' field"
    assert obs.armies.shape == (4, 2, 10, 10), f"Expected (4, 2, 10, 10), got {obs.armies.shape}"
    
    # Check info structure
    assert hasattr(info, 'army'), "Info should have 'army' field"
    assert info.army.shape == (4, 2), f"Expected (4, 2), got {info.army.shape}"
    
    # Take a step
    actions = jnp.zeros((4, 2, 5), dtype=jnp.int32)  # Pass actions
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    # Check outputs
    assert obs.armies.shape == (4, 2, 10, 10), "Observation shape mismatch"
    assert rewards.shape == (4, 2), f"Expected rewards (4, 2), got {rewards.shape}"
    assert terminated.shape == (4,), f"Expected terminated (4,), got {terminated.shape}"
    assert truncated.shape == (4,), f"Expected truncated (4,), got {truncated.shape}"
    
    # Run a few more steps
    for _ in range(10):
        actions = jnp.zeros((4, 2, 5), dtype=jnp.int32)
        obs, rewards, terminated, truncated, info = env.step(actions)
    
    env.close()
    
    print("✓ JAX environment creation and basic usage test passed!")


def test_jax_env_with_defaults():
    """Test JAX environment with default grid factory."""
    
    # Create with defaults (generals.io settings)
    env = VectorizedJaxEnv(num_envs=2)
    
    assert env.grid_size == (24, 24), f"Expected default grid_size (24, 24), got {env.grid_size}"
    assert env.num_envs == 2
    
    # Reset and step
    obs, info = env.reset(seed=123)
    actions = jnp.zeros((2, 2, 5), dtype=jnp.int32)
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    assert obs.armies.shape == (2, 2, 24, 24)
    
    env.close()
    
    print("✓ JAX environment with defaults test passed!")


def test_different_grids_per_env():
    """Test that each environment gets a different grid."""
    
    env = VectorizedJaxEnv(num_envs=3)
    obs, info = env.reset(seed=999)
    
    # Check that grids are different
    grid1 = obs.armies[0]
    grid2 = obs.armies[1]
    grid3 = obs.armies[2]
    
    # They should not all be identical
    assert not (jnp.array_equal(grid1, grid2) and jnp.array_equal(grid2, grid3)), \
        "All grids are identical - should have different grids per environment"
    
    env.close()
    
    print("✓ Different grids per environment test passed!")


if __name__ == "__main__":
    test_jax_env_creation_and_basic_usage()
    test_jax_env_with_defaults()
    test_different_grids_per_env()
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
