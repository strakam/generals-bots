"""
Simple smoke test: Verify JAX environment can be created and run.

This is a basic sanity check that the JAX environment works with the functional API.
"""

import jax.numpy as jnp
import numpy as np

from generals.envs.jax_env import VectorizedJaxEnv


def test_jax_env_generalsio_mode():
    """Test JAX environment with generalsio mode (default)."""
    env = VectorizedJaxEnv(num_envs=4, mode='generalsio')
    
    assert env.mode == 'generalsio'
    assert env.grid_size == (24, 24)  # Padded size
    assert env.num_envs == 4
    
    # Reset
    obs, info = env.reset(seed=42)
    
    # Check observation structure
    assert hasattr(obs, 'armies'), "Observation should have 'armies' field"
    assert obs.armies.shape == (4, 2, 24, 24), f"Expected (4, 2, 24, 24), got {obs.armies.shape}"
    
    # Check info structure
    assert hasattr(info, 'army'), "Info should have 'army' field"
    assert info.army.shape == (4, 2), f"Expected (4, 2), got {info.army.shape}"
    
    # Take a step
    actions = jnp.zeros((4, 2, 5), dtype=jnp.int32)  # Pass actions
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    # Check outputs
    assert obs.armies.shape == (4, 2, 24, 24), "Observation shape mismatch"
    assert rewards.shape == (4, 2), f"Expected rewards (4, 2), got {rewards.shape}"
    assert terminated.shape == (4,), f"Expected terminated (4,), got {terminated.shape}"
    assert truncated.shape == (4,), f"Expected truncated (4,), got {truncated.shape}"
    
    env.close()
    
    print("✓ JAX environment generalsio mode test passed!")


def test_jax_env_fixed_mode():
    """Test JAX environment with fixed grid size."""
    env = VectorizedJaxEnv(
        num_envs=4,
        mode='fixed',
        grid_dims=(12, 12),
        pad_to=12,
    )
    
    assert env.mode == 'fixed'
    assert env.grid_size == (12, 12)
    assert env.num_envs == 4
    
    # Reset and step
    obs, info = env.reset(seed=123)
    actions = jnp.zeros((4, 2, 5), dtype=jnp.int32)
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    assert obs.armies.shape == (4, 2, 12, 12)
    
    env.close()
    
    print("✓ JAX environment fixed mode test passed!")


def test_jax_env_basic_usage():
    """Test basic environment usage with multiple steps."""
    env = VectorizedJaxEnv(num_envs=2, mode='generalsio')
    
    obs, info = env.reset(seed=999)
    
    # Run a few steps
    for _ in range(10):
        actions = jnp.zeros((2, 2, 5), dtype=jnp.int32)
        obs, rewards, terminated, truncated, info = env.step(actions)
    
    env.close()
    
    print("✓ JAX environment basic usage test passed!")


def test_different_grids_per_env():
    """Test that each environment gets a different grid."""
    env = VectorizedJaxEnv(num_envs=3, mode='generalsio')
    obs, info = env.reset(seed=777)
    
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
    test_jax_env_generalsio_mode()
    test_jax_env_fixed_mode()
    test_jax_env_basic_usage()
    test_different_grids_per_env()
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
