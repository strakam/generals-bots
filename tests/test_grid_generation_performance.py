"""Test grid generation performance and correctness."""
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals.core.grid import generate_grid, manhattan_distance_from


def test_grid_generation_no_crash():
    """Test that generating grids doesn't crash."""
    print("\n=== Testing Grid Generation Stability ===")

    num_grids = 100
    key = jax.random.PRNGKey(42)

    print(f"\nGenerating {num_grids} grids...")
    keys = jax.random.split(key, num_grids)

    start = time.time()
    for i, k in enumerate(keys):
        grid = generate_grid(k)

    elapsed = time.time() - start
    grids_per_sec = num_grids / elapsed

    print(f"✓ Generated {num_grids} grids in {elapsed:.2f}s")
    print(f"  Performance: {grids_per_sec:.1f} grids/second")


def test_100_percent_validity():
    """Test that all generated grids are valid (have both generals)."""
    print("\n=== Testing Validity Rate ===")

    num_grids = 100
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, num_grids)

    has_both_generals = 0

    for k in keys:
        grid = generate_grid(k)

        # Check for both generals
        has_g1 = jnp.any(grid == 1)
        has_g2 = jnp.any(grid == 2)
        if has_g1 and has_g2:
            has_both_generals += 1

    print(f"\nResults from {num_grids} grids:")
    print(f"  Grids with both generals: {has_both_generals}/{num_grids}")

    assert has_both_generals == num_grids, "Expected all grids to have both generals"


def test_generals_distance():
    """Test that generals are appropriately spaced."""
    print("\n=== Testing Generals Distance ===")

    num_grids = 100
    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, num_grids)

    min_generals_distance = 17
    distances = []

    for k in keys:
        grid = generate_grid(k)

        # Find general positions (in the unpadded portion)
        g1_pos = jnp.argwhere(grid == 1, size=1)[0]
        g2_pos = jnp.argwhere(grid == 2, size=1)[0]

        distance = abs(int(g1_pos[0]) - int(g2_pos[0])) + abs(int(g1_pos[1]) - int(g2_pos[1]))
        distances.append(distance)

    print(f"\nGeneral distance statistics from {num_grids} grids:")
    print(f"  Mean: {np.mean(distances):.1f}")
    print(f"  Min: {np.min(distances)}")
    print(f"  Max: {np.max(distances)}")

    assert np.min(distances) >= min_generals_distance, f"Min distance {np.min(distances)} < {min_generals_distance}"


def test_grid_properties():
    """Test general properties of generated grids."""
    print("\n=== Testing Grid Properties ===")

    num_grids = 100
    key = jax.random.PRNGKey(789)
    keys = jax.random.split(key, num_grids)

    mountain_counts = []
    city_counts = []

    for k in keys:
        grid = generate_grid(k)

        # Count mountains (value -2)
        num_mountains = int(jnp.sum(grid == -2))
        mountain_counts.append(num_mountains)

        # Count cities (values 40-50)
        num_cities = int(jnp.sum((grid >= 40) & (grid <= 50)))
        city_counts.append(num_cities)

    print(f"\nStatistics from {num_grids} grids:")
    print(f"  Mountains: mean={np.mean(mountain_counts):.1f}, range=[{np.min(mountain_counts)}, {np.max(mountain_counts)}]")
    print(f"  Cities: mean={np.mean(city_counts):.1f}, range=[{np.min(city_counts)}, {np.max(city_counts)}]")

    assert np.mean(city_counts) >= 5, f"Too few cities: {np.mean(city_counts)}"


def test_jit_performance():
    """Test performance with JIT compilation."""
    print("\n=== Testing JIT Performance ===")

    key = jax.random.PRNGKey(0)

    # Warmup JIT compilation
    _ = generate_grid(key)

    num_iterations = 100
    start = time.time()
    for _ in range(num_iterations):
        key, subkey = jax.random.split(key)
        grid = generate_grid(subkey)
        grid.block_until_ready()

    elapsed = time.time() - start
    grids_per_sec = num_iterations / elapsed

    print(f"\nGenerated {num_iterations} grids in {elapsed:.2f}s")
    print(f"Performance: {grids_per_sec:.1f} grids/second")

    assert elapsed < 10.0, f"Too slow: {elapsed}s for {num_iterations} grids"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
