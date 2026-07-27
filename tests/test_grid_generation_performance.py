"""Test grid generation performance and correctness."""
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals.core.grid import bfs_distance_field, generate_grid, manhattan_distance_from


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
    """Generals are spaced by WALKING distance, which is what the rule promises.

    The straight-line gap is deliberately NOT bounded: a pair separated by a
    ridge is far apart to play even when it looks close on the map, and the
    generator now admits exactly those boards. Manhattan is printed alongside so
    the gap between the two measures stays visible.
    """
    print("\n=== Testing Generals Distance ===")

    num_grids = 100
    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, num_grids)

    min_generals_distance = 17
    walking, straight = [], []

    for k in keys:
        grid = generate_grid(k)

        # Find general positions (in the unpadded portion)
        g1_pos = jnp.argwhere(grid == 1, size=1)[0]
        g2_pos = jnp.argwhere(grid == 2, size=1)[0]
        a = (int(g1_pos[0]), int(g1_pos[1]))
        b = (int(g2_pos[0]), int(g2_pos[1]))

        # only mountains block — the same rule game.create_initial_state uses
        walking.append(int(bfs_distance_field(grid != -2, a)[b]))
        straight.append(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    print(f"\nGeneral distance statistics from {num_grids} grids:")
    print(f"  Walking  — mean {np.mean(walking):.1f}, min {np.min(walking)}, max {np.max(walking)}")
    print(f"  Straight — mean {np.mean(straight):.1f}, min {np.min(straight)}, max {np.max(straight)}")

    assert np.min(walking) >= min_generals_distance, \
        f"Min walking distance {np.min(walking)} < {min_generals_distance}"


def test_grid_properties():
    """Test general properties of generated grids."""
    print("\n=== Testing Grid Properties ===")

    num_grids = 100
    key = jax.random.PRNGKey(789)
    keys = jax.random.split(key, num_grids)

    mountain_counts = []
    castle_counts = []

    for k in keys:
        grid = generate_grid(k)

        # Count mountains (value -2)
        num_mountains = int(jnp.sum(grid == -2))
        mountain_counts.append(num_mountains)

        # Count castles (values 40-50)
        num_castles = int(jnp.sum(grid > 2))
        castle_counts.append(num_castles)

    print(f"\nStatistics from {num_grids} grids:")
    print(f"  Mountains: mean={np.mean(mountain_counts):.1f}, range=[{np.min(mountain_counts)}, {np.max(mountain_counts)}]")
    print(f"  Castles: mean={np.mean(castle_counts):.1f}, range=[{np.min(castle_counts)}, {np.max(castle_counts)}]")

    assert np.mean(castle_counts) >= 5, f"Too few castles: {np.mean(castle_counts)}"


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
