"""Tests for action_jax module."""

import numpy as np
import jax.numpy as jnp
import pytest

from generals.core import game_jax
from generals.core.action_jax import (
    create_action,
    compute_valid_move_mask,
    compute_valid_move_mask_obs,
    DIRECTIONS,
)
from generals.core.grid import GridFactory


def test_create_action():
    """Test action creation."""
    # Pass action
    action = create_action(to_pass=True)
    assert action.shape == (5,)
    assert action[0] == 1
    
    # Move action
    action = create_action(to_pass=False, row=3, col=5, direction=2, to_split=False)
    assert action[0] == 0
    assert action[1] == 3
    assert action[2] == 5
    assert action[3] == 2
    assert action[4] == 0
    
    # Split move action
    action = create_action(to_pass=False, row=1, col=2, direction=3, to_split=True)
    assert action[4] == 1


def test_directions():
    """Test that DIRECTIONS match expected values."""
    assert DIRECTIONS.shape == (4, 2)
    assert jnp.array_equal(DIRECTIONS[0], jnp.array([-1, 0]))  # UP
    assert jnp.array_equal(DIRECTIONS[1], jnp.array([1, 0]))   # DOWN
    assert jnp.array_equal(DIRECTIONS[2], jnp.array([0, -1]))  # LEFT
    assert jnp.array_equal(DIRECTIONS[3], jnp.array([0, 1]))   # RIGHT


def test_compute_valid_move_mask_empty():
    """Test mask computation with no owned cells."""
    H, W = 10, 10
    armies = jnp.zeros((H, W), dtype=jnp.int32)
    owned_cells = jnp.zeros((H, W), dtype=jnp.bool_)
    mountains = jnp.zeros((H, W), dtype=jnp.bool_)
    
    mask = compute_valid_move_mask(armies, owned_cells, mountains)
    
    assert mask.shape == (H, W, 4)
    assert jnp.sum(mask) == 0  # No valid moves


def test_compute_valid_move_mask_single_cell():
    """Test mask with a single owned cell."""
    H, W = 5, 5
    armies = jnp.zeros((H, W), dtype=jnp.int32)
    armies = armies.at[2, 2].set(5)  # 5 armies in center
    
    owned_cells = jnp.zeros((H, W), dtype=jnp.bool_)
    owned_cells = owned_cells.at[2, 2].set(True)
    
    mountains = jnp.zeros((H, W), dtype=jnp.bool_)
    
    mask = compute_valid_move_mask(armies, owned_cells, mountains)
    
    # Should have 4 valid moves from center cell
    assert mask[2, 2, :].sum() == 4  # All 4 directions valid
    assert mask.sum() == 4


def test_compute_valid_move_mask_with_mountains():
    """Test that mountains block moves."""
    H, W = 5, 5
    armies = jnp.zeros((H, W), dtype=jnp.int32)
    armies = armies.at[2, 2].set(5)
    
    owned_cells = jnp.zeros((H, W), dtype=jnp.bool_)
    owned_cells = owned_cells.at[2, 2].set(True)
    
    mountains = jnp.zeros((H, W), dtype=jnp.bool_)
    mountains = mountains.at[1, 2].set(True)  # Mountain above
    mountains = mountains.at[2, 3].set(True)  # Mountain to right
    
    mask = compute_valid_move_mask(armies, owned_cells, mountains)
    
    # Only 2 directions should be valid (down and left)
    assert mask[2, 2, :].sum() == 2
    assert mask[2, 2, 0] == False  # UP blocked by mountain
    assert mask[2, 2, 1] == True   # DOWN valid
    assert mask[2, 2, 2] == True   # LEFT valid
    assert mask[2, 2, 3] == False  # RIGHT blocked by mountain


def test_compute_valid_move_mask_boundary():
    """Test that grid boundaries are respected."""
    H, W = 5, 5
    armies = jnp.zeros((H, W), dtype=jnp.int32)
    armies = armies.at[0, 0].set(5)  # Top-left corner
    
    owned_cells = jnp.zeros((H, W), dtype=jnp.bool_)
    owned_cells = owned_cells.at[0, 0].set(True)
    
    mountains = jnp.zeros((H, W), dtype=jnp.bool_)
    
    mask = compute_valid_move_mask(armies, owned_cells, mountains)
    
    # Only 2 directions valid from corner (down and right)
    assert mask[0, 0, :].sum() == 2
    assert mask[0, 0, 0] == False  # UP out of bounds
    assert mask[0, 0, 1] == True   # DOWN valid
    assert mask[0, 0, 2] == False  # LEFT out of bounds
    assert mask[0, 0, 3] == True   # RIGHT valid


def test_compute_valid_move_mask_insufficient_armies():
    """Test that cells with <=1 army cannot move."""
    H, W = 5, 5
    armies = jnp.zeros((H, W), dtype=jnp.int32)
    armies = armies.at[2, 2].set(1)  # Only 1 army
    
    owned_cells = jnp.zeros((H, W), dtype=jnp.bool_)
    owned_cells = owned_cells.at[2, 2].set(True)
    
    mountains = jnp.zeros((H, W), dtype=jnp.bool_)
    
    mask = compute_valid_move_mask(armies, owned_cells, mountains)
    
    # No valid moves with only 1 army
    assert mask.sum() == 0


def test_compute_valid_move_mask_obs_wrapper():
    """Test the observation wrapper function."""
    grid_factory = GridFactory(
        min_grid_dims=(10, 10),
        max_grid_dims=(10, 10),
        general_positions=[[1, 1], [8, 8]],
    )
    
    grid = grid_factory.generate()
    grid_array = np.vectorize(ord)(grid.grid).astype(np.uint8)
    grid_jax = jnp.array(grid_array)
    
    state = game_jax.create_initial_state(grid_jax)
    
    # Run a few steps to get armies > 1
    for _ in range(10):
        actions = jnp.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
        state, _ = game_jax.step(state, actions)
    
    obs = game_jax.get_observation(state, 0)
    mask = compute_valid_move_mask_obs(obs)
    
    assert mask.shape == (*obs.armies.shape, 4)
    # Should have some valid moves by now
    assert mask.sum() > 0


def test_jit_compilation():
    """Test that the mask computation is JIT-compilable."""
    import jax
    
    @jax.jit
    def jitted_mask(armies, owned, mountains):
        return compute_valid_move_mask(armies, owned, mountains)
    
    H, W = 10, 10
    armies = jnp.zeros((H, W), dtype=jnp.int32)
    owned_cells = jnp.zeros((H, W), dtype=jnp.bool_)
    mountains = jnp.zeros((H, W), dtype=jnp.bool_)
    
    # Should compile and run without error
    mask = jitted_mask(armies, owned_cells, mountains)
    assert mask.shape == (H, W, 4)
