"""
JAX-optimized action utilities using arrays and JIT-compatible functions.

Provides:
- Pure JAX array-based actions (no custom classes)
- JIT-compiled valid move mask computation
- Vectorizable action utilities
"""

import jax
import jax.numpy as jnp
from jax import lax


# Direction constants matching config.py
DIRECTIONS = jnp.array([
    [-1, 0],  # UP
    [1, 0],   # DOWN
    [0, -1],  # LEFT
    [0, 1],   # RIGHT
], dtype=jnp.int32)


def create_action(to_pass: bool = False, row: int = 0, col: int = 0, 
                  direction: int = 0, to_split: bool = False) -> jnp.ndarray:
    """
    Create an action array for JAX game.
    
    Args:
        to_pass: Whether to pass this turn
        row: Source row
        col: Source column
        direction: Direction index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        to_split: Whether to split the army
    
    Returns:
        JAX array [pass, row, col, direction, split] of shape (5,)
    """
    return jnp.array([
        int(to_pass), 
        row, 
        col, 
        direction, 
        int(to_split)
    ], dtype=jnp.int32)


@jax.jit
def compute_valid_move_mask(
    armies: jnp.ndarray,
    owned_cells: jnp.ndarray,
    mountains: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute valid move mask for a given observation.
    
    A valid move originates from a cell the agent owns, has at least 2 armies,
    and does not attempt to enter a mountain nor exit the grid.
    
    Args:
        armies: [H, W] army counts
        owned_cells: [H, W] boolean mask of owned cells
        mountains: [H, W] boolean mask of mountains
    
    Returns:
        [H, W, 4] boolean mask where mask[i, j, k] indicates if moving
        from (i, j) in direction k is valid
    """
    H, W = armies.shape
    
    # Can only move from owned cells with >1 army
    can_move_from = owned_cells & (armies > 1)
    
    # Mountains are not passable
    passable = ~mountains
    
    # Initialize mask
    valid_mask = jnp.zeros((H, W, 4), dtype=jnp.bool_)
    
    # Check each direction
    for dir_idx in range(4):
        di, dj = DIRECTIONS[dir_idx]
        
        # Compute destination indices for all cells
        dest_i = jnp.arange(H)[:, None] + di
        dest_j = jnp.arange(W)[None, :] + dj
        
        # Check bounds
        in_bounds = (
            (dest_i >= 0) & (dest_i < H) &
            (dest_j >= 0) & (dest_j < W)
        )
        
        # Safe indexing: clamp to valid range for passable check
        safe_dest_i = jnp.clip(dest_i, 0, H - 1)
        safe_dest_j = jnp.clip(dest_j, 0, W - 1)
        
        # Check if destination is passable
        dest_passable = passable[safe_dest_i, safe_dest_j]
        
        # Valid if: can move from source, destination in bounds, destination passable
        valid = can_move_from & in_bounds & dest_passable
        
        valid_mask = valid_mask.at[:, :, dir_idx].set(valid)
    
    return valid_mask


@jax.jit
def compute_valid_move_mask_obs(observation) -> jnp.ndarray:
    """
    Convenience wrapper that takes an ObservationJax NamedTuple.
    
    Args:
        observation: ObservationJax named tuple
    
    Returns:
        [H, W, 4] boolean mask of valid moves
    """
    return compute_valid_move_mask(
        observation.armies,
        observation.owned_cells,
        observation.mountains,
    )
