"""Action utilities for JAX game."""
import jax
import jax.numpy as jnp
import jax.random as jrandom

# Direction offsets: UP, DOWN, LEFT, RIGHT
DIRECTIONS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


def create_action(
    to_pass: bool = False, row: int = 0, col: int = 0, direction: int = 0, to_split: bool = False
) -> jnp.ndarray:
    """Create action array [pass, row, col, direction, split]."""
    return jnp.array([int(to_pass), row, col, direction, int(to_split)], dtype=jnp.int32)


@jax.jit
def compute_valid_move_mask(
    armies: jnp.ndarray,
    owned_cells: jnp.ndarray,
    mountains: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute valid move mask (fully vectorized).

    Returns (H, W, 4) mask where mask[i, j, d] is True if moving from (i, j)
    in direction d is valid.
    """
    H, W = armies.shape

    can_move_from = owned_cells & (armies > 1)
    passable = ~mountains

    # Create coordinate grids: [H, W]
    i_idx = jnp.arange(H)[:, None]
    j_idx = jnp.arange(W)[None, :]
    
    # Compute destination coords for all 4 directions at once: [H, W, 4]
    # DIRECTIONS shape is [4, 2], we want dest_i[h, w, d] = h + DIRECTIONS[d, 0]
    dest_i = i_idx[:, :, None] + DIRECTIONS[None, None, :, 0]  # [H, W, 4]
    dest_j = j_idx[:, :, None] + DIRECTIONS[None, None, :, 1]  # [H, W, 4]
    
    # Check bounds for all directions: [H, W, 4]
    in_bounds = (dest_i >= 0) & (dest_i < H) & (dest_j >= 0) & (dest_j < W)
    
    # Clip to valid indices for safe lookup
    safe_dest_i = jnp.clip(dest_i, 0, H - 1)
    safe_dest_j = jnp.clip(dest_j, 0, W - 1)
    
    # Check if destinations are passable: [H, W, 4]
    dest_passable = passable[safe_dest_i, safe_dest_j]
    
    # Combine all conditions: [H, W, 4]
    valid_mask = can_move_from[:, :, None] & in_bounds & dest_passable

    return valid_mask


@jax.jit
def compute_valid_move_mask_obs(observation) -> jnp.ndarray:
    """Compute valid move mask from Observation."""
    return compute_valid_move_mask(observation.armies, observation.owned_cells, observation.mountains)


def sample_valid_action(key: jnp.ndarray, observation, allow_pass: bool = True) -> jnp.ndarray:
    """Sample a random valid action from observation."""
    valid_mask = compute_valid_move_mask_obs(observation)
    H, W = observation.armies.shape

    valid_positions = jnp.argwhere(valid_mask, size=H * W * 4, fill_value=-1)
    num_valid = jnp.sum(jnp.all(valid_positions >= 0, axis=-1))

    k1, k2, k3 = jrandom.split(key, 3)

    should_pass = allow_pass & (jrandom.uniform(k1) < 0.1)
    should_pass = should_pass | (num_valid == 0)

    move_idx = jrandom.randint(k2, (), 0, jnp.maximum(num_valid, 1))
    move_idx = jnp.minimum(move_idx, num_valid - 1)
    selected_move = valid_positions[move_idx]

    split = jrandom.randint(k3, (), 0, 2)

    return jnp.array(
        [should_pass.astype(jnp.int32), selected_move[0], selected_move[1], selected_move[2], split],
        dtype=jnp.int32,
    )
