from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=['grid_dims', 'pad_to', 'mountain_density', 'num_cities_range',
                                    'min_generals_distance', 'max_generals_distance', 'castle_val_range'])
def generate_grid(
    key: jax.random.PRNGKey,
    grid_dims: tuple[int, int] = (23, 23),
    pad_to: int | None = None,
    mountain_density: float = 0.2,
    num_cities_range: tuple[int, int] = (9, 11),
    min_generals_distance: int = 17,
    max_generals_distance: int | None = None,
    castle_val_range: tuple[int, int] = (40, 51),
) -> jnp.ndarray:
    """
    Generate grid using JAX-optimal algorithm with guaranteed validity.
    
    Unified generator that supports both square and non-square grids with
    dynamic padding and configurable general distance constraints.
    
    Algorithm:
    1. Place generals FIRST on empty grid (guaranteed valid distance)
    2. Mark protected zones around generals for castle placement
    3. Place castles in protected zones
    4. Place mountains on remaining cells
    5. Place remaining cities
    6. Check connectivity, carve L-path if needed
    7. Apply dynamic padding
    
    Args:
        key: JAX random key
        grid_dims: Grid dimensions (height, width) - supports non-square grids
        pad_to: Pad grid to this size for batching (None = max(h, w) + 1)
        mountain_density: Fraction of tiles that are mountains (0.18-0.22)
        num_cities_range: (min, max) number of cities to place
        min_generals_distance: Minimum BFS (shortest path) distance between generals
        max_generals_distance: Maximum BFS (shortest path) distance between generals (None = no limit)
        castle_val_range: (min, max) army value for cities
        
    Returns:
        Grid is always valid (validity=True always)
    """
    keys = jax.random.split(key, 11)
    
    h, w = grid_dims
    num_tiles = h * w
    
    # Random number of cities in range
    num_cities = jax.random.randint(keys[0], (), num_cities_range[0], num_cities_range[1] + 1)
    
    # Number of mountains: base density + small variation
    base_mountains = int(mountain_density * num_tiles)
    mountain_variation = jax.random.randint(keys[1], (), -10, 11)
    num_mountains = base_mountains + mountain_variation
    
    # =================================================================
    # Step 1: Place generals FIRST on empty grid
    # =================================================================
    grid = jnp.full(grid_dims, 0, dtype=jnp.int32)
    
    # Place Base A: only in positions where Base B can exist within distance constraints
    base_a_valid = valid_base_a_mask(grid_dims, min_generals_distance, max_generals_distance)
    pos_a = sample_from_mask(base_a_valid, keys[2])
    grid = grid.at[pos_a].set(1)
    
    # Place Base B: Manhattan distance as placement heuristic (BFS enforced after terrain)
    dist_from_a = manhattan_distance_from(pos_a, grid_dims)
    base_b_valid = dist_from_a >= min_generals_distance
    if max_generals_distance is not None:
        base_b_valid = base_b_valid & (dist_from_a <= max_generals_distance)
    pos_b = sample_from_mask(base_b_valid, keys[3])
    grid = grid.at[pos_b].set(2)
    
    # =================================================================
    # Step 2: Mark protected zones (within distance 6 of each general)
    # =================================================================
    dist_from_b = manhattan_distance_from(pos_b, grid_dims)
    
    near_a = (dist_from_a > 0) & (dist_from_a <= 6)
    near_b = (dist_from_b > 0) & (dist_from_b <= 6)
    
    # =================================================================
    # Step 3: Place one castle near each general (in protected zones)
    # =================================================================
    # Generate castle values using keys[4] and keys[5]
    castle_val_a = jax.random.randint(keys[4], (), castle_val_range[0], castle_val_range[1])
    castle_val_b = jax.random.randint(keys[5], (), castle_val_range[0], castle_val_range[1])
    
    # Sample castle positions using keys[6] and keys[7] (separate from value keys!)
    # Castle near A (not on A or B)
    castle_a_mask = near_a & (grid == 0)
    pos_castle_a = sample_from_mask(castle_a_mask, keys[6])
    grid = grid.at[pos_castle_a].set(castle_val_a)
    
    # Castle near B (not on A, B, or first castle)
    castle_b_mask = near_b & (grid == 0)
    pos_castle_b = sample_from_mask(castle_b_mask, keys[7])
    grid = grid.at[pos_castle_b].set(castle_val_b)
    
    # =================================================================
    # Step 4: Place mountains (can be anywhere empty, including protected zones)
    # =================================================================
    # Available for mountains: any empty cell (mountains CAN be in protected zones)
    mountain_available = (grid == 0)  # Just empty cells
    flat_available = mountain_available.reshape(-1)
    
    # Use Gumbel-max + top_k for mountain placement
    logits = jnp.where(flat_available, 0.0, -jnp.inf)
    gumbel_noise = jax.random.gumbel(keys[8], shape=logits.shape)
    scores = logits + gumbel_noise
    
    # Get indices for mountains (use static max and mask extras)
    max_mountains = num_tiles // 4  # Upper bound for static shape
    _, mountain_indices = jax.lax.top_k(scores, max_mountains)
    
    # Create flat grid, place mountains only up to num_mountains
    flat_grid = grid.reshape(-1)
    mountain_mask = jnp.arange(max_mountains) < num_mountains
    
    # Place mountains at selected indices
    def place_mountain(flat_grid, idx_and_mask):
        idx, should_place = idx_and_mask
        return jnp.where(should_place, flat_grid.at[idx].set(-2), flat_grid), None
    
    flat_grid, _ = jax.lax.scan(place_mountain, flat_grid, (mountain_indices, mountain_mask))
    grid = flat_grid.reshape(grid_dims)
    
    # =================================================================
    # Step 5: Place remaining cities (num_cities - 2, since 2 are castles)
    # =================================================================
    remaining_cities = num_cities - 2
    city_available = (grid == 0)  # Any empty cell
    flat_city_available = city_available.reshape(-1)
    
    city_logits = jnp.where(flat_city_available, 0.0, -jnp.inf)
    city_gumbel = jax.random.gumbel(keys[9], shape=city_logits.shape)
    city_scores = city_logits + city_gumbel
    
    # Cap max_extra_cities at the grid size to avoid top_k errors
    max_extra_cities = min(20, flat_city_available.shape[0])
    _, city_indices = jax.lax.top_k(city_scores, max_extra_cities)
    
    # Generate random city values
    city_values = jax.random.randint(keys[10], (max_extra_cities,), castle_val_range[0], castle_val_range[1])
    city_mask = jnp.arange(max_extra_cities) < remaining_cities
    
    flat_grid = grid.reshape(-1)
    
    def place_city(flat_grid, args):
        idx, val, should_place = args
        return jnp.where(should_place, flat_grid.at[idx].set(val), flat_grid), None
    
    flat_grid, _ = jax.lax.scan(place_city, flat_grid, (city_indices, city_values, city_mask))
    grid = flat_grid.reshape(grid_dims)
    
    # =================================================================
    # Step 6: Ensure connectivity (carve L-path if needed)
    # =================================================================
    connected = flood_fill_connected(grid, pos_a, pos_b)
    grid = jax.lax.cond(
        connected,
        lambda g: g,  # Already connected, do nothing
        lambda g: carve_l_path(g, pos_a, pos_b),  # Carve path
        grid
    )

    # Step 6b: Enforce max BFS distance (carve L-path if path is too long)
    if max_generals_distance is not None:
        dist = bfs_distance(grid, pos_a, pos_b)
        grid = jax.lax.cond(
            dist > max_generals_distance,
            lambda g: carve_l_path(g, pos_a, pos_b),
            lambda g: g,
            grid
        )
    
    # =================================================================
    # Step 7: Dynamic padding
    # =================================================================
    # Default padding: max dimension + 1 (for batching)
    if pad_to is None:
        target_size = max(h, w) + 1
    else:
        target_size = pad_to
    
    # Pad both dimensions to target_size
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    
    if pad_h > 0 or pad_w > 0:
        grid = jnp.pad(
            grid,
            ((0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=-2,  # Mountains
        )
    
    # Grid is always valid by construction
    return grid


def sample_from_mask(mask: jax.Array, key: jax.random.PRNGKey) -> tuple[int, int]:
    """
    Sample one index from a boolean mask using Gumbel-max trick.
    XLA-efficient alternative to jax.random.choice.
    
    Args:
        mask: 2D boolean array where True indicates valid positions
        key: JAX random key
        
    Returns:
        (i, j) tuple of the sampled position
    """
    flat_mask = mask.reshape(-1).astype(jnp.float32)
    logits = jnp.where(flat_mask > 0, 0.0, -jnp.inf)
    gumbel_noise = jax.random.gumbel(key, shape=logits.shape)
    idx = jnp.argmax(logits + gumbel_noise)
    return jnp.unravel_index(idx, mask.shape)


def sample_k_from_mask(mask: jax.Array, k: int, key: jax.random.PRNGKey) -> jax.Array:
    """
    Sample k indices from a boolean mask using Gumbel-max trick + top_k.
    Maintains static shapes for XLA compatibility.
    
    Args:
        mask: 2D boolean array where True indicates valid positions
        k: Number of positions to sample (must be static)
        key: JAX random key
        
    Returns:
        Array of shape (k,) containing flat indices of sampled positions
    """
    flat_mask = mask.reshape(-1).astype(jnp.float32)
    logits = jnp.where(flat_mask > 0, 0.0, -jnp.inf)
    gumbel_noise = jax.random.gumbel(key, shape=logits.shape)
    scores = logits + gumbel_noise
    _, top_indices = jax.lax.top_k(scores, k)
    return top_indices


def manhattan_distance_from(pos: tuple[int, int], grid_shape: tuple[int, int]) -> jax.Array:
    """
    Compute Manhattan distance from a position to all cells in grid.
    
    Args:
        pos: (i, j) position
        grid_shape: (height, width) of grid
        
    Returns:
        2D array of Manhattan distances
    """
    h, w = grid_shape
    i_idx = jnp.arange(h)[:, None]
    j_idx = jnp.arange(w)[None, :]
    return jnp.abs(i_idx - pos[0]) + jnp.abs(j_idx - pos[1])


def valid_base_a_mask(grid_shape: tuple[int, int], min_distance: int, max_distance: int | None = None) -> jax.Array:
    """
    Create mask of valid positions for Base A.
    A position is valid if there exists at least one cell >= min_distance away
    (and optionally <= max_distance away).
    
    For a cell (i,j), the max Manhattan distance to any corner is:
    max(i+j, i+(w-1-j), (h-1-i)+j, (h-1-i)+(w-1-j))
    
    Args:
        grid_shape: (height, width) of grid
        min_distance: Minimum required distance to Base B
        max_distance: Maximum allowed distance to Base B (None = no limit)
        
    Returns:
        2D boolean mask
    """
    h, w = grid_shape
    i_idx = jnp.arange(h)[:, None]
    j_idx = jnp.arange(w)[None, :]
    
    # Distance to each corner
    dist_top_left = i_idx + j_idx
    dist_top_right = i_idx + (w - 1 - j_idx)
    dist_bottom_left = (h - 1 - i_idx) + j_idx
    dist_bottom_right = (h - 1 - i_idx) + (w - 1 - j_idx)
    
    max_dist = jnp.maximum(
        jnp.maximum(dist_top_left, dist_top_right),
        jnp.maximum(dist_bottom_left, dist_bottom_right)
    )
    
    # Min distance constraint
    valid = max_dist >= min_distance
    
    # Max distance constraint (if specified)
    # For max constraint, we need the min distance to any corner to be within max_distance
    if max_distance is not None:
        min_dist = jnp.minimum(
            jnp.minimum(dist_top_left, dist_top_right),
            jnp.minimum(dist_bottom_left, dist_bottom_right)
        )
        # At least one position should be reachable within max_distance
        # This means the grid diagonal should allow it
        grid_diagonal = h + w - 2
        # If we can't satisfy max_distance, just use min_distance
        valid = jnp.where(
            grid_diagonal >= max_distance,
            valid & (min_dist <= max_distance),
            valid
        )
    
    return valid


def flood_fill_connected(grid: jax.Array, start_pos: tuple[int, int], end_pos: tuple[int, int]) -> bool:
    """
    Check if start_pos can reach end_pos using parallel flood fill with early termination.
    Uses jax.lax.while_loop for efficient early exit when target is reached.
    
    Args:
        grid: 2D grid array (-2=mountain, 0=passable, 1/2=generals, 40-50=cities)
        start_pos: Starting position (i, j)
        end_pos: Target position (i, j)
        
    Returns:
        Boolean indicating if end_pos is reachable from start_pos
    """
    h, w = grid.shape
    
    # Everything except mountains is passable (empty, generals, cities)
    passable = (grid != -2)
    
    # Initialize: only start position is reachable
    reachable = jnp.zeros((h, w), dtype=jnp.bool_)
    reachable = reachable.at[start_pos].set(True)
    
    def dilate(reachable):
        """Single dilation step - expand reachable cells to neighbors."""
        # 4-neighbor dilation using roll + boundary fix
        up = jnp.roll(reachable, -1, axis=0).at[-1, :].set(False)
        down = jnp.roll(reachable, 1, axis=0).at[0, :].set(False)
        left = jnp.roll(reachable, -1, axis=1).at[:, -1].set(False)
        right = jnp.roll(reachable, 1, axis=1).at[:, 0].set(False)
        return (reachable | up | down | left | right) & passable
    
    def cond_fn(state):
        reachable, prev_reachable, _ = state
        # Continue if: target not reached AND frontier is still expanding
        target_reached = reachable[end_pos]
        still_expanding = jnp.any(reachable != prev_reachable)
        return ~target_reached & still_expanding
    
    def body_fn(state):
        reachable, _, step = state
        new_reachable = dilate(reachable)
        return (new_reachable, reachable, step + 1)
    
    # Initialize with one dilation already done
    initial_reachable = dilate(reachable)
    init_state = (initial_reachable, reachable, jnp.int32(1))
    
    final_reachable, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
    
    return final_reachable[end_pos]


def bfs_distance(grid: jax.Array, start_pos: tuple[int, int], end_pos: tuple[int, int]) -> jax.Array:
    """
    Compute shortest path (BFS) distance between two positions.
    Only mountains (-2) are impassable.

    Args:
        grid: 2D grid array
        start_pos: Starting position (i, j)
        end_pos: Target position (i, j)

    Returns:
        Scalar integer: BFS distance, or h*w if unreachable.
    """
    h, w = grid.shape
    passable = (grid != -2)

    reached = jnp.zeros((h, w), dtype=jnp.bool_)
    reached = reached.at[start_pos].set(True)

    def dilate(r):
        up = jnp.roll(r, -1, axis=0).at[-1, :].set(False)
        down = jnp.roll(r, 1, axis=0).at[0, :].set(False)
        left = jnp.roll(r, -1, axis=1).at[:, -1].set(False)
        right = jnp.roll(r, 1, axis=1).at[:, 0].set(False)
        return (r | up | down | left | right) & passable

    def cond_fn(state):
        r, prev_r, _ = state
        return ~r[end_pos] & jnp.any(r != prev_r)

    def body_fn(state):
        r, _, step = state
        return (dilate(r), r, step + 1)

    first = dilate(reached)
    final_r, _, final_step = jax.lax.while_loop(
        cond_fn, body_fn, (first, reached, jnp.int32(1))
    )

    return jnp.where(final_r[end_pos], final_step, h * w)


def carve_l_path(grid: jax.Array, pos_a: tuple[int, int], pos_b: tuple[int, int]) -> jax.Array:
    """
    Carve an L-shaped path between two positions using jnp.where (no branching).
    Clears mountains and cities on the path, preserves generals.
    
    The path goes: horizontal from pos_a to (pos_a[0], pos_b[1]), 
                   then vertical to pos_b.
    
    Args:
        grid: 2D grid array
        pos_a: Start position (i, j)
        pos_b: End position (i, j)
        
    Returns:
        Grid with L-shaped path carved (obstacles removed)
    """
    h, w = grid.shape
    i1, j1 = pos_a
    j2 = pos_b[1]
    i2 = pos_b[0]
    
    # Create coordinate grids
    i_idx = jnp.arange(h)[:, None]
    j_idx = jnp.arange(w)[None, :]
    
    # Horizontal segment: row i1, columns from min(j1, j2) to max(j1, j2)
    h_mask = (i_idx == i1) & \
             (j_idx >= jnp.minimum(j1, j2)) & \
             (j_idx <= jnp.maximum(j1, j2))
    
    # Vertical segment: column j2, rows from min(i1, i2) to max(i1, i2)
    v_mask = (j_idx == j2) & \
             (i_idx >= jnp.minimum(i1, i2)) & \
             (i_idx <= jnp.maximum(i1, i2))
    
    path_mask = h_mask | v_mask
    
    # Clear obstacles on path, but preserve generals (values 1, 2)
    is_obstacle = (grid == -2) | (grid >= 40)  # Mountain or city
    grid = jnp.where(path_mask & is_obstacle, 0, grid)
    
    return grid