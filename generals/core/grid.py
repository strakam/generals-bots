from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=['num_players', 'grid_dims', 'pad_to', 'mountain_density_range',
                                    'num_cities_range', 'min_generals_distance',
                                    'max_generals_distance', 'castle_val_range'])
def generate_grid(
    key: jax.random.PRNGKey,
    num_players: int = 2,
    grid_dims: tuple[int, int] = (23, 23),
    pad_to: int | None = None,
    mountain_density_range: tuple[float, float] = (0.18, 0.26),
    num_cities_range: tuple[int, int] = (9, 11),
    min_generals_distance: int = 17,
    max_generals_distance: int | None = None,
    castle_val_range: tuple[int, int] = (40, 51),
) -> jnp.ndarray:
    """
    Generate a grid for N players with guaranteed validity.

    Algorithm:
    1. Place N generals sequentially, each at min_generals_distance from all earlier ones
    2. Place mountains on remaining empty cells
    3. Place N castles, one within BFS distance 6 of each general
    4. Place remaining cities (num_cities - N)
    5. Ensure each general is reachable from player 0's general (carve L-path if not)
    6. Apply dynamic padding to pad_to

    Args:
        key: JAX random key
        num_players: Number of player generals to place (1..39).
        grid_dims: Grid dimensions (height, width).
        pad_to: Pad grid to this size for batching (None = max(h, w) + 1).
        mountain_density_range: (min, max) fraction of tiles that are mountains.
        num_cities_range: (min, max) total number of cities (including the N castles).
        min_generals_distance: Minimum BFS distance between any two generals.
        max_generals_distance: Maximum BFS distance between any two generals (None = no limit).
        castle_val_range: (min, max) army value for cities.

    Returns:
        Grid is always valid by construction. General positions are encoded as 1..N.
    """
    # Static number of keys; allocate generously.
    keys = jax.random.split(key, 12 + 2 * num_players)
    key_idx = 0

    h, w = grid_dims
    num_tiles = h * w

    num_cities = jax.random.randint(keys[0], (), num_cities_range[0], num_cities_range[1] + 1)

    min_mountains = int(mountain_density_range[0] * num_tiles)
    max_mountains = int(mountain_density_range[1] * num_tiles)
    num_mountains = jax.random.randint(keys[1], (), min_mountains, max_mountains + 1)

    # =================================================================
    # Step 1: Place N generals on an empty grid, respecting distance constraints
    # =================================================================
    grid = jnp.full(grid_dims, 0, dtype=jnp.int32)

    # First general can sit anywhere a second general can also fit (preserves the
    # N=2 valid_base_a_mask behavior; for N>2 this is a weaker but reasonable check).
    first_valid = valid_base_a_mask(grid_dims, min_generals_distance, max_generals_distance)
    positions = []
    pos = sample_from_mask(first_valid, keys[2])
    positions.append(pos)
    grid = grid.at[pos].set(1)

    # Each subsequent general must be in min/max distance of all previously placed
    for i in range(1, num_players):
        valid = jnp.ones(grid_dims, dtype=bool)
        for prev_pos in positions:
            dist = manhattan_distance_from(prev_pos, grid_dims)
            valid = valid & (dist >= min_generals_distance)
            if max_generals_distance is not None:
                valid = valid & (dist <= max_generals_distance)
        # Must not collide with an existing general
        valid = valid & (grid == 0)
        # Fallback: if nothing satisfies the constraint, drop max_distance to allow some placement
        has_any = jnp.any(valid)
        relaxed = jnp.ones(grid_dims, dtype=bool)
        for prev_pos in positions:
            dist = manhattan_distance_from(prev_pos, grid_dims)
            relaxed = relaxed & (dist >= min_generals_distance)
        relaxed = relaxed & (grid == 0)
        valid = jnp.where(has_any, valid, relaxed)
        pos = sample_from_mask(valid, keys[3 + i])
        positions.append(pos)
        grid = grid.at[pos].set(i + 1)

    # =================================================================
    # Step 2: Place mountains on remaining empty cells
    # =================================================================
    mountain_available = (grid == 0)
    flat_available = mountain_available.reshape(-1)

    logits = jnp.where(flat_available, 0.0, -jnp.inf)
    gumbel_noise = jax.random.gumbel(keys[8], shape=logits.shape)
    scores = logits + gumbel_noise

    max_mountains_static = num_tiles // 4
    _, mountain_indices = jax.lax.top_k(scores, max_mountains_static)

    flat_grid = grid.reshape(-1)
    mountain_mask = jnp.arange(max_mountains_static) < num_mountains

    def place_mountain(flat_grid, idx_and_mask):
        idx, should_place = idx_and_mask
        return jnp.where(should_place, flat_grid.at[idx].set(-2), flat_grid), None

    flat_grid, _ = jax.lax.scan(place_mountain, flat_grid, (mountain_indices, mountain_mask))
    grid = flat_grid.reshape(grid_dims)

    # =================================================================
    # Step 3: Place one castle within BFS distance 6 of each general
    # =================================================================
    castle_key_base = 12  # offset into keys[] for castle keys
    for i, pos in enumerate(positions):
        castle_val = jax.random.randint(keys[castle_key_base + i * 2], (), castle_val_range[0], castle_val_range[1])
        near = bfs_reachable_within_k(grid, pos, 6)
        candidates = near & (grid == 0)
        has_candidates = jnp.any(candidates)
        # Fallback: convert a nearby mountain into a castle slot
        fallback = near & (grid == -2)
        mask = jnp.where(has_candidates, candidates, fallback)
        castle_pos = sample_from_mask(mask, keys[castle_key_base + i * 2 + 1])
        grid = grid.at[castle_pos].set(castle_val)

    # =================================================================
    # Step 4: Place remaining cities
    # =================================================================
    remaining_cities = num_cities - num_players
    city_available = (grid == 0)
    flat_city_available = city_available.reshape(-1)

    city_logits = jnp.where(flat_city_available, 0.0, -jnp.inf)
    city_gumbel = jax.random.gumbel(keys[9], shape=city_logits.shape)
    city_scores = city_logits + city_gumbel

    max_extra_cities = min(20, flat_city_available.shape[0])
    _, city_indices = jax.lax.top_k(city_scores, max_extra_cities)

    city_values = jax.random.randint(keys[10], (max_extra_cities,), castle_val_range[0], castle_val_range[1])
    city_mask = jnp.arange(max_extra_cities) < remaining_cities

    flat_grid = grid.reshape(-1)

    def place_city(flat_grid, args):
        idx, val, should_place = args
        return jnp.where(should_place, flat_grid.at[idx].set(val), flat_grid), None

    flat_grid, _ = jax.lax.scan(place_city, flat_grid, (city_indices, city_values, city_mask))
    grid = flat_grid.reshape(grid_dims)

    # =================================================================
    # Step 5: Connectivity — each general must be reachable from positions[0]
    # =================================================================
    pos_anchor = positions[0]
    for i in range(1, num_players):
        connected = flood_fill_connected(grid, pos_anchor, positions[i])
        grid = jax.lax.cond(
            connected,
            lambda g: g,
            lambda g, p=positions[i]: carve_l_path(g, pos_anchor, p),
            grid,
        )

    if max_generals_distance is not None:
        for i in range(1, num_players):
            dist = bfs_distance(grid, pos_anchor, positions[i])
            grid = jax.lax.cond(
                dist > max_generals_distance,
                lambda g, p=positions[i]: carve_l_path(g, pos_anchor, p),
                lambda g: g,
                grid,
            )

    # =================================================================
    # Step 6: Dynamic padding
    # =================================================================
    if pad_to is None:
        target_size = max(h, w) + 1
    else:
        target_size = pad_to

    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)

    if pad_h > 0 or pad_w > 0:
        grid = jnp.pad(
            grid,
            ((0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=-2,
        )

    return grid


def sample_from_mask(mask: jax.Array, key: jax.random.PRNGKey) -> tuple[int, int]:
    """Sample one index from a boolean mask using Gumbel-max trick."""
    flat_mask = mask.reshape(-1).astype(jnp.float32)
    logits = jnp.where(flat_mask > 0, 0.0, -jnp.inf)
    gumbel_noise = jax.random.gumbel(key, shape=logits.shape)
    idx = jnp.argmax(logits + gumbel_noise)
    return jnp.unravel_index(idx, mask.shape)


def sample_k_from_mask(mask: jax.Array, k: int, key: jax.random.PRNGKey) -> jax.Array:
    """Sample k indices from a boolean mask using Gumbel-max trick + top_k."""
    flat_mask = mask.reshape(-1).astype(jnp.float32)
    logits = jnp.where(flat_mask > 0, 0.0, -jnp.inf)
    gumbel_noise = jax.random.gumbel(key, shape=logits.shape)
    scores = logits + gumbel_noise
    _, top_indices = jax.lax.top_k(scores, k)
    return top_indices


def manhattan_distance_from(pos: tuple[int, int], grid_shape: tuple[int, int]) -> jax.Array:
    """Compute Manhattan distance from a position to all cells in grid."""
    h, w = grid_shape
    i_idx = jnp.arange(h)[:, None]
    j_idx = jnp.arange(w)[None, :]
    return jnp.abs(i_idx - pos[0]) + jnp.abs(j_idx - pos[1])


def valid_base_a_mask(grid_shape: tuple[int, int], min_distance: int, max_distance: int | None = None) -> jax.Array:
    """Mask of positions for the first general such that at least one cell satisfies the distance constraint."""
    h, w = grid_shape
    i_idx = jnp.arange(h)[:, None]
    j_idx = jnp.arange(w)[None, :]

    dist_top_left = i_idx + j_idx
    dist_top_right = i_idx + (w - 1 - j_idx)
    dist_bottom_left = (h - 1 - i_idx) + j_idx
    dist_bottom_right = (h - 1 - i_idx) + (w - 1 - j_idx)

    max_dist = jnp.maximum(
        jnp.maximum(dist_top_left, dist_top_right),
        jnp.maximum(dist_bottom_left, dist_bottom_right)
    )

    valid = max_dist >= min_distance

    if max_distance is not None:
        min_dist = jnp.minimum(
            jnp.minimum(dist_top_left, dist_top_right),
            jnp.minimum(dist_bottom_left, dist_bottom_right)
        )
        grid_diagonal = h + w - 2
        valid = jnp.where(
            grid_diagonal >= max_distance,
            valid & (min_dist <= max_distance),
            valid
        )

    return valid


def bfs_reachable_within_k(grid: jax.Array, start_pos: tuple[int, int], k: int) -> jax.Array:
    """BFS flood fill from start_pos for k steps over non-mountain terrain."""
    h, w = grid.shape
    passable = grid != -2

    reachable = jnp.zeros((h, w), dtype=jnp.bool_)
    reachable = reachable.at[start_pos].set(True)

    def dilate(reachable):
        up = jnp.roll(reachable, -1, axis=0).at[-1, :].set(False)
        down = jnp.roll(reachable, 1, axis=0).at[0, :].set(False)
        left = jnp.roll(reachable, -1, axis=1).at[:, -1].set(False)
        right = jnp.roll(reachable, 1, axis=1).at[:, 0].set(False)
        return (reachable | up | down | left | right) & passable

    def body_fn(_, reachable):
        return dilate(reachable)

    reachable = jax.lax.fori_loop(0, k, body_fn, reachable)

    reachable = reachable.at[start_pos].set(False)
    return reachable


def flood_fill_connected(grid: jax.Array, start_pos: tuple[int, int], end_pos: tuple[int, int]) -> bool:
    """Check if start_pos can reach end_pos through non-mountain, non-city cells."""
    h, w = grid.shape

    # Passable for connectivity: not a mountain, not a city. Generals and empty cells qualify.
    passable = (grid != -2) & (grid < 40)

    reachable = jnp.zeros((h, w), dtype=jnp.bool_)
    reachable = reachable.at[start_pos].set(True)

    def dilate(reachable):
        up = jnp.roll(reachable, -1, axis=0).at[-1, :].set(False)
        down = jnp.roll(reachable, 1, axis=0).at[0, :].set(False)
        left = jnp.roll(reachable, -1, axis=1).at[:, -1].set(False)
        right = jnp.roll(reachable, 1, axis=1).at[:, 0].set(False)
        return (reachable | up | down | left | right) & passable

    def cond_fn(state):
        reachable, prev_reachable, _ = state
        target_reached = reachable[end_pos]
        still_expanding = jnp.any(reachable != prev_reachable)
        return ~target_reached & still_expanding

    def body_fn(state):
        reachable, _, step = state
        new_reachable = dilate(reachable)
        return (new_reachable, reachable, step + 1)

    initial_reachable = dilate(reachable)
    init_state = (initial_reachable, reachable, jnp.int32(1))

    final_reachable, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)

    return final_reachable[end_pos]


def bfs_distance(grid: jax.Array, start_pos: tuple[int, int], end_pos: tuple[int, int]) -> jax.Array:
    """Shortest path distance between two positions through non-mountain, non-city cells."""
    h, w = grid.shape
    passable = (grid != -2) & (grid < 40)

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
    """Carve an L-shaped path between two positions (clear mountains and cities on the path)."""
    h, w = grid.shape
    i1, j1 = pos_a
    j2 = pos_b[1]
    i2 = pos_b[0]

    i_idx = jnp.arange(h)[:, None]
    j_idx = jnp.arange(w)[None, :]

    h_mask = (i_idx == i1) & \
             (j_idx >= jnp.minimum(j1, j2)) & \
             (j_idx <= jnp.maximum(j1, j2))

    v_mask = (j_idx == j2) & \
             (i_idx >= jnp.minimum(i1, i2)) & \
             (i_idx <= jnp.maximum(i1, i2))

    path_mask = h_mask | v_mask

    is_obstacle = (grid == -2) | (grid >= 40)
    grid = jnp.where(path_mask & is_obstacle, 0, grid)

    return grid
