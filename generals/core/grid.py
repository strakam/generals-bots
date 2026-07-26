from functools import partial

import jax
import jax.numpy as jnp


def _bfs_field_from_mask(passable: jax.Array, start: jax.Array) -> jax.Array:
    """bfs_distance_field, seeded by a boolean mask so it can be vmapped."""
    h, w = passable.shape
    unreachable = h * w
    dist = jnp.where(start, 0, unreachable).astype(jnp.int32)

    def cond_fn(state):
        _, _, _, grew = state
        return grew

    def body_fn(state):
        reached, dist, step, _ = state
        grown = _dilate4(reached) & passable
        newly = grown & ~reached
        return (grown, jnp.where(newly, step, dist), step + 1, jnp.any(newly))

    _, dist, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (start, dist, jnp.int32(1), jnp.bool_(True))
    )
    return dist


def bfs_distance_field(passable: jax.Array, start_pos: tuple[int, int]) -> jax.Array:
    """Step distance from `start_pos` to every cell, over 4-connected open ground.

    One breadth-first dilation: each round expands the reached set by one step
    and stamps that round's number onto the cells newly touched, so the value
    left behind IS the shortest path length. Unreachable cells (and cells behind
    mountains) keep the sentinel h*w, which is larger than any real path.

    Same loop as bfs_distance, but it answers "how far is everything" in one
    pass rather than "how far is that one cell" — which is what makes spawn
    placement a single BFS instead of one per candidate.

    Args:
        passable: (h, w) boolean mask of walkable cells; start_pos must be True
        start_pos: source (i, j)

    Returns:
        (h, w) int32 array of step distances, h*w where unreachable.
    """
    start = jnp.zeros(passable.shape, dtype=jnp.bool_).at[start_pos].set(True)
    return _bfs_field_from_mask(passable, start)


def _dilate4(mask: jax.Array) -> jax.Array:
    """Expand a boolean mask to its 4-neighbours (no wrap-around at the borders)."""
    up = jnp.roll(mask, -1, axis=0).at[-1, :].set(False)
    down = jnp.roll(mask, 1, axis=0).at[0, :].set(False)
    left = jnp.roll(mask, -1, axis=1).at[:, -1].set(False)
    right = jnp.roll(mask, 1, axis=1).at[:, 0].set(False)
    return mask | up | down | left | right


# How many spawn spots to try for general A before settling. Each costs one BFS
# dilation and they run batched, so this is cheap insurance against A landing in
# a sealed pocket — a board that would otherwise seat both generals on top of
# each other. 4 makes that outcome vanishingly rare (see tests/test_spawn_distance.py).
SPAWN_CANDIDATES = 4


@partial(jax.jit, static_argnames=['grid_dims', 'pad_to', 'mountain_density_range', 'num_castles_range',
                                    'min_generals_distance', 'max_generals_distance', 'castle_val_range'])
def generate_grid(
    key: jax.random.PRNGKey,
    grid_dims: tuple[int, int] = (23, 23),
    pad_to: int | None = None,
    mountain_density_range: tuple[float, float] = (0.18, 0.26),
    num_castles_range: tuple[int, int] = (9, 11),
    min_generals_distance: int = 17,
    max_generals_distance: int | None = None,
    castle_val_range: tuple[int, int] = (40, 51),
) -> jnp.ndarray:
    """
    Generate grid using JAX-optimal algorithm with guaranteed validity.
    
    Unified generator that supports both square and non-square grids with
    dynamic padding and configurable general distance constraints.
    
    Algorithm:
    1. Place mountains on the empty grid — terrain FIRST, so the spawn
       constraint below is a real walking distance and not a straight-line proxy
    2. Carve the castles out of mountains, also before anyone spawns: a castle
       is walkable, so carving it later would open routes around the very ridge
       the spawn distance was measured on
    3. Place general A on open ground, preferring a castle within ~6 steps
    4. One BFS dilation from A: step distance to every reachable cell
    5. Sample general B among cells at least min_generals_distance steps away
       (connected by construction, so nothing needs carving afterwards)
    6. Re-assert the two generals (ground truth — the grid always has both)
    7. Apply dynamic padding

    Spawn distance is measured by BFS over open ground, so two generals with a
    ridge between them count as far apart even when the straight line is short —
    and a board is never thrown away for a short straight line that is a long
    walk. Because every terrain decision precedes both spawns, the distance the
    generator enforces is the distance the match is played at.
    
    Args:
        key: JAX random key
        grid_dims: Grid dimensions (height, width) - supports non-square grids
        pad_to: Pad grid to this size for batching (None = max(h, w) + 1)
        mountain_density_range: (min, max) fraction of tiles that are mountains
        num_castles_range: (min, max) number of castles to place
        min_generals_distance: Minimum BFS (shortest path over open ground) distance between generals
        max_generals_distance: Maximum BFS (shortest path over open ground) distance between generals (None = no limit)
        castle_val_range: (min, max) army value for castles
        
    Returns:
        A valid grid: exactly two generals, an empty connecting path between
        them, and a castle within walking distance ~6 of each general.
    """
    keys = jax.random.split(key, 12)

    h, w = grid_dims
    num_tiles = h * w

    # Random number of castles in range
    num_castles = jax.random.randint(keys[0], (), num_castles_range[0], num_castles_range[1] + 1)

    # Number of mountains: sample uniformly from density range
    min_mountains = int(mountain_density_range[0] * num_tiles)
    max_mountains = int(mountain_density_range[1] * num_tiles)
    num_mountains = jax.random.randint(keys[1], (), min_mountains, max_mountains + 1)

    # =================================================================
    # Step 1: Place mountains on the empty grid.
    # Terrain goes down BEFORE the generals so the spawn constraint can be a
    # walking distance. (Placing generals first only ever admitted a
    # straight-line proxy, since there was no terrain yet to walk around.)
    # =================================================================
    grid = jnp.full(grid_dims, 0, dtype=jnp.int32)

    # Gumbel-max + top_k: every cell is a candidate, extras masked off
    gumbel_noise = jax.random.gumbel(keys[8], shape=(num_tiles,))

    # Get indices for mountains (use static max and mask extras)
    max_mountains = num_tiles // 4  # Upper bound for static shape
    _, mountain_indices = jax.lax.top_k(gumbel_noise, max_mountains)

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
    # Step 2: Cut the castles out of the mountains NOW, before anyone spawns.
    #
    # This ordering is the whole ballgame. A castle is walkable — capturable in
    # the base game, and stripped to plain ground outright under the
    # build-castles ruleset — so every castle carved from a mountain can open a
    # new route. Placing them after the generals let a run of castles tunnel through
    # the ridge the spawn distance was measured around: boards generated at 17+
    # were played at 8. Carving first means the distance field below is measured
    # on the terrain the match is actually played on.
    # =================================================================
    mountain_flat = (grid == -2).reshape(-1)
    castle_logits = jnp.where(mountain_flat, 0.0, -jnp.inf)
    castle_scores = castle_logits + jax.random.gumbel(keys[9], shape=castle_logits.shape)
    max_castles = min(20, num_tiles)  # static bound for top_k
    _, castle_indices = jax.lax.top_k(castle_scores, max_castles)
    castle_values = jax.random.randint(keys[10], (max_castles,), castle_val_range[0], castle_val_range[1])
    # Only take slots that are (a) wanted and (b) actually a mountain — on a
    # tiny or nearly mountain-free grid there may be fewer mountains than
    # castles, and a castle must never land on open ground it would then block.
    castle_mask = (jnp.arange(max_castles) < num_castles) & mountain_flat[castle_indices]

    def place_castle_cell(flat_grid, args):
        idx, val, should_place = args
        return jnp.where(should_place, flat_grid.at[idx].set(val), flat_grid), None

    flat_grid, _ = jax.lax.scan(place_castle_cell, grid.reshape(-1),
                                (castle_indices, castle_values, castle_mask))
    grid = flat_grid.reshape(grid_dims)

    # =================================================================
    # Step 3: General A on open ground, preferring a spot with a castle within
    # ~6 steps so each player has an early expansion target (the castles are
    # already down, so this is a mask lookup rather than another conversion —
    # which would have moved the terrain again).
    # =================================================================
    passable = grid != -2                      # castles included: they are walkable
    near_castle = grid > 2
    for _ in range(6):
        near_castle = _dilate4(near_castle) & passable

    # Random mountains leave the odd sealed-off pocket, and a general stranded in
    # one has nowhere far to put its opponent. Draw SPAWN_CANDIDATES spots at
    # once (Gumbel top-k over open ground, biased toward an early castle) and
    # keep the first that can actually seat an opponent at range. Fixed work, no
    # rejection loop: this is why the L-path carve could be dropped, and with it
    # the repair step that used to shorten the very distance being enforced.
    logits = jnp.where(passable, jnp.where(near_castle, 20.0, 0.0), -jnp.inf).reshape(-1)
    _, cand_flat = jax.lax.top_k(logits + jax.random.gumbel(keys[2], shape=(num_tiles,)),
                                 SPAWN_CANDIDATES)

    # =================================================================
    # Step 4: One BFS dilation per candidate — step distance to every cell.
    # =================================================================
    unreachable = num_tiles  # the field's "no route" sentinel

    def field_from_flat(idx):
        start = (jnp.arange(num_tiles) == idx).reshape(grid_dims) & passable
        return _bfs_field_from_mask(passable, start)

    fields = jax.vmap(field_from_flat)(cand_flat)              # (K, h, w)
    reach = (fields < unreachable) & passable[None]
    if max_generals_distance is not None:
        reach = reach & (fields <= max_generals_distance)
    far = reach & (fields >= min_generals_distance)

    # Prefer the earliest candidate with somewhere far enough to put B; if none
    # has (a very tight board), take whichever reaches furthest.
    seats = jnp.any(far, axis=(1, 2))
    span = jnp.max(jnp.where(reach, fields, -1), axis=(1, 2))
    order = jnp.arange(SPAWN_CANDIDATES)
    pick = jnp.argmax(jnp.where(seats, 2 * num_tiles - order, span))

    chosen = cand_flat[pick]
    pos_first = (chosen // w, chosen % w)
    dist_from_first = fields[pick]
    allowed = reach[pick]
    far_enough = far[pick]

    # =================================================================
    # Step 5: General B among the cells far enough away by that distance. B is
    # drawn from the cells REACHED by the dilation, so the two generals are
    # connected by construction — no L-path carve, and therefore no repair step
    # that could quietly shorten the spawn distance it just enforced.
    #
    # Fallback: if A landed in a pocket too small to hold anyone far enough
    # away, take the farthest cell that IS reachable. That keeps generation
    # total (never rejects, never loops) at the cost of a closer spawn on those
    # rare boards; tests/test_spawn_distance.py measures how often it happens.
    # =================================================================
    # same early-castle preference as A, dropped if it would leave nothing
    near_and_far = far_enough & near_castle
    far_enough = jnp.where(jnp.any(near_and_far), near_and_far, far_enough)
    farthest = allowed & (dist_from_first == jnp.max(jnp.where(allowed, dist_from_first, -1)))
    second_valid = jnp.where(jnp.any(far_enough), far_enough, farthest)
    pos_second = sample_from_mask(second_valid, keys[3])

    # Randomly assign which position becomes p0 (Base A) vs p1 (Base B)
    swap = jax.random.bernoulli(keys[11])
    pos_a = jax.tree.map(lambda a, b: jnp.where(swap, b, a), pos_first, pos_second)
    pos_b = jax.tree.map(lambda a, b: jnp.where(swap, a, b), pos_first, pos_second)

    grid = grid.at[pos_a].set(1)
    grid = grid.at[pos_b].set(2)

    # =================================================================
    # Step 6: Re-assert the generals as ground truth — the grid ALWAYS ends with
    # exactly one P0 (1) and one P1 (2) general, so a spawn can never go missing.
    # =================================================================
    grid = grid.at[pos_a].set(1)
    grid = grid.at[pos_b].set(2)
    
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


def bfs_reachable_within_k(grid: jax.Array, start_pos: tuple[int, int], k: int) -> jax.Array:
    """
    BFS flood fill from start_pos for exactly k steps.
    Returns boolean mask of all cells reachable within k BFS steps.
    The start cell itself is excluded from the result.

    Passable cells: not mountains (-2). Generals and empty cells are passable.

    Args:
        grid: 2D grid array
        start_pos: Starting position (i, j)
        k: Number of BFS steps

    Returns:
        2D boolean mask of reachable cells (excluding start_pos)
    """
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

    # Exclude the start position itself
    reachable = reachable.at[start_pos].set(False)
    return reachable


def flood_fill_connected(grid: jax.Array, start_pos: tuple[int, int], end_pos: tuple[int, int]) -> bool:
    """
    Check if start_pos can reach end_pos using parallel flood fill with early termination.
    Uses jax.lax.while_loop for efficient early exit when target is reached.
    
    Args:
        grid: 2D grid array (-2=mountain, 0=passable, 1/2=generals, 40-50=castles)
        start_pos: Starting position (i, j)
        end_pos: Target position (i, j)
        
    Returns:
        Boolean indicating if end_pos is reachable from start_pos
    """
    h, w = grid.shape
    
    # Only empty tiles and generals are passable (not castles/castles)
    passable = (grid >= 0) & (grid <= 2)
    
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

    See also bfs_distance_field, which is this same dilation but keeps the
    distance to EVERY cell instead of one — that is what spawn placement uses.
    """
    h, w = grid.shape
    # Only empty tiles and generals are passable (not castles/castles)
    passable = (grid >= 0) & (grid <= 2)

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
    Clears mountains and castles on the path, preserves generals.
    
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
    is_obstacle = (grid == -2) | (grid > 2)  # Mountain or castle
    grid = jnp.where(path_mask & is_obstacle, 0, grid)
    
    return grid