from functools import partial
from typing import Literal, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import MOUNTAIN

DEFAULT_MIN_GRID_DIM = (18, 18)
DEFAULT_MAX_GRID_DIM = (23, 23)
DEFAULT_GRID_DIMS = (15, 18)
DEFAULT_MOUNTAIN_DENSITY = 0.2
DEFAULT_CITY_DENSITY = 0.05
MAX_GENERALSIO_ATTEMPTS = 20
MIN_GENERALS_DISTANCE = 17
RADIUS_FROM_GENERAL = [6]


@partial(jax.jit, static_argnames=['mode', 'grid_dims', 'pad_to', 'mountain_density', 
                                    'num_castles_range', 'min_generals_distance', 
                                    'castle_val_range'])
def generate_grid(
    key: jax.random.PRNGKey,
    mode: Literal['fixed', 'generalsio'] = 'generalsio',
    grid_dims: Tuple[int, int] = (20, 20),
    pad_to: int = 24,
    mountain_density: float = 0.2,
    num_castles_range: Tuple[int, int] = (9, 15),
    min_generals_distance: int = 17,
    castle_val_range: Tuple[int, int] = (40, 51),
) -> Tuple[jnp.ndarray, bool]:
    """
    Generate a Generals.io grid using JAX.
    
    Args:
        key: JAX random key
        mode: 'fixed' for fixed grid size, 'generalsio' for random size like online game
        grid_dims: Grid dimensions (height, width). 
            - If mode='fixed': exact size
            - If mode='generalsio': base size, actual will be random 18-23
        pad_to: Pad grid to this size (for batching)
        mountain_density: Fraction of tiles that are mountains (base density)
        num_castles_range: (min, max) number of castles to place
        min_generals_distance: Minimum distance between generals
        castle_val_range: (min, max) army value for castles
    
    Returns:
        (grid, is_valid): Numeric grid and validity flag
            Grid encoding: -2=mountain, 0=passable, 1=general1, 2=general2, 40-50=cities
    
    Examples:
        # Generals.io mode (random size 18-23, like online game)
        >>> key = jax.random.PRNGKey(0)
        >>> grid, valid = generate_grid(key, mode='generalsio')
        
        # Fixed size mode (custom training)
        >>> grid, valid = generate_grid(key, mode='fixed', grid_dims=(15, 15), pad_to=16)
    """
    if mode == 'generalsio':
        return _generate_generalsio_grid(
            key, pad_to, mountain_density, num_castles_range, 
            min_generals_distance, castle_val_range
        )
    else:  # mode == 'fixed'
        return _generate_fixed_grid(
            key, grid_dims, pad_to, mountain_density, num_castles_range,
            min_generals_distance, castle_val_range
        )


@partial(jax.jit, static_argnames=['pad_to', 'mountain_density', 'num_castles_range',
                                    'min_generals_distance', 'castle_val_range'])
def _generate_generalsio_grid(
    key: jax.random.PRNGKey,
    pad_to: int,
    mountain_density: float,
    num_castles_range: Tuple[int, int],
    min_generals_distance: int,
    castle_val_range: Tuple[int, int],
) -> Tuple[jnp.ndarray, bool]:
    """
    Generate grid with random size like generals.io online game.
    
    Grid size: 18-23 x 18-23 (random)
    Castles: 9-14 (5 + random [4,5,6])
    Mountains: base_density + 2% random variation
    
    Note: Generates at max size (23x23) then pads. The actual grid boundaries
    are determined by mountain placement at the edges.
    """
    _, *subkeys = jax.random.split(key, 11)  # Need 11 subkeys (10 + 1 for _)
    
    # Use fixed max size for JAX compatibility (23x23)
    # We'll pad the unused area with mountains
    max_size = 23
    grid_dims = (max_size, max_size)
    num_tiles = max_size * max_size
    
    # Castles: 5 + choice([4, 5, 6]) = 9-11 total
    castle_bonus = jax.random.choice(subkeys[2], jnp.array([4, 5, 6]))
    num_cities = 5 + castle_bonus  # 9-11 cities
    
    # Mountains: base density + 2% variation
    mountain_variation = 0.02 * num_tiles * jax.random.uniform(subkeys[3])
    num_mountains = mountain_density * num_tiles + mountain_variation
    
    # Generate grid at max size
    grid = jnp.full((max_size, max_size), 0, dtype=jnp.int32)
    grid = place_mountains(grid, num_mountains, subkeys[4])
    grid = place_cities(grid, num_cities - 2, subkeys[5])  # -2 because 2 near generals
    
    # Place generals
    grid, (i1, j1) = place_value_on_mask(grid, grid == 0, 1, subkeys[6])
    distances_from_g1 = bfs_distances((i1, j1), grid)
    
    grid, (i2, j2) = place_value_on_mask(grid, distances_from_g1 > min_generals_distance, 2, subkeys[7])
    distances_from_g2 = bfs_distances((i2, j2), grid)
    
    # Place one castle near each general
    castle_val_1 = jax.random.randint(subkeys[8], (), castle_val_range[0], castle_val_range[1])
    castle_val_2 = jax.random.randint(subkeys[9], (), castle_val_range[0], castle_val_range[1])
    grid, _ = place_value_on_mask(grid, distances_from_g1 <= 6, castle_val_1, subkeys[8])
    grid, _ = place_value_on_mask(grid, distances_from_g2 <= 6, castle_val_2, subkeys[9])
    
    # Pad to target size
    if pad_to > max_size:
        grid = jnp.pad(
            grid,
            ((0, pad_to - max_size), (0, pad_to - max_size)),
            mode='constant',
            constant_values=-2,  # Mountains
        )
    
    # Validate
    g1_reachable = jnp.sum((distances_from_g1 > 0) & (distances_from_g1 <= 5))
    g2_reachable = jnp.sum((distances_from_g2 > 0) & (distances_from_g2 <= 5))
    far_enough = distances_from_g1[i2, j2] > min_generals_distance
    fair_spawn = jnp.abs(g1_reachable - g2_reachable) < 10
    valid = far_enough & fair_spawn
    
    return grid, valid


@partial(jax.jit, static_argnames=['grid_dims', 'pad_to', 'mountain_density', 
                                    'num_castles_range', 'min_generals_distance', 
                                    'castle_val_range'])
def _generate_fixed_grid(
    key: jax.random.PRNGKey,
    grid_dims: Tuple[int, int],
    pad_to: int,
    mountain_density: float,
    num_castles_range: Tuple[int, int],
    min_generals_distance: int,
    castle_val_range: Tuple[int, int],
) -> Tuple[jnp.ndarray, bool]:
    """
    Generate grid with fixed dimensions.
    
    Useful for custom training scenarios.
    """
    _, *subkeys = jax.random.split(key, 9)
    num_tiles = grid_dims[0] * grid_dims[1]
    
    # Random number of castles in range
    num_cities = jax.random.randint(subkeys[0], (), num_castles_range[0], num_castles_range[1])
    num_mountains = mountain_density * num_tiles
    
    # Generate grid
    grid = jnp.full(grid_dims, 0, dtype=jnp.int32)
    grid = place_mountains(grid, num_mountains, subkeys[2])
    grid = place_cities(grid, num_cities - 2, subkeys[3])
    
    # Place generals
    grid, (i1, j1) = place_value_on_mask(grid, grid == 0, 1, subkeys[4])
    distances_from_g1 = bfs_distances((i1, j1), grid)
    
    grid, (i2, j2) = place_value_on_mask(grid, distances_from_g1 > min_generals_distance, 2, subkeys[5])
    distances_from_g2 = bfs_distances((i2, j2), grid)
    
    # Place castles near generals
    castle_val_1 = jax.random.randint(subkeys[6], (), castle_val_range[0], castle_val_range[1])
    castle_val_2 = jax.random.randint(subkeys[7], (), castle_val_range[0], castle_val_range[1])
    grid, _ = place_value_on_mask(grid, distances_from_g1 <= 6, castle_val_1, subkeys[6])
    grid, _ = place_value_on_mask(grid, distances_from_g2 <= 6, castle_val_2, subkeys[7])
    
    # Pad
    grid = jnp.pad(
        grid,
        ((0, pad_to - grid_dims[0]), (0, pad_to - grid_dims[1])),
        mode='constant',
        constant_values=-2,
    )
    
    # Validate
    g1_reachable = jnp.sum((distances_from_g1 > 0) & (distances_from_g1 <= 5))
    g2_reachable = jnp.sum((distances_from_g2 > 0) & (distances_from_g2 <= 5))
    far_enough = distances_from_g1[i2, j2] > min_generals_distance
    fair_spawn = jnp.abs(g1_reachable - g2_reachable) < 10
    valid = far_enough & fair_spawn
    
    return grid, valid


# =============================================================================
# Helper functions (used by both modes)
# =============================================================================


class InvalidGridError(Exception):
    pass


class Grid:
    """
    Represents the game grid containing passable areas, mountains, cities, and generals.

    Attributes:
        grid (np.ndarray): 2D array representing the grid.
    """

    def __init__(self, grid: str | np.ndarray):
        """
        Initializes the Grid either from a string or a NumPy array.

        Args:
            grid (str | np.ndarray): The grid representation.

        Raises:
            ValueError: If grid is not a string or NumPy array.
            InvalidGridError: If the grid layout is invalid.
        """
        if not isinstance(grid, str | np.ndarray):
            raise ValueError(f"grid must be a str or np.ndarray. Received grid with type: {type(grid)}.")

        if isinstance(grid, str):
            grid = grid.strip()
            grid = Grid.numpify_grid(grid)

        Grid.ensure_grid_is_valid(grid)
        self.grid = grid

    @property
    def shape(self):
        return self.grid.shape

    @staticmethod
    def ensure_grid_is_valid(grid: np.ndarray):
        if not Grid.are_generals_connected(grid):
            raise InvalidGridError("Invalid grid layout - generals cannot reach each other.")

        # check that exactly one 'A' and one 'B' are present in the grid
        first_general = np.argwhere(np.isin(grid, ["A"]))
        second_general = np.argwhere(np.isin(grid, ["B"]))
        if len(first_general) != 1 or len(second_general) != 1:
            raise InvalidGridError("Exactly one 'A' and one 'B' should be present in the grid.")

    def __eq__(self, other):
        return np.array_equal(self.grid, other.grid)

    @staticmethod
    def generals_distance(grid: "Grid") -> int:
        generals = np.argwhere(np.isin(grid.grid, ["A", "B"]))
        return abs(generals[0][0] - generals[1][0]) + abs(generals[0][1] - generals[1][1])

    @staticmethod
    def numpify_grid(grid: str) -> np.ndarray:
        return np.array([list(row) for row in grid.strip().split("\n")])

    @staticmethod
    def stringify_grid(grid: np.ndarray) -> str:
        return "\n".join(["".join(row) for row in grid])

    @staticmethod
    def are_generals_connected(grid: np.ndarray | str) -> bool:
        """
        Returns True if there is a path connecting the two generals.
        """
        if isinstance(grid, str):
            grid = Grid.numpify_grid(grid)

        height, width = grid.shape

        def dfs(grid, visited, square):
            i, j = square
            if i < 0 or i >= height or j < 0 or j >= width or visited[i, j]:
                return
            if grid[i, j] == MOUNTAIN or str(grid[i, j]).isdigit() or grid[i, j] == "x":  # mountain or city
                return
            visited[i, j] = True
            for di, dj in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                new_square = (i + di, j + dj)
                dfs(grid, visited, new_square)

        generals = np.argwhere(np.isin(grid, ["A", "B"]))
        start, end = generals[0], generals[1]

        visited = np.zeros_like(grid, dtype=bool)
        dfs(grid, visited, start)
        return visited[end[0], end[1]]

    def __str__(self):
        return Grid.stringify_grid(self.grid)

def place_value_on_mask(grid: jnp.ndarray, mask: jnp.ndarray, value: int, key: jax.random.PRNGKey):
    """
    Given mask of zeros and ones, pick random position with "1" and put "value" on it.
    Make it jit-able.
    """
    mask = mask.reshape(-1)
    num_ones = jnp.sum(mask)

    p = mask / num_ones
    idx = jax.random.choice(key, jnp.arange(len(p)), shape=(), p=p)
    idx = jnp.unravel_index(idx, grid.shape)
    grid = grid.at[idx].set(value)
    return grid, idx


def create_distance_mask(distances: jax.Array, max_distance: int):
    return (distances <= max_distance) & (distances > 0)


def random_ranks(flat_map, key):
    # Get a random permutation's inverse: each index's position in the permutation.
    perm = jax.random.permutation(key, jnp.arange(flat_map.shape[0]))
    return jnp.argsort(perm)


def place_mountains(map: jax.Array, num_mountains: int, key):
    flat_map = map.reshape(-1)
    ranks = random_ranks(flat_map, key)
    updated = jnp.where(ranks < num_mountains, -2, flat_map)
    return updated.reshape(map.shape)


def place_cities(map: jax.Array, num_cities: int, key):
    flat_map = map.reshape(-1)
    ranks = random_ranks(flat_map, key)
    # Generate random city armies with the same shape as flat_map.
    city_armies = jax.random.randint(key, shape=flat_map.shape, minval=40, maxval=51)
    updated = jnp.where(ranks < num_cities, city_armies, flat_map)
    return updated.reshape(map.shape)


def bfs_distances(start_pos: tuple[int, int], grid: jax.Array) -> jax.Array:
    """
    Computes distances from a starting position to all other cells in the grid using BFS.
    JAX-compatible implementation that works with jax.jit.
    Can only traverse through cells with value 0.

    Args:
        start_pos: Starting position (i, j) for BFS
        grid: 2D JAX array representing the grid

    Returns:
        2D JAX array of the same shape as grid, where each cell contains the
        shortest distance from start_pos, or infinity if unreachable
    """
    height, width = grid.shape
    max_steps = height * width  # Maximum possible BFS steps

    # Initialize distances matrix with infinity
    distances = jnp.full(grid.shape, -1)
    # Set starting position to 0
    distances = distances.at[start_pos].set(0)

    # Create initial frontier matrix (1 for cells in current frontier, 0 otherwise)
    frontier = jnp.zeros(grid.shape, dtype=jnp.bool_)
    frontier = frontier.at[start_pos].set(True)

    # Define BFS step function for while_loop
    def bfs_step(state):
        current_step, current_distances, current_frontier = state

        # Create new frontier from neighbors of current frontier
        new_frontier = jnp.zeros_like(current_frontier)

        # Expand in four directions
        for di, dj in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            # Shift the frontier in each direction
            shifted_i = jnp.roll(current_frontier, di, axis=0)
            if di > 0:  # Fix boundary wrapping
                shifted_i = shifted_i.at[:di].set(False)
            elif di < 0:
                shifted_i = shifted_i.at[di:].set(False)

            shifted = jnp.roll(shifted_i, dj, axis=1)
            if dj > 0:  # Fix boundary wrapping
                shifted = shifted.at[:, :dj].set(False)
            elif dj < 0:
                shifted = shifted.at[:, dj:].set(False)

            # Cells must be passable (0), not visited yet (-1 distance), and neighbors of current frontier
            valid_cells = ((grid >= 0) & (grid < 10)) & (current_distances == -1) & shifted
            new_frontier = new_frontier | valid_cells

        # Update distances for new frontier cells
        new_distances = jnp.where(new_frontier, current_step + 1, current_distances)

        return current_step + 1, new_distances, new_frontier

    # Define while_loop condition
    def condition(state):
        current_step, _, current_frontier = state
        # Continue if frontier is not empty and we haven't reached max steps
        return (jnp.any(current_frontier)) & (current_step < max_steps)

    # Run BFS using jax.lax.while_loop
    _, final_distances, _ = jax.lax.while_loop(condition, bfs_step, (0, distances, frontier))

    return final_distances
