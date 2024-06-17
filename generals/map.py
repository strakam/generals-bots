import numpy as np
import importlib.resources

PASSABLE, MOUNTAIN, CITY, GENERAL = 0, 1, 2, 3

def generate_map(
    grid_size: int = 10,
    mountain_density: float = 0.1,
    town_density: float = 0.1,
    n_generals: int = 2
) -> np.ndarray:

    spatial_dim = (grid_size, grid_size)
    p_neutral = 1 - mountain_density - town_density
    probs = [p_neutral, mountain_density, town_density]
    map = np.random.choice(
        [PASSABLE, MOUNTAIN, CITY],
        size=spatial_dim,
        p=probs
    )

    # place generals
    passable_indices = np.argwhere(map == PASSABLE)
    # pick random indices for generals
    general_indices = passable_indices[
        np.random.choice(len(passable_indices), size=n_generals, replace=False)
    ]
    for i, idx in enumerate(general_indices):
        map[idx[0], idx[1]] = GENERAL + i

    # generate until valid map
    return map if validate_map(map) else generate_map(grid_size, mountain_density, town_density, n_generals)

def load_map(map_name: str) -> np.ndarray:
    """
    Load map from file.

    Args:
        map_name: str

    Returns:
        np.ndarray: map layout
    """
    try:
        with importlib.resources.path('generals.maps', map_name) as path:
            with open(path, 'r') as f:
                map = np.array([list(line.strip()) for line in f]).astype(np.float32)
                validity = validate_map(map)
                if not validity:
                    raise ValueError('The map is invalid, because generals are separated by mountains')
            return map
    except ValueError:
        raise ValueError('Invalid map format or shape')

def validate_map(map: np.ndarray) -> bool:
    """
    Validate map layout.
    WORKS ONLY FOR 2 GENERALS (for now)

    Args:
        map: np.ndarray

    Returns:
        bool: True if map is valid, False otherwise
    """
    # DFS

    generals = np.argwhere(np.isin(map, [3, 4]))
    start, end = generals[0], generals[1]

    def dfs(map, visited, square):
        i, j = square
        if i < 0 or i >= map.shape[0] or j < 0 or j >= map.shape[1] or visited[i, j]:
            return
        if map[i, j] == MOUNTAIN:
            return
        visited[i, j] = True
        for di, dj in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            new_square = (i + di, j + dj)
            dfs(map, visited, new_square)

    visited = np.zeros_like(map, dtype=bool)
    dfs(map, visited, start)
    return visited[end[0], end[1]]


