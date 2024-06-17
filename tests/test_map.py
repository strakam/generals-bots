import numpy as np
from generals.map import generate_map, validate_map


def test_validate_map():
    map = np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 1, 1, 2],
        [0, 0, 0, 2, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 0, 4, 0],
    ])
    assert validate_map(map)

    map = np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 1, 1, 2],
        [1, 1, 0, 2, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 4, 0],
    ])
    assert not validate_map(map)

    map = np.array([
        [0, 0, 0, 0, 0],
        [0, 3, 1, 1, 2],
        [1, 1, 0, 2, 0],
        [0, 0, 2, 1, 1],
        [0, 0, 0, 4, 0],
    ])
    assert validate_map(map)

    map = np.array([
        [0, 0, 1, 0, 0],
        [0, 3, 1, 1, 2],
        [1, 1, 0, 2, 0],
        [0, 0, 2, 1, 1],
        [0, 0, 0, 4, 0],
    ])
    assert not validate_map(map)

    map = np.array([
        [0, 0, 0, 0, 0],
        [4, 3, 2, 1, 2],
        [1, 1, 0, 2, 0],
        [0, 0, 2, 1, 1],
        [0, 0, 0, 0, 0],
    ])
    assert validate_map(map)

    map = np.array([
        [0, 0, 0, 1, 0],
        [1, 3, 2, 1, 2],
        [1, 1, 0, 1, 4],
        [0, 0, 2, 1, 1],
        [0, 0, 0, 0, 0],
    ])
    assert not validate_map(map)

    map = np.array([
        [0, 0, 0, 1, 0],
        [3, 1, 2, 1, 2],
        [1, 1, 0, 1, 4],
        [0, 0, 2, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    assert validate_map(map)

def test_generate_map():
    grid_size, mountain_density, town_density, n_generals = 16, 0.1, 0.1, 2
    for _ in range(5):
        map = generate_map(grid_size, mountain_density, town_density, n_generals)
        assert validate_map(map) # map has to be valid
        assert map.shape == (grid_size, grid_size)
        generals = np.isin(map, [3, 4]) # ONLY FOR 2 PLAYERS
        assert np.sum(generals) == n_generals
        # count tiles with number bigger than 4
        assert np.sum(map > 4) == 0

    grid_size, mountain_density, town_density, n_generals = 10, 0.2, 0.2, 2
    for _ in range(5):
        map = generate_map(grid_size, mountain_density, town_density, n_generals)
        assert validate_map(map) # map has to be valid
        assert map.shape == (grid_size, grid_size)
        generals = np.isin(map, [3, 4]) # ONLY FOR 2 PLAYERS
        assert np.sum(generals) == n_generals
        # count tiles with number bigger than 4
        assert np.sum(map > 4) == 0
