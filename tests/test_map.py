import numpy as np
from generals.utils import generate_map, validate_map, map_from_string


def test_validate_map():
    map = """
.....
.A##2
...2.
..22.
...B.
    """
    map = map_from_string(map)
    assert validate_map(map)

    map = \
    """
.....
.A##2
##.2.
..###
...B.
    """
    map = map_from_string(map)
    assert not validate_map(map)

    map = \
    """
.....
.A##2
##.2.
..2##
...B.
    """
    map = map_from_string(map)
    assert validate_map(map)

    map = \
    """
..#..
.A##2
##.2.
..2##
...B.
    """
    map = map_from_string(map)
    assert not validate_map(map)

    map = \
    """
.....
BA2#2
##.2.
..2##
.....
    """
    map = map_from_string(map)
    assert validate_map(map)

    map = \
    """
...#.
#A2#2
##.#B
..2##
.....
    """
    map = map_from_string(map)
    assert not validate_map(map)

    map = \
    """
...#.
A#2#2
##.#B
..2#.
.....
    """
    map = map_from_string(map)
    assert validate_map(map)

def test_generate_map():
    grid_size, mountain_density, town_density, n_generals = 16, 0.1, 0.1, 2
    for _ in range(5):
        map = generate_map(grid_size, mountain_density, town_density, n_generals)
        assert validate_map(map) # map has to be valid
        assert map.shape == (grid_size, grid_size)
        generals = np.isin(map, ['A', 'B'])
        assert np.sum(generals) == n_generals

    grid_size, mountain_density, town_density, n_generals = 10, 0.2, 0.2, 2
    for _ in range(5):
        map = generate_map(grid_size, mountain_density, town_density, n_generals)
        assert validate_map(map) # map has to be valid
        assert map.shape == (grid_size, grid_size)
        generals = np.isin(map, ['A', 'B'])
        assert np.sum(generals) == n_generals
