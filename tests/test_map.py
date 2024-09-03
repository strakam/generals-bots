import numpy as np
from generals.utils import map_from_generator, validate_map, map_from_string


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

    map = """
.....
.A##2
##.2.
..###
...B.
    """
    map = map_from_string(map)
    assert not validate_map(map)

    map = """
.....
.A##2
##.2.
..2##
...B.
    """
    map = map_from_string(map)
    assert validate_map(map)

    map = """
..#..
.A##2
##.2.
..2##
...B.
    """
    map = map_from_string(map)
    assert not validate_map(map)

    map = """
.....
BA2#2
##.2.
..2##
.....
    """
    map = map_from_string(map)
    assert validate_map(map)

    map = """
...#.
#A2#2
##.#B
..2##
.....
    """
    map = map_from_string(map)
    assert not validate_map(map)

    map = """
...#.
A#2#2
##.#B
..2#.
.....
    """
    map = map_from_string(map)
    assert validate_map(map)


def test_generate_map():
    grid_size, mountain_density, town_density  = 16, 0.1, 0.1
    for _ in range(5):
        map = map_from_generator(grid_size, mountain_density, town_density)
        assert validate_map(map)  # map has to be valid
        assert map.shape == (grid_size, grid_size)

    grid_size, mountain_density, town_density  = 10, 0.2, 0.2
    for _ in range(5):
        map = map_from_generator(grid_size, mountain_density, town_density)
        assert validate_map(map)  # map has to be valid
        assert map.shape == (grid_size, grid_size)
