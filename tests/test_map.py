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

