from generals.map import Mapper
import numpy as np

def test_validate_map():
    mapper = Mapper()
    map = """
.....
.A##2
...2.
..22.
...B.
    """
    assert mapper.validate_map(map)

    map = """
.....
.A##2
##.2.
..###
...B.
    """
    assert not mapper.validate_map(map)

    map = """
.....
.A##2
##.2.
..2##
...B.
    """
    assert mapper.validate_map(map)

    map = """
..#..
.A##2
##.2.
..2##
...B.
    """
    assert not mapper.validate_map(map)

    map = """
.....
BA2#2
##.2.
..2##
.....
    """
    assert mapper.validate_map(map)

    map = """
...#.
#A2#2
##.#B
..2##
.....
    """
    assert not mapper.validate_map(map)

    map = """
...#.
A#2#2
##.#B
..2#.
.....
    """
    assert mapper.validate_map(map)


def test_generate_map():
    mapper = Mapper()
    grid_dims = (16, 16)
    mountain_density, city_density = 0.1, 0.1
    for _ in range(5):
        map_str = mapper.generate_map(grid_dims, mountain_density, city_density)
        map = mapper.numpify_map(map_str)
        assert mapper.validate_map(map_str)  # map has to be valid
        assert map.shape == grid_dims

    grid_dims = (10, 10)
    mountain_density, town_density = 0.2, 0.2
    for _ in range(5):
        map_str = mapper.generate_map(grid_dims, mountain_density, town_density)
        assert mapper.validate_map(map_str)  # map has to be valid
        map = mapper.numpify_map(map_str)
        assert map.shape == grid_dims

def test_numpify_map():
    mapper = Mapper()
    map_str = """
.....
.A##2
...2.
..22.
...B.
    """
    map = mapper.numpify_map(map_str)
    reference_map = np.array([
        [".", ".", ".", ".", "."],
        [".", "A", "#", "#", "2"],
        [".", ".", ".", "2", "."],
        [".", ".", "2", "2", "."],
        [".", ".", ".", "B", "."],
    ])
    assert (map == reference_map).all()

def test_stringify_map():

    mapper = Mapper()
    # make map different than from previous example
    np_map = np.array([
        ["#", ".", ".", ".", "A"],
        [".", ".", "#", "#", "2"],
        [".", ".", ".", "2", "."],
        ["4", ".", "2", "2", "."],
        [".", "1", "#", "B", "#"],
    ])

    map_str = mapper.stringify_map(np_map)
    reference_map = """
#...A
..##2
...2.
4.22.
.1#B#
    """
    assert map_str == reference_map.strip()
