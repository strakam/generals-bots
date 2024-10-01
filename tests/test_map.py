from generals.grid import Grid
import numpy as np


def test_verify_grid():
    map = """
.....
.A##2
...2.
..22.
...B.
    """
    map = Grid.numpify_grid(map)
    assert Grid.verify_grid(map)

    map = """
.....
.A##2
##.2.
..###
...B.
    """

    map = Grid.numpify_grid(map)
    assert not Grid.verify_grid(map)

    map = """
.....
.A##2
##.2.
..2##
...B.
    """
    map = Grid.numpify_grid(map)
    assert Grid.verify_grid(map)

    map = """
..#..
.A##2
##.2.
..2##
...B.
    """
    map = Grid.numpify_grid(map)
    assert not Grid.verify_grid(map)

    map = """
.....
BA2#2
##.2.
..2##
.....
    """
    map = Grid.numpify_grid(map)
    assert Grid.verify_grid(map)

    map = """
...#.
#A2#2
##.#B
..2##
.....
    """
    map = Grid.numpify_grid(map)
    assert not Grid.verify_grid(map)

    map = """
...#.
A#2#2
##.#B
..2#.
.....
    """
    map = Grid.numpify_grid(map)
    assert Grid.verify_grid(map)


def test_numpify_map():
    map_str = """
.....
.A##2
...2.
..22.
...B.
    """
    map = Grid.numpify_grid(map_str)
    reference_map = np.array(
        [
            [".", ".", ".", ".", "."],
            [".", "A", "#", "#", "2"],
            [".", ".", ".", "2", "."],
            [".", ".", "2", "2", "."],
            [".", ".", ".", "B", "."],
        ]
    )
    assert (map == reference_map).all()


def test_stringify_map():
    # make map different than from previous example
    np_map = np.array(
        [
            ["#", ".", ".", ".", "A"],
            [".", ".", "#", "#", "2"],
            [".", ".", ".", "2", "."],
            ["4", ".", "2", "2", "."],
            [".", "1", "#", "B", "#"],
        ]
    )

    map_str = Grid.stringify_grid(np_map)
    reference_map = """
#...A
..##2
...2.
4.22.
.1#B#
    """
    assert map_str == reference_map.strip()
