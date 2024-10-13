import itertools

import numpy as np
import pytest

import generals.core.game as game
from generals.core.grid import Grid, GridFactory


def get_game(grid=None):
    if grid is None:
        grid_factory = GridFactory(
            grid_dims=(4, 4),
            mountain_density=0.1,
            city_density=0.1,
            general_positions=[[3, 3], [1, 3]],
        )
        grid = grid_factory.grid_from_generator()
    return game.Game(grid, ["red", "blue"])


def test_grid_creation():
    """
    For given configuration, we should get grid of given size.
    """
    for _ in range(10):
        game = get_game()
        assert game.grid_dims == (4, 4)

        # mountain and city should be disjoint
        assert np.logical_and(game.channels.mountain, game.channels.city).sum() == 0

        owners = ["neutral"] + game.agents
        # for every pair of agents, the ownership channels should be disjoint
        pairs = itertools.combinations(owners, 2)
        for pair in pairs:
            ownership_a = game.channels.ownership[pair[0]]
            ownership_b = game.channels.ownership[pair[1]]
            assert np.logical_and(ownership_a, ownership_b).sum() == 0

        # but union of all ownerships should be equal to passable channel
        ownerships = [game.channels.ownership[owner] for owner in owners]
        union = np.logical_or.reduce(ownerships)
        assert (union == game.channels.passable).all()


def test_channel_to_indices():
    """
    For given channel, we should get indices of cells that are 1.
    """
    map = """...#
#..A
#..#
.#.B
"""
    grid = Grid(map)
    game = get_game(grid)

    channel = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    reference = np.array([(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)])
    indices = game.channel_to_indices(channel)
    assert (indices == reference).all()

    channel = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    reference = np.empty((0, 2))
    indices = game.channel_to_indices(channel)
    assert (indices == reference).all()


def test_visibility_channel():
    """
    For given ownership mask, we should get visibility mask.
    """
    dummy_game = get_game()

    ownership = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    reference = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    visibility = dummy_game.visibility_channel(ownership)
    assert (visibility == reference).all()

    ownership = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ]
    )
    reference = np.array(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ]
    )
    visibility = dummy_game.visibility_channel(ownership)
    assert (visibility == reference).all()


def test_action_mask():
    """
    For given ownership mask and passable mask, we should get NxNx4 mask of valid actions.
    """
    game = get_game()
    game.grid_dims = (4, 4)
    game.channels.army = np.array(
        [
            [3, 0, 1, 0],
            [0, 3, 6, 2],
            [1, 1, 5, 0],
            [2, 0, 5, 8],
        ],
        dtype=int,
    )
    game.channels._set_passable(
        np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 0, 0, 0],
            ],
            dtype=bool,
        )
    )

    game.channels.ownership["red"] = np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    )
    reference = np.array(
        [
            # UP
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            # DOWN
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            # LEFT
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ],
            # RIGHT
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
        dtype=bool,
    )

    action_mask = game.action_mask("red")
    assert (action_mask[:, :, 0] == reference[0]).all()
    assert (action_mask[:, :, 1] == reference[1]).all()
    assert (action_mask[:, :, 2] == reference[2]).all()
    assert (action_mask[:, :, 3] == reference[3]).all()


def test_observations():
    """
    For given actions, we should get new state of the game.
    """
    map = """...#
#..A
#..#
.#.B
"""
    grid = Grid(map)
    game = get_game(grid)
    game.channels.ownership["red"] = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    game.channels.ownership["blue"] = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    game.channels.army = np.array(
        [
            [3, 0, 0, 0],
            [0, 3, 6, 2],
            [1, 9, 5, 0],
            [0, 0, 5, 8],
        ],
        dtype=np.float32,
    )
    game.channels.ownership["neutral"] = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    ############
    # TEST RED #
    ############
    red_observation = game.agent_observation("red")["observation"]
    reference_opponent_ownership = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (red_observation["opponent_cells"] == reference_opponent_ownership).all()

    reference_neutral_ownership = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (red_observation["neutral_cells"] == reference_neutral_ownership).all()

    reference_ownership = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    assert (red_observation["owned_cells"] == reference_ownership).all()

    # union of all ownerships should be zero
    assert (
        np.logical_and.reduce(
            [
                red_observation["opponent_cells"],
                red_observation["neutral_cells"],
                red_observation["owned_cells"],
            ]
        )
    ).sum() == 0

    reference_army = np.array(
        [
            [0, 0, 0, 0],
            [0, 3, 6, 2],
            [1, 9, 5, 0],
            [0, 0, 5, 8],
        ],
        dtype=np.float32,
    )
    assert (red_observation["army"] == reference_army).all()

    #############
    # TEST BLUE #
    #############
    blue_observation = game.agent_observation("blue")["observation"]
    reference_opponent_ownership = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (blue_observation["opponent_cells"] == reference_opponent_ownership).all()

    reference_neutral_ownership = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (blue_observation["neutral_cells"] == reference_neutral_ownership).all()

    reference_ownership = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (blue_observation["owned_cells"] == reference_ownership).all()

    reference_army = np.array(
        [
            [3, 0, 0, 0],
            [0, 3, 6, 2],
            [1, 9, 5, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (blue_observation["army"] == reference_army).all()

    # union of all ownerships should be zero
    assert (
        np.logical_and.reduce(
            [
                blue_observation["opponent_cells"],
                blue_observation["neutral_cells"],
                blue_observation["owned_cells"],
            ]
        )
    ).sum() == 0


def test_game_step():
    """
    Test a number of moves from this situation
    """
    map = """...#
#..A
#..#
.#.B
"""
    grid = Grid(map)
    game = get_game(grid)
    game.channels.ownership["red"] = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=int,
    )
    game.channels.ownership["blue"] = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    game.channels.army = np.array(
        [
            [3, 0, 0, 0],
            [0, 3, 6, 2],
            [1, 9, 5, 0],
            [0, 0, 5, 8],
        ],
        dtype=np.float32,
    )
    game.channels.ownership["neutral"] = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    # Test capturing (if equal armies meet, the defender keeps the cell)
    #############################################################################################################
    # red moves from (2, 1) UP (captures blue square), blue moves from (1, 2) DOWN, (doesnt capture red square) #
    #############################################################################################################
    red_move = {
        "pass": 0,
        "cell": np.array([2, 1]),
        "direction": 0,
        "split": 0,
    }
    blue_move = {
        "pass": 0,
        "cell": np.array([1, 2]),
        "direction": 1,
        "split": 0,
    }
    moves = {
        "red": red_move,
        "blue": blue_move,
    }
    game.step(moves)
    reference_army = np.array(
        [
            [3, 0, 0, 0],
            [0, 5, 1, 2],
            [1, 1, 0, 0],
            [0, 0, 5, 8],
        ],
        dtype=np.float32,
    )
    assert (game.channels.army == reference_army).all()

    reference_ownership_red = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership["red"] == reference_ownership_red).all()

    reference_ownership_blue = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership["blue"] == reference_ownership_blue).all()

    reference_ownership_neutral = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership_neutral == reference_ownership_neutral).all()

    reference_total_army_red = 20
    stats = game.get_infos()
    assert stats["red"]["army"] == reference_total_army_red

    reference_total_army_blue = 6
    assert stats["blue"]["army"] == reference_total_army_blue

    reference_total_army_land = 6
    assert stats["red"]["land"] == reference_total_army_land

    reference_total_army_land = 3
    assert stats["blue"]["land"] == reference_total_army_land

    # Test raising of warning on invalid move (moving from cell with only 1 army)
    ##################################################################################
    # Now red moves from (2, 1) DOWN (invalid move) and blue moves from (0, 0) RIGHT #
    ##################################################################################
    red_move = {
        "pass": 0,
        "cell": np.array([2, 1]),
        "direction": 1,
        "split": 0,
    }
    blue_move = {
        "pass": 0,
        "cell": np.array([0, 0]),
        "direction": 3,
        "split": 0,
    }
    moves = {
        "red": red_move,
        "blue": blue_move,
    }

    with pytest.warns(UserWarning):  # we expect a warning
        game.step(moves)

    # this is second move, so army increments in base
    reference_army = np.array(
        [
            [1, 2, 0, 0],
            [0, 5, 1, 3],
            [1, 1, 0, 0],
            [0, 0, 5, 9],
        ],
        dtype=np.float32,
    )
    assert (game.channels.army == reference_army).all()

    reference_ownership_red = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership["red"] == reference_ownership_red).all()

    reference_ownership_blue = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership["blue"] == reference_ownership_blue).all()

    reference_ownership_neutral = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership_neutral == reference_ownership_neutral).all()

    reference_total_army_red = 21
    stats = game.get_infos()
    assert stats["red"]["army"] == reference_total_army_red

    reference_total_army_blue = 7
    assert stats["blue"]["army"] == reference_total_army_blue

    reference_total_army_land = 6
    assert stats["red"]["land"] == reference_total_army_land

    reference_total_army_land = 4
    assert stats["blue"]["land"] == reference_total_army_land

    # Test splitting of army
    #####################################################################################
    # Red sends half army from (3, 3) LEFT and blue sends half army from (1, 3) LEFT    #
    #####################################################################################
    red_move = {
        "pass": 0,
        "cell": np.array([3, 3]),
        "direction": 2,
        "split": 1,
    }
    blue_move = {
        "pass": 0,
        "cell": np.array([1, 3]),
        "direction": 2,
        "split": 1,
    }
    moves = {
        "red": red_move,
        "blue": blue_move,
    }
    game.step(moves)
    reference_army = np.array(
        [
            [1, 2, 0, 0],
            [0, 5, 2, 2],
            [1, 1, 0, 0],
            [0, 0, 9, 5],
        ],
        dtype=np.float32,
    )
    assert (game.channels.army == reference_army).all()

    reference_ownership_red = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership["red"] == reference_ownership_red).all()

    reference_ownership_blue = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership["blue"] == reference_ownership_blue).all()

    reference_ownership_neutral = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership_neutral == reference_ownership_neutral).all()

    reference_total_army_red = 21
    stats = game.get_infos()
    assert stats["red"]["army"] == reference_total_army_red

    reference_total_army_blue = 7
    assert stats["blue"]["army"] == reference_total_army_blue

    reference_total_army_land = 6
    assert stats["red"]["land"] == reference_total_army_land

    reference_total_army_land = 4
    assert stats["blue"]["land"] == reference_total_army_land

    # Test passing a move
    ##############################################################
    # Red moves army from (3, 2) UP and blue is passing the move #
    ##############################################################
    red_move = {
        "pass": 0,
        "cell": np.array([3, 2]),
        "direction": 0,
        "split": 0,
    }
    blue_move = {
        "pass": 1,
        "cell": np.array([1, 3]),
        "direction": 2,
        "split": 0,
    }
    moves = {
        "red": red_move,
        "blue": blue_move,
    }

    game.step(moves)
    reference_army = np.array(
        [
            [1, 2, 0, 0],
            [0, 5, 2, 3],
            [1, 1, 8, 0],
            [0, 0, 1, 6],
        ],
        dtype=np.float32,
    )
    assert (game.channels.army == reference_army).all()

    reference_ownership_red = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership["red"] == reference_ownership_red).all()

    reference_ownership_blue = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership["blue"] == reference_ownership_blue).all()

    reference_ownership_neutral = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership_neutral == reference_ownership_neutral).all()

    reference_total_army_red = 22
    stats = game.get_infos()
    assert stats["red"]["army"] == reference_total_army_red

    reference_total_army_blue = 8
    assert stats["blue"]["army"] == reference_total_army_blue

    reference_total_army_land = 6
    assert stats["red"]["land"] == reference_total_army_land

    reference_total_army_land = 4
    assert stats["blue"]["land"] == reference_total_army_land

    # Test order of moves (smaller army has priority)
    #############################################################
    # Red moves from (2, 2) UP and blue moves from (1, 2) RIGHT #
    #############################################################
    red_move = {
        "pass": 0,
        "cell": np.array([2, 2]),
        "direction": 0,
        "split": 0,
    }
    blue_move = {
        "pass": 0,
        "cell": np.array([1, 2]),
        "direction": 3,
        "split": 0,
    }
    moves = {
        "red": red_move,
        "blue": blue_move,
    }

    game.step(moves)
    reference_army = np.array(
        [
            [1, 2, 0, 0],
            [0, 5, 6, 4],
            [1, 1, 1, 0],
            [0, 0, 1, 6],
        ],
        dtype=np.float32,
    )
    assert (game.channels.army == reference_army).all()

    reference_ownership_red = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership["red"] == reference_ownership_red).all()

    reference_ownership_blue = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership["blue"] == reference_ownership_blue).all()

    reference_ownership_neutral = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    assert (game.channels.ownership_neutral == reference_ownership_neutral).all()

    reference_total_army_red = 21
    stats = game.get_infos()
    assert stats["red"]["army"] == reference_total_army_red

    reference_total_army_blue = 7
    assert stats["blue"]["army"] == reference_total_army_blue

    reference_total_army_land = 7
    assert stats["red"]["land"] == reference_total_army_land

    reference_total_army_land = 3
    assert stats["blue"]["land"] == reference_total_army_land

    ##############################
    # Test global army increment #
    ##############################
    game.time = 50
    game._global_game_update()
    reference_army = np.array(
        [
            [2, 3, 0, 0],
            [0, 6, 7, 6],
            [2, 2, 2, 0],
            [0, 0, 2, 8],
        ],
        dtype=np.float32,
    )
    assert (game.channels.army == reference_army).all()

    reference_total_army_red = 29
    stats = game.get_infos()
    assert stats["red"]["army"] == reference_total_army_red

    reference_total_army_blue = 11
    assert stats["blue"]["army"] == reference_total_army_blue

    reference_total_army_land = 7
    assert stats["red"]["land"] == reference_total_army_land

    reference_total_army_land = 3
    assert stats["blue"]["land"] == reference_total_army_land


# def test_end_of_game():
#     map = """...#
# #..A
# #..#
# .#.B
# """
#     game = get_game(map)
#     game.general_positions = {"red": [3, 3], "blue": [1, 3]}
#     game.channels.ownership["red"] = np.array(
#         [
#             [0, 0, 0, 0],
#             [0, 0, 1, 0],
#             [1, 1, 1, 0],
#             [0, 0, 1, 1],
#         ],
#         dtype=np.float32,
#     )
#
#     game.channels.ownership["blue"] = np.array(
#         [
#             [1, 1, 1, 0],
#             [0, 1, 0, 1],
#             [0, 0, 0, 0],
#             [0, 0, 0, 0],
#         ],
#         dtype=np.float32,
#     )
#
#     game.channels.army = np.array(
#         [
#             [3, 2, 2, 0],
#             [0, 3, 6, 2],
#             [1, 9, 5, 0],
#             [0, 0, 5, 8],
#         ],
#         dtype=np.float32,
#     )
#
#     game.channels.ownership_neutral = np.array(
#         [
#             [0, 0, 0, 0],
#             [0, 0, 0, 0],
#             [0, 0, 0, 0],
#             [1, 0, 0, 0],
#         ],
#         dtype=np.float32,
#     )
#
#     moves = {"red": (0, np.array([3, 2]), 0, 0), "blue": (0, np.array([1, 3]), 2, 1)}
#     game.step(moves)
#
#     # Neither should win
#     assert not game.agent_won("red")
#     assert not game.agent_won("blue")
#     assert not game.is_done()
#
#     # Red moves to blues general, blue makes random move
#     moves = {"red": (0, np.array([2, 2]), 0, 0), "blue": (0, np.array([1, 3]), 3, 0)}
#     game.step(moves)
#
#     # Red should win
#     assert game.agent_won("red")
#
#     # Blue should be dead
#     assert not game.agent_won("blue")
#
#     # Game should be done
#     assert game.is_done()
