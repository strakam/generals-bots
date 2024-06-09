import numpy as np

import generals.config as conf
import generals.game

def get_game():
    config = conf.Config(
        grid_size=10,
        starting_positions=[[1, 1], [5, 5]]
    )
    return generals.game.Game(config)

def test_channel_to_indices():
    """
    For given channel, we should get indices of cells that are 1.
    """
    game = get_game()

    channel = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])
    reference = np.array([(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)])
    indices = game.channel_to_indices(channel)
    assert (indices == reference).all()

    channel = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    reference = np.empty((0, 2))
    indices = game.channel_to_indices(channel)
    assert (indices == reference).all()

def test_visibility_channel():
    """
    For given ownership mask, we should get visibility mask.
    """
    dummy_game = get_game()

    ownership = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    reference = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    visibility = dummy_game.visibility_channel(ownership)
    assert (visibility == reference).all()

    ownership = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1]
    ])
    reference = np.array([
        [1, 1, 1, 0 ,0],
        [1, 1, 1, 0 ,0],
        [1, 1, 1, 1 ,1],
        [1, 1, 1, 1 ,1],
        [0, 0, 0, 1 ,1]
    ])
    visibility = dummy_game.visibility_channel(ownership)
    assert (visibility == reference).all()

def test_valid_actions():
    """
    For given ownership mask and passable mask, we should get NxNx4 mask of valid actions.
    """
    game = get_game()
    game.grid_size = 4
    game.channels['passable'] = np.array([
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 0],
    ], dtype=np.float32)

    game.channels['ownership_1'] = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ], dtype=np.float32)
    reference = np.array([
        # UP
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        # DOWN
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        # LEFT
        [
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        # RIGHT
        [
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    ], dtype=np.float32)

    valid_actions = game.valid_actions(game.channels['ownership_1'])
    assert (valid_actions[:, :, 0] == reference[0]).all()
    assert (valid_actions[:, :, 1] == reference[1]).all()
    assert (valid_actions[:, :, 2] == reference[2]).all()
    assert (valid_actions[:, :, 3] == reference[3]).all()
        



