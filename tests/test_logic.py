import pytest
import numpy as np

import generals.game_config as game_config
import generals.game

config = game_config.GameConfig(
    grid_size=10,
    starting_positions=[[1, 1], [5, 5]]
)

dummy_game = generals.game.Game(config)


def test_channel_to_indices():
    channel = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])
    reference = np.array([(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)])
    indices = dummy_game.channel_to_indices(channel)
    assert (indices == reference).all()

    channel = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    reference = np.empty((0, 2))
    indices = dummy_game.channel_to_indices(channel)
    assert (indices == reference).all()

def test_visibility_channel():
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
