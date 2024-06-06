from typing import List, Tuple

import numpy as np
from . import game_config


class Grid():
    TERRAIN = 0
    TOWN = 1
    PLAIN = 2
    GENERAL = 3
    ARMY = 4
    OWNERSHIP = 5

    CHANNEL_NAMES = [
        'terrain',
        'town',
        'plain',
        'general',
        'army',
        'ownership'
    ]

    def __init__(self, config: game_config.GameConfig):
        self.config = config

        # generate layout of the grid
        p_plain = 1 - self.config.terrain_density - self.config.town_density
        probs = [self.config.terrain_density, self.config.town_density, p_plain]
        map = np.random.choice([0, 1, 2], size=(self.config.grid_size, self.config.grid_size), p=probs)

        for i, channel_name in enumerate(self.CHANNEL_NAMES[:3]):
            setattr(self, channel_name, np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32))
            getattr(self, channel_name)[map == i] = 1

        # cells with general in them have value 1, otherwise 0
        self.general = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32)
        for general in self.config.starting_positions:
            self.general[general[0], general[1]] = 1

        # if ownership_channel[i, j] == k, then cell (i, j) is owned by player k
        self.ownership = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32)
        
        # values of army size in each cell
        self.army = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32)

    def get_channel(self, channel_name) -> List[Tuple[int, int]]:
        """
        Returns a list of indices of cells from specified channel among [terrain, town, plain]
        """
        return np.argwhere(getattr(self, channel_name) == 1)




