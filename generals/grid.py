from typing import List, Tuple

import numpy as np
from . import game_config

ROAD = 0
TERRAIN = 1
TOWN = 2
GENERAL = 3
ARMY = 4
OWNERSHIP = 5

class Grid():
    def __init__(self, config: game_config.GameConfig):
        self.config = config
        print(config)
        # generate layout of the grid
        p_plain = 1 - self.config.terrain_density - self.config.town_density
        probs = [p_plain, self.config.terrain_density, self.config.town_density]
        map = np.random.choice([ROAD, TERRAIN, TOWN], size=(self.config.grid_size, self.config.grid_size), p=probs)

        # place generals
        for general in self.config.starting_positions:
            map[general[0], general[1]] = 0 # general square is considered passable, so we give it 0

        for i, channel_name in enumerate(['road', 'terrain', 'castle']):
            setattr(self, channel_name, np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32))
            getattr(self, channel_name)[map == i] = 1

        # cells with general in them have value 1, otherwise 0
        self.general = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32)
        for general in self.config.starting_positions:
            self.general[general[0], general[1]] = 1

        # if ownership_channel[i, j] == 1, then player owns cell (i, j)
        # only for 1v1 now
        self.ownership = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32)
        
        # values of army size in each cell
        self.army = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.float32)

    def get_channel(self, channel_name) -> List[Tuple[int, int]]:
        """
        Returns a list of indices of cells from specified channel among [terrain, castle, road, general, ownership]
        Ownership is supported only for 1v1 now
        """
        return np.argwhere(getattr(self, channel_name) == 1)




