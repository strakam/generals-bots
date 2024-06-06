

import numpy as np
from . import game_config


class Grid():
    def __init__(self, config: game_config.GameConfig):
        self.config = config
        self.n_channels = 5
        self.grid = self.generate()


    def generate(self):
        map = np.zeros((self.config.grid_size, self.config.grid_size, self.n_channels), dtype=np.float32)
        map[self.config.starting_positions[0][0], self.config.starting_positions[0][1], 0] = 1
        map[self.config.starting_positions[1][0], self.config.starting_positions[1][1], 0] = 1
        return map


