

import numpy as np
from grid_config import GridConfig


class Grid():
    def __init__(self, config: GridConfig):
        self.config = config
        self.n_channels = 5
        self.grid = self.generate()
        print(self.grid[..., 0])


    def generate(self):
        map = np.zeros((self.config.grid_size, self.config.grid_size, self.n_channels), dtype=np.float32)
        map[self.config.starting_positions[0][0], self.config.starting_positions[0][1], 0] = 1
        map[self.config.starting_positions[1][0], self.config.starting_positions[1][1], 0] = 1
        return map


config = GridConfig(
    grid_size=10,
    starting_positions=[[1, 1], [5, 5]]
)

g = Grid(config)

