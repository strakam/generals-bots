

from . import grid
from . import game_config


class Game():
    def __init__(self, game_config: game_config.GameConfig):
        self.grid_config = game_config
        self.grid = grid.Grid(self.grid_config)
