import pygame
from . import grid
from . import game_config
from generals import utils

class Game():
    def __init__(self, config: game_config.GameConfig):
        pygame.init()
        self.grid_size = config.grid_size
        self.screen = pygame.display.set_mode((utils.WINDOW_SIZE, utils.WINDOW_SIZE))
        self.clock = pygame.time.Clock()

        self.grid_config = config
        self.grid = grid.Grid(self.grid_config)

    def render(self):
        utils.draw_static_parts(self.screen, self.grid)
        pygame.display.flip()

