import pygame
import cairosvg
import io
import importlib.resources
from . import game_config
from . import grid


def load_image(filename):
    with importlib.resources.path("generals.images", filename) as path:
        png_data = cairosvg.svg2png(url=str(path))
        return pygame.image.load(io.BytesIO(png_data), "png")

class Visualizer():
    square_size = 40
    colors = {
        "fog_of_war": (88, 87, 92),
    }

    def __init__(self, config: game_config.GameConfig, grid: grid.Grid):
        pygame.init()
        self.grid_size = config.grid_size
        self.window_size = self.square_size * config.grid_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        
        self.terrain_image = load_image("resized.svg")

        self.screen.fill(color=self.colors["fog_of_war"])
        terrain_indices = grid.get_channel("terrain")
        for i, j in terrain_indices:
            self.screen.blit(self.terrain_image, (i*self.square_size, j*self.square_size+2))



    def draw_grid(self, grid):
        for i in range(1, self.grid_size):
            pygame.draw.line(self.screen, "black", (self.square_size*i, 0), (self.square_size*i, self.window_size), 2)
            pygame.draw.line(self.screen, "black", (0, self.square_size*i), (self.window_size, self.square_size*i), 2)

        pygame.display.flip()

