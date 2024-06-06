import pygame
import cairosvg
import io
import importlib.resources
from . import game_config
from . import grid


def load_image(filename):
    if filename.endswith(".svg"):
        with importlib.resources.path("generals.images", filename) as path:
            png_data = cairosvg.svg2png(url=str(path))
            return pygame.image.load(io.BytesIO(png_data), "png")
    elif filename.endswith(".png"):
        with importlib.resources.path("generals.images", filename) as path:
            return pygame.image.load(str(path), "png")
    else:
        raise ValueError("Unsupported image format")
        

class Visualizer():
    square_size = 40
    visual_offset = 5
    colors = {
        "fog_of_war": (80, 83, 86),
    }

    def __init__(self, config: game_config.GameConfig, grid: grid.Grid):
        pygame.init()
        self.grid_size = config.grid_size
        self.window_size = self.square_size * config.grid_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        
        self.screen.fill(color=self.colors["fog_of_war"])


        def position_offset(x, y):
            return (
                x * self.square_size + self.visual_offset,
                y * self.square_size + self.visual_offset
            )

        terrain_indices = grid.get_channel("terrain")
        for i, j in terrain_indices:
            self.screen.blit(load_image("terrain.svg"), position_offset(i, j))

        towns_indices = grid.get_channel("castle")
        for i, j in towns_indices:
            self.screen.blit(load_image("castle.png"), position_offset(i, j))

        for i in range(1, self.grid_size):
            pygame.draw.line(self.screen, "black", (self.square_size*i, 0), (self.square_size*i, self.window_size), 2)
            pygame.draw.line(self.screen, "black", (0, self.square_size*i), (self.window_size, self.square_size*i), 2)

        pygame.display.flip()


    def draw_grid(self, grid):
        pass

