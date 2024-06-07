import numpy as np
import pygame
import cairosvg
import io
import importlib.resources
from . import grid

from typing import Dict, Tuple

GRID_SIZE = 10
SQUARE_SIZE = 40
WINDOW_SIZE = SQUARE_SIZE * GRID_SIZE
VISUAL_OFFSET = 5

COLORS = {
    "fog_of_war": (80, 83, 86),
    "player_1": (255, 0, 0),
    "player_2": (0, 0, 255),
}

def load_image(filename):
    """
    Check if there is filename.svg, otherwise try filename.png
    """
    # check if there is filename.svg
    if importlib.resources.is_resource("generals.images", filename + '.svg'):
        with importlib.resources.path("generals.images", filename + '.svg') as path:
            png_data = cairosvg.svg2png(url=str(path))
            return pygame.image.load(io.BytesIO(png_data), "png")
    elif importlib.resources.is_resource("generals.images", filename + '.png'):
        with importlib.resources.path("generals.images", filename + '.png') as path:
            return pygame.image.load(str(path), "png")

    raise ValueError("Unsupported image format")

def draw_static_parts(screen, channels: Dict[str, list[Tuple[int, int]]]):

    def position_offset(x, y):
        return (
            x * SQUARE_SIZE + VISUAL_OFFSET,
            y * SQUARE_SIZE + VISUAL_OFFSET
        )

    for square_type in ["general", "terrain", "castle"]:
        indices = channels[square_type]
        image = load_image(f"{square_type}")
        for i, j in indices:
            screen.blit(image, position_offset(i, j))

    for i in range(1, GRID_SIZE):
        pygame.draw.line(screen, "black", (SQUARE_SIZE*i, 0), (SQUARE_SIZE*i, WINDOW_SIZE), 2)
        pygame.draw.line(screen, "black", (0, SQUARE_SIZE*i), (WINDOW_SIZE, SQUARE_SIZE*i), 2)
        

def draw_pov(screen, channel: list[Tuple[int, int]], color: Tuple[int, int, int]):
    """
    pov is an agent ids from which we will get perspective
    """
    for i, j in channel:
        pygame.draw.rect(screen, color, (i * SQUARE_SIZE, j * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
