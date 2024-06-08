import numpy as np
import pygame
import cairosvg
import io
import importlib.resources
from . import game

from typing import Dict, Tuple

GRID_SIZE = 10
SQUARE_SIZE = 40
WINDOW_SIZE = SQUARE_SIZE * GRID_SIZE
VISUAL_OFFSET = 5

COLORS = {
    "fog_of_war": (80, 83, 86),
    "visible": (200, 200, 200),
    "player_1": (255, 0, 0),
    "player_2": (0, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
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

MOUNTAIN_IMG = load_image("mountainie")
CITY_IMG = load_image("citie")
GENERAL_IMG = load_image("crownie")

# if user presses 'q', quit the game
def check_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN: # quit game if q is pressed
            if event.key == pygame.K_q:
                pygame.quit()
                quit()

def render(game: game.Game, agent_ids: list[int]):
    """
    Method that orchestrates rendering of the game
    """

    check_quit()
    
    # draw visibility squares
    visible_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for id in agent_ids:
        visibility = game.visibility_mask(id)
        draw_channel(game.screen, game.channel_as_list(visibility), COLORS["visible"])
        # logical OR with reminder
        visible_map = np.logical_or(visible_map, visibility)

    # draw ownership squares
    for id in agent_ids:
        ownership = game.channels['ownership_' + str(id)]
        draw_channel(game.screen, game.channel_as_list(ownership), COLORS["player_" + str(id)])

    # draw lines
    for i in range(1, GRID_SIZE):
        pygame.draw.line(game.screen, COLORS["black"], (SQUARE_SIZE*i, 0), (SQUARE_SIZE*i, WINDOW_SIZE), 2)
        pygame.draw.line(game.screen, COLORS["black"], (0, SQUARE_SIZE*i), (WINDOW_SIZE, SQUARE_SIZE*i), 2)

    # # draw background as squares
    draw_channel(game.screen, game.channel_as_list(np.logical_not(visible_map)), COLORS["fog_of_war"])

    # draw mountain
    draw_images(game.screen, game.channel_as_list(game.channels['mountain']), MOUNTAIN_IMG)

    # draw general
    draw_images(game.screen, game.channel_as_list(game.channels['general']), GENERAL_IMG)

    # draw cities
    draw_images(game.screen, game.channel_as_list(np.logical_and(game.channels['city'], visible_map)), CITY_IMG)

    # draw army counts on visibility mask
    army = game.channels['army'] * visible_map
    font = pygame.font.Font(None, 20)
    for i, j in game.channel_as_list(army):
        # font color white
        text = font.render(str(int(army[i, j])), True, COLORS["white"])
        game.screen.blit(text, (j * SQUARE_SIZE + 12, i * SQUARE_SIZE + 15))
    
    pygame.display.flip()

def draw_channel(screen, channel: list[Tuple[int, int]], color: Tuple[int, int, int]):
    """draw channel colors"""
    for i, j in channel:
        pygame.draw.rect(screen, color, (j * SQUARE_SIZE, i * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_images(screen, channel: list[Tuple[int, int]], image):
    """draw channel images"""
    for i, j in channel:
        screen.blit(image, (j * SQUARE_SIZE + 3, i * SQUARE_SIZE + 2))
