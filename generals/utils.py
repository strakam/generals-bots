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

GRID_OFFSET = 50
WINDOW_HEIGHT = SQUARE_SIZE * GRID_SIZE + GRID_OFFSET
WINDOW_WIDTH = SQUARE_SIZE * GRID_SIZE

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

def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN:
            # quit game if q is pressed
            if event.key == pygame.K_q:
                pygame.quit()
                quit()

def render(game: game.Game, agent_ids: list[int]):
    """
    Method that orchestrates rendering of the game
    """

    handle_events()
    render_grid(game, agent_ids)
    
    pygame.display.flip()


def render_grid(game: game.Game, agent_ids: list[int]):
    # draw visibility squares
    visible_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for id in agent_ids:
        visibility = game.visibility_channel(id)
        visibility_indices = game.channel_as_list(visibility)
        draw_channel(game.screen, visibility_indices, COLORS["visible"])
        visible_map = np.logical_or(visible_map, visibility) # get all visible cells

    # draw ownership squares
    for id in agent_ids:
        ownership = game.channels['ownership_' + str(id)]
        ownership_indices = game.channel_as_list(ownership)
        draw_channel(game.screen, ownership_indices, COLORS["player_" + str(id)])

    # draw lines
    for i in range(1, GRID_SIZE):
        pygame.draw.line(
            game.screen,
            COLORS["black"],
            (0, i * SQUARE_SIZE + GRID_OFFSET),
            (WINDOW_WIDTH, i * SQUARE_SIZE + GRID_OFFSET),
            2
        )
        pygame.draw.line(
            game.screen,
            COLORS["black"],
            (i * SQUARE_SIZE, GRID_OFFSET),
            (i * SQUARE_SIZE, WINDOW_HEIGHT),
            2
        )

    # # draw background as squares
    invisible_map = np.logical_not(visible_map)
    invisible_indices = game.channel_as_list(invisible_map)
    draw_channel(game.screen, invisible_indices, COLORS["fog_of_war"])

    # draw mountain
    mountain_indices = game.channel_as_list(game.channels['mountain'])
    draw_images(game.screen, mountain_indices, MOUNTAIN_IMG)

    # draw general
    general_indices = game.channel_as_list(game.channels['general'])
    draw_images(game.screen, general_indices, GENERAL_IMG)

    # draw cities
    visible_cities = np.logical_and(game.channels['city'], visible_map)
    visible_cities_indices = game.channel_as_list(visible_cities)
    draw_images(game.screen, visible_cities_indices, CITY_IMG)

    # draw army counts on visibility mask
    army = game.channels['army'] * visible_map
    font = pygame.font.Font(None, 20)
    for i, j in game.channel_as_list(army):
        # font color white
        text = font.render(str(int(army[i, j])), True, COLORS["white"])
        game.screen.blit(text, (j * SQUARE_SIZE + 12, i * SQUARE_SIZE + 15 + GRID_OFFSET))


def draw_channel(screen, channel: list[Tuple[int, int]], color: Tuple[int, int, int]):
    """draw channel colors"""
    for i, j in channel:
        pygame.draw.rect(
            screen,
            color,
            (j * SQUARE_SIZE, i * SQUARE_SIZE + GRID_OFFSET, SQUARE_SIZE, SQUARE_SIZE)
        )

def draw_images(screen, channel: list[Tuple[int, int]], image):
    """draw channel images"""
    for i, j in channel:
        screen.blit(image, (j * SQUARE_SIZE + 3, i * SQUARE_SIZE + 2 + GRID_OFFSET))
