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

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()

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

    Args:
        filename: name of the file (without extension)
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
    """
    Handle pygame events
    """
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
    
    Args:
        game: Game object
        agent_ids: list of agent ids
    """

    handle_events()
    render_grid(game, agent_ids)
    
    pygame.display.flip()


def render_grid(game: game.Game, agent_ids: list[int]):
    """
    Render grid part of the game.

    Args:
        game: Game object
        agent_ids: list of agent ids
    """
    # draw visibility squares
    visible_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for id in agent_ids:
        ownership = game.channels['ownership_' + str(id)]
        visibility = game.visibility_channel(ownership)
        visibility_indices = game.channel_to_indices(visibility)
        draw_channel(visibility_indices, COLORS["visible"])
        visible_map = np.logical_or(visible_map, visibility) # get all visible cells

    # draw ownership squares
    for id in agent_ids:
        ownership = game.channels['ownership_' + str(id)]
        ownership_indices = game.channel_to_indices(ownership)
        draw_channel(ownership_indices, COLORS["player_" + str(id)])

    # draw lines
    for i in range(1, GRID_SIZE):
        pygame.draw.line(
            screen,
            COLORS["black"],
            (0, i * SQUARE_SIZE + GRID_OFFSET),
            (WINDOW_WIDTH, i * SQUARE_SIZE + GRID_OFFSET),
            2
        )
        pygame.draw.line(
            screen,
            COLORS["black"],
            (i * SQUARE_SIZE, GRID_OFFSET),
            (i * SQUARE_SIZE, WINDOW_HEIGHT),
            2
        )

    # # draw background as squares
    invisible_map = np.logical_not(visible_map)
    invisible_indices = game.channel_to_indices(invisible_map)
    draw_channel(invisible_indices, COLORS["fog_of_war"])

    # draw mountain
    mountain_indices = game.channel_to_indices(game.channels['mountain'])
    draw_images(mountain_indices, MOUNTAIN_IMG)

    # draw general
    general_indices = game.channel_to_indices(game.channels['general'])
    draw_images(general_indices, GENERAL_IMG)

    # draw cities
    visible_cities = np.logical_and(game.channels['city'], visible_map)
    visible_cities_indices = game.channel_to_indices(visible_cities)
    draw_images(visible_cities_indices, CITY_IMG)

    # draw army counts on visibility mask
    army = game.channels['army'] * visible_map
    font = pygame.font.Font(None, 20)
    for agent_id in agent_ids:
        ownership_channel = game.channels['ownership_' + str(agent_id)]
        ownership_indices = game.channel_to_indices(ownership_channel)
        for i, j in ownership_indices:
            text = font.render(str(int(army[i, j])), True, COLORS["white"])
            screen.blit(text, (j * SQUARE_SIZE + 12, i * SQUARE_SIZE + 15 + GRID_OFFSET))


def draw_channel(channel: list[Tuple[int, int]], color: Tuple[int, int, int]):
    """
    Draw channel squares on the screen

    Args:
        channel: list of tuples with indices of the channel
        color: color of the squares
    """
    for i, j in channel:
        pygame.draw.rect(
            screen,
            color,
            (j * SQUARE_SIZE, i * SQUARE_SIZE + GRID_OFFSET, SQUARE_SIZE, SQUARE_SIZE)
        )

def draw_images(channel: list[Tuple[int, int]], image):
    """
    Draw images on the screen

    Args:
        screen: pygame screen object
        channel: list of tuples with indices of the channel
        image: pygame image object
    """
    for i, j in channel:
        screen.blit(image, (j * SQUARE_SIZE + 3, i * SQUARE_SIZE + 2 + GRID_OFFSET))
