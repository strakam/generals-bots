import numpy as np
import pygame
import importlib.resources
from . import game
from . import config as conf

from typing import Tuple, List, Dict


def init_screen(game_config: conf.Config):
    """
    Initialize pygame window

    Args:
        config: game config object
    """
    pygame.init()
    pygame.display.set_caption("Generals")

    global screen, clock, config
    config = game_config
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    global MOUNTAIN_IMG, CITY_IMG, GENERAL_IMG, font, font_offsets
    MOUNTAIN_IMG = load_image("mountainie.png")
    CITY_IMG = load_image("citie.png")
    GENERAL_IMG = load_image("crownie.png")
    font_offsets = [20, 16, 12, 8, 4] # for 0 digits, 1 digit, 2 digits,..
    try:
        # Options for font are Quicksand-Regular.ttf, Quicksand-SemiBold.ttf, Quicksand-Medium.ttf, Quicksand-Light.ttf
        with importlib.resources.path("generals.fonts", "Quicksand-Medium.ttf") as path:
            font = pygame.font.Font(str(path), 18)

    except FileNotFoundError:
        raise ValueError("Font not found")


def load_image(filename):
    """
    Load image from the resources (only png files are supported)

    Args:
        filename: name of the file
    """
    try:
        with importlib.resources.path("generals.images", filename) as path:
            return pygame.image.load(str(path), "png")
    except FileNotFoundError:
        raise ValueError("Image not found")


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

def render_gui(game: game.Game, agent_id_to_name: Dict[int, str]):
    ids = agent_id_to_name.keys()
    names = ["Player"] + [str(agent_id_to_name[id]) for id in ids]
    army_counts = ["Army"] + [str(game.player_stats[id]['army']) for id in ids]
    land_counts = ["Land"] + [str(game.player_stats[id]['land']) for id in ids]

    # white background
    pygame.draw.rect(screen, config.WHITE, (0, 0, config.WINDOW_WIDTH, config.GRID_OFFSET))

    # draw rows with player colors
    cell_width = 100
    for i, agent_id in enumerate(ids):
        pygame.draw.rect(
            screen,
            config.PLAYER_COLORS[agent_id],
            (0, (i+1) * 50, config.WINDOW_WIDTH-2*cell_width, 50)
        )

    # draw lines between rows
    for i in range(1, 3):
        pygame.draw.line(
            screen,
            config.BLACK,
            (0, i * 50),
            (config.WINDOW_WIDTH, i * 50),
            2
        )

    # draw vertical lines cell_width from the end and 2*cell_width from the end
    # make them tall as config.GRID_OFFSET
    for i in range(1, 3):
        pygame.draw.line(
            screen,
            config.BLACK,
            (config.WINDOW_WIDTH - i*cell_width, 0),
            (config.WINDOW_WIDTH - i*cell_width, config.GRID_OFFSET),
            2
        )

    # draw 
    for i in range(len(ids)+1):
        text = font.render(names[i], True, config.BLACK)
        screen.blit(text, (10, i*50 + 15))
        text = font.render(army_counts[i], True, config.BLACK)
        screen.blit(text, (config.WINDOW_WIDTH - 2*cell_width + 25, i*50 + 15))
        text = font.render(land_counts[i], True, config.BLACK)
        screen.blit(text, (config.WINDOW_WIDTH - cell_width + 25, i*50 + 15))


def render_grid(game: game.Game, agent_ids: list[int]):
    """
    Render grid part of the game.

    Args:
        game: Game object
        agent_ids: list of agent ids from which perspective the game is rendered
    """
    # draw visibility squares
    visible_map = np.zeros((config.grid_size, config.grid_size), dtype=np.float32)
    for id in agent_ids:
        ownership = game.channels['ownership_' + str(id)]
        visibility = game.visibility_channel(ownership)
        visibility_indices = game.channel_to_indices(visibility)
        draw_channel(visibility_indices, config.WHITE)
        visible_map = np.logical_or(visible_map, visibility) # get all visible cells

    # draw ownership squares
    for id in agent_ids:
        ownership = game.channels['ownership_' + str(id)]
        ownership_indices = game.channel_to_indices(ownership)
        draw_channel(ownership_indices, config.PLAYER_COLORS[id])

    # draw lines
    for i in range(1, config.grid_size):
        pygame.draw.line(
            screen,
            config.BLACK,
            (0, i * config.SQUARE_SIZE + config.GRID_OFFSET),
            (config.WINDOW_WIDTH, i * config.SQUARE_SIZE + config.GRID_OFFSET),
            2
        )
        pygame.draw.line(
            screen,
            config.BLACK,
            (i * config.SQUARE_SIZE, config.GRID_OFFSET),
            (i * config.SQUARE_SIZE, config.WINDOW_HEIGHT),
            2
        )

    # # draw background as squares
    invisible_map = np.logical_not(visible_map)
    invisible_indices = game.channel_to_indices(invisible_map)
    draw_channel(invisible_indices, config.FOG_OF_WAR)

    # draw mountain
    mountain_indices = game.channel_to_indices(game.channels['mountain'])
    visible_mountain = np.logical_and(game.channels['mountain'], visible_map)
    visible_mountain_indices = game.channel_to_indices(visible_mountain)
    draw_channel(visible_mountain_indices, config.VISIBLE_MOUNTAIN)
    draw_images(mountain_indices, MOUNTAIN_IMG)


    # draw general
    general_indices = game.channel_to_indices(game.channels['general'])
    draw_images(general_indices, GENERAL_IMG)

    # draw neutral visible city color
    visible_cities = np.logical_and(game.channels['city'], visible_map)
    visible_cities_neutral = np.logical_and(visible_cities, game.channels['ownership_0'])
    visible_cities_neutral_indices = game.channel_to_indices(visible_cities_neutral)
    draw_channel(visible_cities_neutral_indices, config.NEUTRAL_CASTLE)

    # draw visible city images
    visible_cities_indices = game.channel_to_indices(visible_cities)
    draw_images(visible_cities_indices, CITY_IMG)

    # draw army counts on visibility mask
    army = game.channels['army'] * visible_map
    visible_army_indices = game.channel_to_indices(army)
    y_offset = 15
    for i, j in visible_army_indices:
        text = font.render(str(int(army[i, j])), True, config.WHITE)
        x_offset = font_offsets[min(len(font_offsets)-1, len(str(int(army[i, j])))-1)]
        screen.blit(text, (j * config.SQUARE_SIZE + x_offset, i * config.SQUARE_SIZE + y_offset + config.GRID_OFFSET))


def draw_channel(channel: list[Tuple[int, int]], color: Tuple[int, int, int]):
    """
    Draw channel squares on the screen

    Args:
        channel: list of tuples with indices of the channel
        color: color of the squares
    """
    size, offset = config.SQUARE_SIZE, config.GRID_OFFSET
    for i, j in channel:
        pygame.draw.rect(
            screen,
            color,
            (j * size + 2, i * size + 2 + offset, size - 2, size - 2)
        )

def draw_images(channel: list[Tuple[int, int]], image):
    """
    Draw images on the screen

    Args:
        screen: pygame screen object
        channel: list of tuples with indices of the channel
        image: pygame image object
    """
    size, offset = config.SQUARE_SIZE, config.GRID_OFFSET
    for i, j in channel:
        screen.blit(image, (j * size + 3, i * size + 3 + offset))
