import numpy as np
import pygame
import time
from . import game
from . import constants as c

from typing import Tuple, List



class Renderer():
    def __init__(self, agents: List[str], grid_size: int):
        """
        Initialize pygame window

        Args:
            config: game config object
        """
        pygame.init()
        pygame.display.set_caption("Generals")

        global screen, clock, _grid_size, window_width, window_height, grid_offset
        _grid_size = grid_size
        grid_offset = c.UI_ROW_HEIGHT * (len(agents) + 1)
        # TODO: remove constants
        window_width = max(400, c.SQUARE_SIZE * grid_size)
        window_height = max(400, c.SQUARE_SIZE * grid_size + grid_offset)

        screen = pygame.display.set_mode((window_width, window_height))
        clock = pygame.time.Clock()

        global agent_pov, game_speed
        agent_pov = {name: True for name in agents}
        game_speed = c.GAME_SPEED

        global MOUNTAIN_IMG, GENERAL_IMG, CITY_IMG
        MOUNTAIN_IMG = pygame.image.load(str(c.MOUNTAIN_PATH), "png")
        GENERAL_IMG = pygame.image.load(str(c.GENERAL_PATH), "png")
        CITY_IMG = pygame.image.load(str(c.CITY_PATH), "png")

        global FONT
        FONT = pygame.font.Font(c.FONT_PATH, c.FONT_SIZE)


    def render(self, game: game.Game):
        self.handle_events(game)
        self.render_grid(game)
        self.render_gui(game)
        pygame.display.flip()
        time.sleep(game_speed)


    def handle_events(self, game: game.Game):
        """
        Handle pygame events
        """
        agents = game.agents
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                global game_speed
                # quit game if q is pressed
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()
                # speed up game right arrow is pressed
                if event.key == pygame.K_RIGHT:
                    game_speed = max(1/2**12, game_speed / 2)
                # slow down game left arrow is pressed
                if event.key == pygame.K_LEFT:
                    game_speed = min(1, game_speed * 2)
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                for i, agent in enumerate(agents):
                    if y < c.UI_ROW_HEIGHT * (i + 2) and y > c.UI_ROW_HEIGHT * (i+1):
                        agent_pov[agent] = not agent_pov[agent]


    def render_gui(self, game: game.Game):
        names = game.agents
        ids = [game.agent_id[name] for name in names]
        army_counts = ["Army"] + [str(game.player_stats[name]["army"]) for name in names]
        land_counts = ["Land"] + [str(game.player_stats[name]["land"]) for name in names]
        povs = ["POV"] + ["  X" if agent_pov[name] else " " for name in names]
        names = ["Turn"] + [name for name in names]

        # white background for GUI
        pygame.draw.rect(screen, c.WHITE, (0, 0, window_width, grid_offset))

        # draw rows with player colors
        for i, agent_id in enumerate(ids):
            pygame.draw.rect(
                screen,
                c.PLAYER_COLORS[agent_id],
                (
                    0,
                    (i + 1) * c.UI_ROW_HEIGHT,
                    window_width - 3 * c.GUI_CELL_WIDTH,
                    c.UI_ROW_HEIGHT,
                ),
            )

        # draw lines between rows
        for i in range(1, 3):
            pygame.draw.line(
                screen,
                c.BLACK,
                (0, i * c.UI_ROW_HEIGHT),
                (window_width, i * c.UI_ROW_HEIGHT),
                c.LINE_WIDTH,
            )

        # draw vertical lines cell_width from the end and 2*cell_width from the end
        for i in range(1, 4):
            pygame.draw.line(
                screen,
                c.BLACK,
                (window_width - i * c.GUI_CELL_WIDTH, 0),
                (window_width - i * c.GUI_CELL_WIDTH, grid_offset),
                c.LINE_WIDTH,
            )

        # write live statistics
        speed = "Paused" if game_speed == 0 else str(1/(2*game_speed)) + "x"
        text = FONT.render(f"Game speed: {speed}", True, c.BLACK)
        screen.blit(text, (150, 15))
        for i in range(len(ids) + 1):
            if i == 0:
                turn = str(game.time // 2) + ("." if game.time % 2 == 1 else "")
                text = FONT.render(f"{names[i]}: {turn}", True, c.BLACK)
            else:
                text = FONT.render(f"{names[i]}", True, c.BLACK)
            top_offset = i * c.UI_ROW_HEIGHT + 15
            screen.blit(text, (10, top_offset))
            text = FONT.render(army_counts[i], True, c.BLACK)
            screen.blit(text, (window_width - 2 * c.GUI_CELL_WIDTH + 25, top_offset))
            text = FONT.render(land_counts[i], True, c.BLACK)
            screen.blit(text, (window_width - c.GUI_CELL_WIDTH + 25, top_offset))
            text = FONT.render(povs[i], True, c.BLACK)
            screen.blit(text, (window_width - 3 * c.GUI_CELL_WIDTH + 25, top_offset))


    def render_grid(self, game: game.Game):
        """
        Render grid part of the game.

        Args:
            game: Game object
            agent_ids: list of agent ids from which perspective the game is rendered
        """
        agents = game.agents
        # draw visibility squares
        visible_map = np.zeros((_grid_size, _grid_size), dtype=np.float32)
        for agent in agents:
            if not agent_pov[agent]:
                continue
            ownership = game.channels["ownership_" + agent]
            visibility = game.visibility_channel(ownership)
            visibility_indices = game.channel_to_indices(visibility)
            self.draw_channel(visibility_indices, c.WHITE)
            visible_map = np.logical_or(visible_map, visibility)  # get all visible cells

        # draw ownership squares
        for agent in agents:
            ownership = game.channels["ownership_" + agent]
            ownership_indices = game.channel_to_indices(ownership)
            self.draw_channel(ownership_indices, c.PLAYER_COLORS[game.agent_id[agent]])

        # draw lines
        for i in range(1, _grid_size):
            pygame.draw.line(
                screen,
                c.BLACK,
                (0, i * c.SQUARE_SIZE + grid_offset),
                (window_width, i * c.SQUARE_SIZE + grid_offset),
                c.LINE_WIDTH,
            )
            pygame.draw.line(
                screen,
                c.BLACK,
                (i * c.SQUARE_SIZE, grid_offset),
                (i * c.SQUARE_SIZE, window_height),
                c.LINE_WIDTH,
            )

        # # draw background as squares
        invisible_map = np.logical_not(visible_map)
        invisible_indices = game.channel_to_indices(invisible_map)
        self.draw_channel(invisible_indices, c.FOG_OF_WAR)

        # draw mountain
        mountain_indices = game.channel_to_indices(game.channels["mountain"])
        visible_mountain = np.logical_and(game.channels["mountain"], visible_map)
        visible_mountain_indices = game.channel_to_indices(visible_mountain)
        self.draw_channel(visible_mountain_indices, c.VISIBLE_MOUNTAIN)
        self.draw_images(mountain_indices, MOUNTAIN_IMG)

        # draw general
        for agent, pov in agent_pov.items():
            if pov:
                agent_general = np.logical_and(game.channels[f'ownership_{agent}'], game.channels["general"])
                general_indices = game.channel_to_indices(agent_general)
                self.draw_images(general_indices, GENERAL_IMG)

        # draw neutral visible city color
        visible_cities = np.logical_and(game.channels["city"], visible_map)
        visible_cities_neutral = np.logical_and(
            visible_cities, game.channels["ownership_neutral"]
        )
        visible_cities_neutral_indices = game.channel_to_indices(visible_cities_neutral)
        self.draw_channel(visible_cities_neutral_indices, c.NEUTRAL_CASTLE)

        # draw visible city images
        visible_cities_indices = game.channel_to_indices(visible_cities)
        self.draw_images(visible_cities_indices, CITY_IMG)

        # draw army counts on visibility mask
        army = game.channels["army"] * visible_map
        visible_army_indices = game.channel_to_indices(army)
        y_offset = 15
        for i, j in visible_army_indices:
            text = FONT.render(str(int(army[i, j])), True, c.WHITE)
            x_offset = c.FONT_OFFSETS[
                min(len(c.FONT_OFFSETS) - 1, len(str(int(army[i, j]))) - 1)
            ]
            screen.blit(
                text,
                (
                    j * c.SQUARE_SIZE + x_offset,
                    i * c.SQUARE_SIZE + y_offset + grid_offset,
                ),
            )


    def draw_channel(self, channel: list[Tuple[int, int]], color: Tuple[int, int, int]):
        """
        Draw channel squares on the screen

        Args:
            channel: list of tuples with indices of the channel
            color: color of the squares
        """
        size, offset = c.SQUARE_SIZE, grid_offset
        w = c.LINE_WIDTH
        for i, j in channel:
            pygame.draw.rect(
                screen,
                color,
                (j * size + w, i * size + w + offset, size - w, size - w),
            )


    def draw_images(self, channel: list[Tuple[int, int]], image):
        """
        Draw images on the screen

        Args:
            screen: pygame screen object
            channel: list of tuples with indices of the channel
            image: pygame image object
        """
        size, offset = c.SQUARE_SIZE, grid_offset
        for i, j in channel:
            screen.blit(image, (j * size + 3, i * size + 3 + offset))
