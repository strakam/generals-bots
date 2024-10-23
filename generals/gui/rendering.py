from typing import TypeAlias

import numpy as np
import pygame

from generals.core.config import Dimension, Path
from generals.gui.properties import GuiMode, Properties

Color: TypeAlias = tuple[int, int, int]
FOG_OF_WAR: Color = (70, 73, 76)
NEUTRAL_CASTLE: Color = (128, 128, 128)
VISIBLE_PATH: Color = (200, 200, 200)
VISIBLE_MOUNTAIN: Color = (187, 187, 187)
BLACK: Color = (0, 0, 0)
WHITE: Color = (230, 230, 230)


class Renderer:
    def __init__(self, properties: Properties):
        """
        Initialize the pygame GUI
        """
        pygame.init()
        pygame.display.set_caption("Generals")
        pygame.key.set_repeat(500, 64)

        self.properties = properties

        self.mode = self.properties.mode
        self.game = self.properties.game

        self.agent_data = self.properties.agent_data
        self.agent_fov = self.properties.agent_fov

        self.grid_height = self.properties.grid_height
        self.grid_width = self.properties.grid_width
        self.display_grid_width = self.properties.display_grid_width
        self.display_grid_height = self.properties.display_grid_height
        self.right_panel_width = self.properties.right_panel_width

        ############
        # Surfaces #
        ############
        window_width = self.display_grid_width + self.right_panel_width
        window_height = self.display_grid_height + 1

        width = Dimension.GUI_CELL_WIDTH.value
        height = Dimension.GUI_CELL_HEIGHT.value

        # Main window
        self.screen = pygame.display.set_mode((window_width, window_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        # Scoreboard
        self.right_panel = pygame.Surface((self.right_panel_width, window_height))
        self.score_cols = {}
        for col in ["Player", "Army", "Land"]:
            size = (width, height)
            if col == "Player":
                size = (2 * width, height)
            self.score_cols[col] = [pygame.Surface(size) for _ in range(3)]

        self.info_panel = {
            "time": pygame.Surface((self.right_panel_width / 2, height)),
            "speed": pygame.Surface((self.right_panel_width / 2, height)),
        }
        # Game area and tiles
        self.game_area = pygame.Surface((self.display_grid_width, self.display_grid_height))
        self.tiles = [
            [pygame.Surface((Dimension.SQUARE_SIZE.value, Dimension.SQUARE_SIZE.value)) for _ in range(self.grid_width)]
            for _ in range(self.grid_height)
        ]

        self._mountain_img = pygame.image.load(str(Path.MOUNTAIN_PATH), "png").convert_alpha()
        self._general_img = pygame.image.load(str(Path.GENERAL_PATH), "png").convert_alpha()
        self._city_img = pygame.image.load(Path.CITY_PATH, "png").convert_alpha()

        self._font = pygame.font.Font(Path.FONT_PATH, self.properties.font_size)

    def render(self, fps=None):
        self.render_grid()
        self.render_stats()
        pygame.display.flip()
        if fps:
            self.properties.clock.tick(fps)

    def render_cell_text(
        self,
        cell: pygame.Surface,
        text: str,
        fg_color: Color = BLACK,
        bg_color: Color = WHITE,
    ):
        """
        Draw a text in the middle of the cell with given foreground and background colors

        Args:
            cell: cell to draw
            text: text to write on the cell
            fg_color: foreground color of the text
            bg_color: background color of the cell
        """
        center = (cell.get_width() // 2, cell.get_height() // 2)

        text_surface = self._font.render(text, True, fg_color)
        if bg_color:
            cell.fill(bg_color)
        cell.blit(text_surface, text_surface.get_rect(center=center))

    def render_stats(self):
        """
        Draw player stats and additional info on the right panel
        """
        names = self.game.agents
        player_stats = self.game.get_infos()
        gui_cell_height = Dimension.GUI_CELL_HEIGHT.value
        gui_cell_width = Dimension.GUI_CELL_WIDTH.value

        # Write names
        for i, name in enumerate(["Player"] + names):
            color = self.agent_data[name]["color"] if name in self.agent_data else WHITE
            # add opacity to the color, where color is a Color(r,g,b)
            if name in self.agent_fov and not self.agent_fov[name]:
                color = tuple([int(0.5 * rgb) for rgb in color])
            self.render_cell_text(self.score_cols["Player"][i], name, bg_color=color)

        # Write other columns
        for i, col in enumerate(["Army", "Land"]):
            self.render_cell_text(self.score_cols[col][0], col)
            for j, name in enumerate(names):
                if name in self.agent_fov and not self.agent_fov[name]:
                    color = (128, 128, 128)
                self.render_cell_text(
                    self.score_cols[col][j + 1],
                    str(player_stats[name][col.lower()]),
                    bg_color=WHITE,
                )

        # Blit each right_panel cell to the right_panel surface
        for i, col in enumerate(["Player", "Army", "Land"]):
            for j, cell in enumerate(self.score_cols[col]):
                rect_dim = (0, 0, cell.get_width(), cell.get_height())
                pygame.draw.rect(cell, BLACK, rect_dim, 1)

                position = ((i + 1) * gui_cell_width, j * gui_cell_height)
                if col == "Player":
                    position = (0, j * gui_cell_height)
                self.right_panel.blit(cell, position)

        info_text = {
            "time": f"Time: {str(self.game.time // 2) + ('.' if self.game.time % 2 == 1 else '')}",
            "speed": "Paused"
            if self.mode == GuiMode.REPLAY and self.properties.paused
            else f"Speed: {str(self.properties.game_speed)}x",
        }

        # Write additional info
        for i, key in enumerate(["time", "speed"]):
            self.render_cell_text(self.info_panel[key], info_text[key])

            rect_dim = (
                0,
                0,
                self.info_panel[key].get_width(),
                self.info_panel[key].get_height(),
            )
            pygame.draw.rect(self.info_panel[key], BLACK, rect_dim, 1)

            self.right_panel.blit(self.info_panel[key], (i * 2 * gui_cell_width, 3 * gui_cell_height))
        # Render right_panel on the screen
        self.screen.blit(self.right_panel, (self.display_grid_width, 0))

    def render_grid(self):
        """
        Render the game grid
        """
        agents = self.game.agents
        # Maps of all owned and visible cells
        owned_map = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        visible_map = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        for agent in agents:
            ownership = self.game.channels.ownership[agent]
            owned_map = np.logical_or(owned_map, ownership)
            if self.agent_fov[agent]:
                visibility = self.game.channels.get_visibility(agent)
                visible_map = np.logical_or(visible_map, visibility)

        # Helper maps for not owned and invisible cells
        not_owned_map = np.logical_not(owned_map)
        invisible_map = np.logical_not(visible_map)

        # Draw background of visible owned squares
        for agent in agents:
            ownership = self.game.channels.ownership[agent]
            visible_ownership = np.logical_and(ownership, visible_map)
            self.draw_channel(visible_ownership, self.agent_data[agent]["color"])

        # Draw visible generals
        visible_generals = np.logical_and(self.game.channels.generals, visible_map)
        self.draw_images(visible_generals, self._general_img)

        # Draw background of visible but not owned squares
        visible_not_owned = np.logical_and(visible_map, not_owned_map)
        self.draw_channel(visible_not_owned, WHITE)

        # Draw background of squares in fog of war
        self.draw_channel(invisible_map, FOG_OF_WAR)

        # Draw background of visible mountains
        visible_mountain = np.logical_and(self.game.channels.mountains, visible_map)
        self.draw_channel(visible_mountain, VISIBLE_MOUNTAIN)

        # Draw mountains (even if they are not visible)
        self.draw_images(self.game.channels.mountains, self._mountain_img)

        # Draw background of visible neutral cities
        visible_cities = np.logical_and(self.game.channels.cities, visible_map)
        visible_cities_neutral = np.logical_and(visible_cities, self.game.channels.ownership_neutral)
        self.draw_channel(visible_cities_neutral, NEUTRAL_CASTLE)

        # Draw invisible cities as mountains
        invisible_cities = np.logical_and(self.game.channels.cities, invisible_map)
        self.draw_images(invisible_cities, self._mountain_img)

        # Draw visible cities
        self.draw_images(visible_cities, self._city_img)

        # Draw nonzero army counts on visible squares
        visible_army = self.game.channels.armies * visible_map
        visible_army_indices = self.channel_to_indices(visible_army)
        for i, j in visible_army_indices:
            self.render_cell_text(
                self.tiles[i][j],
                str(int(visible_army[i, j])),
                fg_color=WHITE,
                bg_color=None,  # Transparent background
            )

        # Blit tiles to the self.game_area
        square_size = Dimension.SQUARE_SIZE.value
        for i, j in np.ndindex(self.grid_height, self.grid_width):
            self.game_area.blit(self.tiles[i][j], (j * square_size, i * square_size))
        self.screen.blit(self.game_area, (0, 0))

    def channel_to_indices(self, channel: np.ndarray) -> np.ndarray:
        """
        Returns a list of indices of cells with non-zero values from specified a channel.
        """
        return np.argwhere(channel != 0)

    def draw_channel(self, channel: np.ndarray, color: Color):
        """
        Draw background and borders (left and top) for grid tiles of a given channel
        """
        square_size = Dimension.SQUARE_SIZE.value
        for i, j in self.channel_to_indices(channel):
            self.tiles[i][j].fill(color)
            pygame.draw.line(self.tiles[i][j], BLACK, (0, 0), (0, square_size), 1)
            pygame.draw.line(self.tiles[i][j], BLACK, (0, 0), (square_size, 0), 1)

    def draw_images(self, channel: np.ndarray, image: pygame.Surface):
        """
        Draw images on grid tiles of a given channel
        """
        for i, j in self.channel_to_indices(channel):
            self.tiles[i][j].blit(image, (3, 2))
