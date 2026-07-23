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

# Brightness factors for owned structure tiles, relative to the owner's color.
GENERAL_SHADE = 0.65
CASTLE_SHADE = 0.8
GRID_LINE_SHADE = 0.72


def shade(color, factor: float) -> Color:
    """Scale a color's brightness by `factor`. Accepts tuples or names ("red")."""
    c = pygame.Color(color)
    return (int(c.r * factor), int(c.g * factor), int(c.b * factor))


class Renderer:
    def __init__(self, properties: Properties):
        """
        Initialize the pygame GUI
        """
        pygame.init()
        pygame.display.set_caption("Generals")
        pygame.key.set_repeat()  # OS key-repeat off; replay does its own hold-to-run

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
        width = Dimension.GUI_CELL_WIDTH.value
        height = Dimension.GUI_CELL_HEIGHT.value
        window_height = self.display_grid_height + 1

        # Fonts (loaded up-front so we can measure names to size the columns).
        self._font = pygame.font.Font(Path.FONT_PATH, self.properties.font_size)
        self._debug_font = pygame.font.Font(Path.FONT_PATH, 10)  # Smaller font for debug
        self._controls_font = pygame.font.Font(Path.FONT_PATH, 14)  # replay control legend

        # Size the Player (name) column to the widest agent name so names never
        # overflow the cell; Army/Land follow at the standard cell width. The
        # panel width is whatever that adds up to.
        names = list(self.agent_data.keys())
        max_name_w = max((self._font.size(n)[0] for n in names), default=0)
        self.player_col_width = max(2 * width, max_name_w + 20)
        self.right_panel_width = self.player_col_width + 2 * width

        window_width = self.display_grid_width + self.right_panel_width

        # Main window
        self.screen = pygame.display.set_mode((window_width, window_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        # Scoreboard
        self.right_panel = pygame.Surface((self.right_panel_width, window_height))
        self.score_cols = {}
        for col in ["Player", "Army", "Land"]:
            size = (self.player_col_width, height) if col == "Player" else (width, height)
            self.score_cols[col] = [pygame.Surface(size) for _ in range(3)]

        # Time box aligns under the (name) Player column; the speed/status box
        # fills the rest of the width (under Army + Land).
        self.info_panel = {
            "time": pygame.Surface((self.player_col_width, height)),
            "speed": pygame.Surface((self.right_panel_width - self.player_col_width, height)),
        }
        # Game area and tiles
        self.game_area = pygame.Surface((self.display_grid_width, self.display_grid_height))
        self.tiles = [
            [pygame.Surface((Dimension.SQUARE_SIZE.value, Dimension.SQUARE_SIZE.value)) for _ in range(self.grid_width)]
            for _ in range(self.grid_height)
        ]

        # Load pre-scaled images (crownie, citie, mountainie are already the right size)
        self._mountain_img = pygame.image.load(str(Path.MOUNTAIN_PATH), "png").convert_alpha()
        self._general_img = pygame.image.load(str(Path.GENERAL_PATH), "png").convert_alpha()
        self._castle_img = pygame.image.load(Path.CASTLE_PATH, "png").convert_alpha()
        # Dimmed mountain for terrain that is only remembered under fog of war
        self._fog_mountain_img = self._mountain_img.copy()
        self._fog_mountain_img.set_alpha(110)

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
        shadow: bool = False,
    ):
        """
        Draw a text in the middle of the cell with given foreground and background colors

        Args:
            cell: cell to draw
            text: text to write on the cell
            fg_color: foreground color of the text
            bg_color: background color of the cell
            shadow: draw a 1px black drop shadow under the text (keeps light
                text readable on light tiles)
        """
        center = (cell.get_width() // 2, cell.get_height() // 2)

        text_surface = self._font.render(text, True, fg_color)
        if bg_color:
            cell.fill(bg_color)
        if shadow:
            shadow_surface = self._font.render(text, True, BLACK)
            cell.blit(shadow_surface, shadow_surface.get_rect(center=(center[0] + 1, center[1] + 1)))
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

        # Blit each right_panel cell. The Player column is sized to the widest
        # name; Army/Land follow it at the standard cell width.
        col_x = {
            "Player": 0,
            "Army": self.player_col_width,
            "Land": self.player_col_width + gui_cell_width,
        }
        for col in ["Player", "Army", "Land"]:
            for j, cell in enumerate(self.score_cols[col]):
                rect_dim = (0, 0, cell.get_width(), cell.get_height())
                pygame.draw.rect(cell, BLACK, rect_dim, 1)
                self.right_panel.blit(cell, (col_x[col], j * gui_cell_height))

        if self.mode == GuiMode.REPLAY:
            speed_text = "Paused" if self.properties.paused else "Playing"
        else:
            speed_text = f"Speed: {str(self.properties.game_speed)}x"
        info_text = {
            "time": f"Time: {str(self.game.time // 2) + ('.' if self.game.time % 2 == 1 else '')}",
            "speed": speed_text,
        }

        # Write additional info. Time box sits under the name column; the
        # speed/status box fills the remaining width under Army + Land.
        info_x = {"time": 0, "speed": self.player_col_width}
        for key in ["time", "speed"]:
            self.render_cell_text(self.info_panel[key], info_text[key])

            rect_dim = (
                0,
                0,
                self.info_panel[key].get_width(),
                self.info_panel[key].get_height(),
            )
            pygame.draw.rect(self.info_panel[key], BLACK, rect_dim, 1)

            self.right_panel.blit(self.info_panel[key], (info_x[key], 3 * gui_cell_height))

        if self.mode == GuiMode.REPLAY:
            self._render_controls()

        # Render right_panel on the screen
        self.screen.blit(self.right_panel, (self.display_grid_width, 0))

    def _render_controls(self):
        """Draw a tidy two-column control legend below the scoreboard (REPLAY)."""
        KEY_COLOR = (236, 206, 112)    # soft gold — key names
        DESC_COLOR = (170, 174, 178)   # muted gray — descriptions
        RULE_COLOR = (88, 92, 98)      # subtle divider

        pad = 12
        top = 4 * Dimension.GUI_CELL_HEIGHT.value + 8
        rows = [
            ("Space", "play / pause"),
            ("Left / Right", "step a frame"),
            ("", "hold to run through"),
            ("R", "restart"),
            ("Q", "quit"),
        ]
        line_h = 21
        head_h = self.properties.font_size + 14

        # Clear the legend area, then draw a divider, heading, and the rows.
        self.right_panel.fill(
            BLACK, pygame.Rect(0, top, self.right_panel_width, head_h + len(rows) * line_h + 16)
        )
        pygame.draw.line(self.right_panel, RULE_COLOR,
                         (pad, top + 2), (self.right_panel_width - pad, top + 2), 1)
        self.right_panel.blit(self._font.render("Controls", True, WHITE), (pad, top + 10))

        key_x, desc_x = pad + 4, pad + 120
        y = top + head_h + 4
        for key, desc in rows:
            if key:
                self.right_panel.blit(self._controls_font.render(key, True, KEY_COLOR), (key_x, y))
            self.right_panel.blit(self._controls_font.render(desc, True, DESC_COLOR), (desc_x, y))
            y += line_h

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

        # Draw background of visible owned squares. Generals and castles get a
        # darker shade of the owner's color so they stand out from body tiles.
        for agent in agents:
            ownership = self.game.channels.ownership[agent]
            visible_ownership = np.logical_and(ownership, visible_map)
            color = self.agent_data[agent]["color"]
            self.draw_channel(visible_ownership, color)
            owned_generals = np.logical_and(visible_ownership, self.game.channels.generals)
            self.draw_channel(owned_generals, shade(color, GENERAL_SHADE))
            owned_castles = np.logical_and(visible_ownership, self.game.channels.castles)
            self.draw_channel(owned_castles, shade(color, CASTLE_SHADE))

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

        # Draw mountains: full icon when visible, dimmed when under fog of war
        self.draw_images(visible_mountain, self._mountain_img)
        fog_mountain = np.logical_and(self.game.channels.mountains, invisible_map)
        self.draw_images(fog_mountain, self._fog_mountain_img)

        # Draw background of visible neutral castles
        visible_castles = np.logical_and(self.game.channels.castles, visible_map)
        visible_castles_neutral = np.logical_and(visible_castles, self.game.channels.ownership_neutral)
        self.draw_channel(visible_castles_neutral, NEUTRAL_CASTLE)

        # Draw invisible castles as dimmed mountains
        invisible_castles = np.logical_and(self.game.channels.castles, invisible_map)
        self.draw_images(invisible_castles, self._fog_mountain_img)

        # Draw visible castles
        self.draw_images(visible_castles, self._castle_img)

        # Draw nonzero army counts on visible squares
        visible_army = self.game.channels.armies * visible_map
        visible_army_indices = self.channel_to_indices(visible_army)
        for i, j in visible_army_indices:
            self.render_cell_text(
                self.tiles[i][j],
                str(int(visible_army[i, j])),
                fg_color=WHITE,
                bg_color=None,  # Transparent background
                shadow=True,
            )

        # Draw tile type debug labels if enabled
        if self.properties.show_tile_types:
            self.draw_tile_types()

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
        # Grid lines in a darker shade of the fill, softer than pure black
        border = shade(color, GRID_LINE_SHADE)
        for i, j in self.channel_to_indices(channel):
            self.tiles[i][j].fill(color)
            pygame.draw.line(self.tiles[i][j], border, (0, 0), (0, square_size), 1)
            pygame.draw.line(self.tiles[i][j], border, (0, 0), (square_size, 0), 1)

    def draw_images(self, channel: np.ndarray, image: pygame.Surface):
        """
        Draw images on grid tiles of a given channel
        """
        square_size = Dimension.SQUARE_SIZE.value
        # Center the image in the cell
        img_width, img_height = image.get_size()
        x_offset = (square_size - img_width) // 2
        y_offset = (square_size - img_height) // 2
        for i, j in self.channel_to_indices(channel):
            self.tiles[i][j].blit(image, (x_offset, y_offset))

    def draw_tile_types(self):
        """
        Draw tile type labels in the upper-right corner of each tile.
        Types: 0=empty, -2=mountain, 1=general0, 2=general1, 40-50=castle
        """
        square_size = Dimension.SQUARE_SIZE.value
        channels = self.game.channels
        agents = self.game.agents

        for i in range(self.grid_height):
            for j in range(self.grid_width):
                # Determine tile type
                if channels.mountains[i, j]:
                    tile_type = "-2"
                elif channels.generals[i, j]:
                    if channels.ownership[agents[0]][i, j]:
                        tile_type = "1"
                    else:
                        tile_type = "2"
                elif channels.castles[i, j]:
                    tile_type = "C"  # Castle
                else:
                    tile_type = "0"

                # Render the type label in upper-right corner
                text_surface = self._debug_font.render(tile_type, True, (0, 255, 0))  # Green text
                text_rect = text_surface.get_rect()
                # Position in upper-right with small padding
                x_pos = square_size - text_rect.width - 2
                y_pos = 2
                self.tiles[i][j].blit(text_surface, (x_pos, y_pos))
