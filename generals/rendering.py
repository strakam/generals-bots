import time
import pygame
import numpy as np
import generals.game as game
import generals.config as c
from typing import Tuple, Dict, Any


class Renderer:
    def __init__(self, game: game.Game, agent_data: Dict[str, Dict[str, Any]], from_replay=False):
        """
        Initialize the pygame GUI

        Args:
            game: game object
            from_replay: bool, whether the game is from a replay
        """
        pygame.init()
        pygame.display.set_caption("Generals")
        pygame.key.set_repeat(500, 64)

        self.game = game
        self.from_replay = from_replay
        self.agent_data = agent_data
        self.grid_size = game.grid_size
        self.grid_width = c.SQUARE_SIZE * self.grid_size
        self.grid_height = c.SQUARE_SIZE * self.grid_size
        self.right_panel_width = 4 * c.GUI_CELL_WIDTH
        ############
        # Surfaces #
        ############
        window_width = self.grid_width + self.right_panel_width
        window_height = self.grid_height + 1

        # Main window
        self.screen = pygame.display.set_mode(
            (window_width, window_height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        # Scoreboard
        self.right_panel = pygame.Surface((self.right_panel_width, window_height))
        self.score_cols = {}
        for col in ["Agent", "Army", "Land"]:
            size = (c.GUI_CELL_WIDTH, c.GUI_ROW_HEIGHT)
            if col == "Agent":
                size = (2 * c.GUI_CELL_WIDTH, c.GUI_ROW_HEIGHT)
            self.score_cols[col] = [pygame.Surface(size) for _ in range(3)]

        self.info_panel = {
            "time": pygame.Surface((self.right_panel_width / 2, c.GUI_ROW_HEIGHT)),
            "speed": pygame.Surface((self.right_panel_width / 2, c.GUI_ROW_HEIGHT)),
        }
        # Game area and tiles
        self.game_area = pygame.Surface((self.grid_width, self.grid_height))
        self.tiles = [
            [
                pygame.Surface((c.SQUARE_SIZE, c.SQUARE_SIZE))
                for _ in range(self.grid_size)
            ]
            for _ in range(self.grid_size)
        ]

        self.agent_fov = {name: True for name in self.agent_data.keys()}
        self.game_speed = 1
        self.paused = False
        self.clock = pygame.time.Clock()
        self.last_render_time = time.time()

        self._mountain_img = pygame.image.load(
            str(c.MOUNTAIN_PATH), "png"
        ).convert_alpha()
        self._general_img = pygame.image.load(
            str(c.GENERAL_PATH), "png"
        ).convert_alpha()
        self._city_img = pygame.image.load(str(c.CITY_PATH), "png").convert_alpha()

        self._font = pygame.font.Font(c.FONT_PATH, c.FONT_SIZE)

    def handle_events(self):
        """
        Handle pygame GUI events
        """
        agents = self.game.agents
        control_events = {
            "time_change": 0,
        }
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_q
            ):
                pygame.quit()
                quit()

            # Control game speed, pause, and replay frames if the game is from a replay
            if event.type == pygame.KEYDOWN and self.from_replay:
                # Speed up game right arrow is pressed
                if event.key == pygame.K_RIGHT:
                    self.game_speed = max(1 / 128, self.game_speed / 2)
                # Slow down game left arrow is pressed
                if event.key == pygame.K_LEFT:
                    self.game_speed = min(32, self.game_speed * 2)
                # Toggle play/pause
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                # Control replay frames
                if event.key == pygame.K_h:
                    control_events["time_change"] = -1
                    self.paused = True
                if event.key == pygame.K_l:
                    control_events["time_change"] = 1
                    self.paused = True

            # GUI clicks
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                for i, agent in enumerate(agents):
                    # if clicked agents row in the right panel
                    if x >= self.grid_width and y >= (i + 1) * c.GUI_ROW_HEIGHT and y < (i + 2) * c.GUI_ROW_HEIGHT:
                        self.agent_fov[agent] = not self.agent_fov[agent]
                        break
        return control_events

    def render(self):
        control_events = self.handle_events()
        self.render_grid()
        self.render_stats()
        pygame.display.flip()
        return control_events

    def render_cell_text(self, cell, text, fg_color=c.BLACK, bg_color=c.WHITE):
        """
        Draw a text in the middle of the cell with given foreground and background colors

        Args:
            cell: cell to draw
            text: text to write on the cell
            fg_color: foreground color of the text
            bg_color: background color of the cell
        """
        center = (cell.get_width() // 2, cell.get_height() // 2)

        text = self._font.render(text, True, fg_color)
        if bg_color:
            cell.fill(bg_color)
        cell.blit(text, text.get_rect(center=center))

    def render_stats(self):
        """
        Draw player stats and additional info on the right panel
        """
        names = self.game.agents
        player_stats = self.game.get_infos()

        # Write names
        for i, name in enumerate(["Agent"] + names):
            color = self.agent_data[name]["color"] if name in self.agent_data else c.WHITE
            # add opacity to the color, where color is a tuple (r,g,b)
            if name in self.agent_fov and not self.agent_fov[name]:
                color = tuple([int(0.5 * c) for c in color])
            self.render_cell_text(self.score_cols["Agent"][i], name, bg_color=color)

        # Write other columns
        for i, col in enumerate(["Army", "Land"]):
            self.render_cell_text(self.score_cols[col][0], col)
            for j, name in enumerate(names):
                # Give darkish color if agents FoV is off
                color = c.WHITE
                if name in self.agent_fov and not self.agent_fov[name]:
                    color = (128, 128, 128)
                self.render_cell_text(
                    self.score_cols[col][j + 1],
                    str(player_stats[name][col.lower()]),
                    bg_color=color,
                )

        # Blit each right_panel cell to the right_panel surface
        for i, col in enumerate(["Agent", "Army", "Land"]):
            for j, cell in enumerate(self.score_cols[col]):
                rect_dim = (0,0,cell.get_width(),cell.get_height())
                pygame.draw.rect(cell,c.BLACK,rect_dim,1)

                position = ((i+1) * c.GUI_CELL_WIDTH, j * c.GUI_ROW_HEIGHT)
                if col == "Agent":
                    position = (0, j * c.GUI_ROW_HEIGHT)
                self.right_panel.blit(cell, position)

        info_text = {
            "time": f"Time: {str(self.game.time // 2) + ('.' if self.game.time % 2 == 1 else '')}",
            "speed": "Paused" if self.paused else f"Speed: {str(1 / self.game_speed)}x",
        }

        # Write additional info
        for i, key in enumerate(["time", "speed"]):
            self.render_cell_text(self.info_panel[key], info_text[key])

            rect_dim = (0,0,self.info_panel[key].get_width(),self.info_panel[key].get_height())
            pygame.draw.rect(self.info_panel[key],c.BLACK,rect_dim,1)

            self.right_panel.blit(
                self.info_panel[key], (i*2*c.GUI_CELL_WIDTH, 3 * c.GUI_ROW_HEIGHT)
            )
        # Render right_panel on the screen
        self.screen.blit(self.right_panel, (self.grid_width, 0))

    def render_grid(self):
        """
        Render the game grid
        """
        agents = self.game.agents
        # Maps of all owned and visible cells
        owned_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        visible_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for agent in agents:
            ownership = self.game.channels["ownership_" + agent]
            owned_map = np.logical_or(owned_map, ownership)
            if self.agent_fov[agent]:
                visibility = self.game.visibility_channel(ownership)
                visible_map = np.logical_or(visible_map, visibility)

        # Helper maps for not owned and invisible cells
        not_owned_map = np.logical_not(owned_map)
        invisible_map = np.logical_not(visible_map)

        # Draw background of visible owned squares
        for agent in agents:
            ownership = self.game.channels["ownership_" + agent]
            visible_ownership = np.logical_and(ownership, visible_map)
            visible_ownership_indices = self.game.channel_to_indices(visible_ownership)
            self.draw_channel(visible_ownership_indices, self.agent_data[agent]["color"])

        # Draw visible generals
        visible_generals = np.logical_and(self.game.channels["general"], visible_map)
        visible_generals_indices = self.game.channel_to_indices(visible_generals)
        self.draw_images(visible_generals_indices, self._general_img)

        # Draw background of visible but not owned squares
        visible_not_owned = np.logical_and(visible_map, not_owned_map)
        visible_not_owned_indices = self.game.channel_to_indices(visible_not_owned)
        self.draw_channel(visible_not_owned_indices, c.WHITE)

        # Draw background of squares in fog of war
        invisible_map_indices = self.game.channel_to_indices(invisible_map)
        self.draw_channel(invisible_map_indices, c.FOG_OF_WAR)

        # Draw background of visible mountains
        visible_mountain = np.logical_and(self.game.channels["mountain"], visible_map)
        visible_mountain_indices = self.game.channel_to_indices(visible_mountain)
        self.draw_channel(visible_mountain_indices, c.VISIBLE_MOUNTAIN)

        # Draw mountains (even if they are not visible)
        mountain_indices = self.game.channel_to_indices(self.game.channels["mountain"])
        self.draw_images(mountain_indices, self._mountain_img)

        # Draw background of visible neutral cities
        visible_cities = np.logical_and(self.game.channels["city"], visible_map)
        visible_cities_neutral = np.logical_and(
            visible_cities, self.game.channels["ownership_neutral"]
        )
        visible_cities_neutral_indices = self.game.channel_to_indices(
            visible_cities_neutral
        )
        self.draw_channel(visible_cities_neutral_indices, c.NEUTRAL_CASTLE)

        # Draw invisible cities as mountains
        invisible_cities = np.logical_and(self.game.channels["city"], invisible_map)
        invisible_cities_indices = self.game.channel_to_indices(invisible_cities)
        self.draw_images(invisible_cities_indices, self._mountain_img)

        # Draw visible cities
        visible_cities_indices = self.game.channel_to_indices(visible_cities)
        self.draw_images(visible_cities_indices, self._city_img)

        # Draw nonzero army counts on visible squares
        visible_army = self.game.channels["army"] * visible_map
        visible_army_indices = self.game.channel_to_indices(visible_army)
        for i, j in visible_army_indices:
            self.render_cell_text(
                self.tiles[i][j],
                str(int(visible_army[i, j])),
                fg_color=c.WHITE,
                bg_color=None,  # Transparent background
            )

        # Blit tiles to the self.game_area
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.game_area.blit(
                    self.tiles[i][j], (j * c.SQUARE_SIZE, i * c.SQUARE_SIZE)
                )
        self.screen.blit(self.game_area, (0, 0))

    def draw_channel(self, channel: list[Tuple[int, int]], color: Tuple[int, int, int]):
        """
        Draw background and borders (left and top) for grid tiles of a given channel

        Args:
            channel: list of tuples with indices of grid tiles
            color: background color of the squares
        """
        for i, j in channel:
            self.tiles[i][j].fill(color)
            pygame.draw.line(self.tiles[i][j], c.BLACK, (0, 0), (0, c.SQUARE_SIZE), 1)
            pygame.draw.line(self.tiles[i][j], c.BLACK, (0, 0), (c.SQUARE_SIZE, 0), 1)

    def draw_images(self, channel: list[Tuple[int, int]], image):
        """
        Draw images on grid tiles of a given channel

        Args:
            channel: list of tuples with indices of grid tiles
            image: pygame image object
        """
        for i, j in channel:
            self.tiles[i][j].blit(image, (3, 2))
