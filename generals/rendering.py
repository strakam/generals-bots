import time
import pygame
import numpy as np
import generals.game as game
import generals.config as c
from typing import Tuple


class Renderer:
    def __init__(self, game: game.Game):
        """
        Initialize pygame window

        Args:
            config: game config object
        """
        pygame.init()
        pygame.display.set_caption("Generals")
        pygame.key.set_repeat(500, 64)

        self.game = game
        self.agents = game.agents
        self.grid_size = game.grid_size
        self.grid_offset = c.GUI_ROW_HEIGHT * (len(self.agents) + 1)  # area for GUI
        self.grid_width = c.SQUARE_SIZE * self.grid_size
        self.grid_height = c.SQUARE_SIZE * self.grid_size
        self.player_colors = {
            agent: c.PLAYER_COLORS[i] for i, agent in enumerate(self.agents)
        }

        ############
        # Surfaces #
        ############
        window_width = self.grid_width + 4 * c.GUI_CELL_WIDTH
        window_height = self.grid_height

        # Main window
        self.screen = pygame.display.set_mode(
            (window_width, window_height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        # Scoreboard
        self.right_panel = pygame.Surface(
            (4*c.GUI_CELL_WIDTH, window_height)
        )
        self.score_cols = {}
        for col in ["Agent", "Army", "Land"]:
            size = (c.GUI_CELL_WIDTH, c.GUI_ROW_HEIGHT)
            if col == "Agent":
                size = (2*c.GUI_CELL_WIDTH, c.GUI_ROW_HEIGHT)
            self.score_cols[col] = [
                pygame.Surface(size) for _ in range(3)
            ]

        self.info_panel = {
            "time": pygame.Surface((2*c.GUI_CELL_WIDTH, c.GUI_ROW_HEIGHT)),
            "speed": pygame.Surface((2*c.GUI_CELL_WIDTH, c.GUI_ROW_HEIGHT)),
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

        self.agent_fov = {name: True for name in self.agents}
        self.game_speed = 1
        self.paused = True
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
            if event.type == pygame.KEYDOWN:
                # Speed up game right arrow is pressed
                if event.key == pygame.K_RIGHT and not self.paused:
                    self.game_speed = max(1 / 16, self.game_speed / 2)
                # Slow down game left arrow is pressed
                if event.key == pygame.K_LEFT and not self.paused:
                    self.game_speed = min(32, self.game_speed * 2)
                # Pause game
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
                _, y = pygame.mouse.get_pos()
                for i, agent in enumerate(agents):
                    if y < c.UI_ROW_HEIGHT * (i + 2) and y > c.UI_ROW_HEIGHT * (i + 1):
                        self.agent_fov[agent] = not self.agent_fov[agent]
        return control_events

    def render(self, from_replay=False):
        control_events = self.handle_events()
        self.render_grid()
        self.render_gui(from_replay)
        pygame.display.flip()
        return control_events

    def render_gui_cell(self, cell, text, color):
        """
        Draw a text in the middle of the cell with the given background color

        Args:
            cell: cell to draw
            text: text to write on the cell
            color: color of the cell
        """
        center = (cell.get_width() // 2, cell.get_height() // 2)

        text = self._font.render(text, True, c.BLACK)
        cell.fill(color)
        cell.blit(text, text.get_rect(center=center))

    def render_gui(self, from_replay=False):
        """
        Draw player stats on the right_panel surface
        """
        names = self.game.agents
        player_stats = self.game.get_infos()

        # Write names
        for i, name in enumerate(["Agent"] + names):
            color = self.player_colors[name] if name in self.player_colors else c.WHITE
            self.render_gui_cell(self.score_cols["Agent"][i], name, color)

        # Write other columns
        for i, col in enumerate(["Army", "Land"]):
            self.render_gui_cell(self.score_cols[col][0], col, c.WHITE)
            for j, name in enumerate(names):
                self.render_gui_cell(
                    self.score_cols[col][j + 1],
                    str(player_stats[name][col.lower()]),
                    c.WHITE,
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
            "speed": "Paused" if self.paused and from_replay else f"Speed: {str(1 / self.game_speed)}x",
        }

        for i, key in enumerate(["time", "speed"]):
            self.render_gui_cell(
                self.info_panel[key],
                info_text[key],
                c.WHITE,
            )

            rect_dim = (0,0,self.info_panel[key].get_width(),self.info_panel[key].get_height())
            pygame.draw.rect(self.info_panel[key],c.BLACK,rect_dim,1)

            self.right_panel.blit(
                self.info_panel[key], (i*2*c.GUI_CELL_WIDTH, 3 * c.GUI_ROW_HEIGHT)
            )
        self.screen.blit(self.right_panel, (self.grid_width, 0))

    def render_grid(self):
        """
        Render grid part of the game on
        """
        agents = self.game.agents
        # draw visibility squares
        visible_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        owned_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for agent in agents:
            owned_map = np.logical_or(
                owned_map, self.game.channels["ownership_" + agent]
            )
        not_owned_map = np.logical_not(owned_map)
        for agent in agents:
            if not self.agent_fov[agent]:
                continue
            ownership = self.game.channels["ownership_" + agent]
            visibility = self.game.visibility_channel(ownership)
            _rendered_visibility = np.logical_and(visibility, not_owned_map)
            visibility_indices = self.game.channel_to_indices(_rendered_visibility)
            self.draw_channel(visibility_indices, c.WHITE)
            visible_map = np.logical_or(
                visible_map, visibility
            )  # get all visible cells

        # draw ownership squares
        for agent in agents:
            ownership = self.game.channels["ownership_" + agent]
            ownership_indices = self.game.channel_to_indices(ownership)
            self.draw_channel(ownership_indices, self.player_colors[agent])

        # draw background as squares
        invisible_map = np.logical_not(visible_map)
        invisible_indices = self.game.channel_to_indices(invisible_map)
        self.draw_channel(invisible_indices, c.FOG_OF_WAR)

        # Draw different color for visible mountains
        visible_mountain = np.logical_and(self.game.channels["mountain"], visible_map)
        visible_mountain_indices = self.game.channel_to_indices(visible_mountain)
        self.draw_channel(visible_mountain_indices, c.VISIBLE_MOUNTAIN)

        # Draw mountain image everywhere it is
        mountain_indices = self.game.channel_to_indices(self.game.channels["mountain"])
        self.draw_images(mountain_indices, self._mountain_img)

        # Draw invisible cities as mountains
        invisible_cities = np.logical_and(self.game.channels["city"], invisible_map)
        invisible_cities_indices = self.game.channel_to_indices(invisible_cities)
        self.draw_images(invisible_cities_indices, self._mountain_img)

        # draw general
        for agent, fov in self.agent_fov.items():
            if fov:
                visible_squares = self.game.visibility_channel(
                    self.game.channels["ownership_" + agent]
                )
                agent_general = np.logical_and(
                    visible_squares, self.game.channels["general"]
                )
                general_indices = self.game.channel_to_indices(agent_general)
                self.draw_images(general_indices, self._general_img)

        # draw neutral visible city color
        visible_cities = np.logical_and(self.game.channels["city"], visible_map)
        visible_cities_neutral = np.logical_and(
            visible_cities, self.game.channels["ownership_neutral"]
        )
        visible_cities_neutral_indices = self.game.channel_to_indices(
            visible_cities_neutral
        )
        self.draw_channel(visible_cities_neutral_indices, c.NEUTRAL_CASTLE)

        # draw visible city images
        visible_cities_indices = self.game.channel_to_indices(visible_cities)
        self.draw_images(visible_cities_indices, self._city_img)

        # draw army counts on visibility mask
        army = self.game.channels["army"] * visible_map
        visible_army_indices = self.game.channel_to_indices(army)
        for i, j in visible_army_indices:
            text = self._font.render(str(int(army[i, j])), True, c.WHITE)
            text_rect = text.get_rect(center=(c.SQUARE_SIZE / 2, c.SQUARE_SIZE / 2))
            self.tiles[i][j].blit(text, text_rect)

        # Blit tiles to the self.game_area
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.game_area.blit(
                    self.tiles[i][j], (j * c.SQUARE_SIZE, i * c.SQUARE_SIZE)
                )
        self.screen.blit(self.game_area, (0, 0))

    def draw_channel(self, channel: list[Tuple[int, int]], color: Tuple[int, int, int]):
        """
        Draw channel squares on the self.screen

        Args:
            channel: list of tuples with indices of the channel
            color: color of the squares
        """
        for i, j in channel:
            self.tiles[i][j].fill(color)
            pygame.draw.line(self.tiles[i][j], c.BLACK, (0, 0), (0, c.SQUARE_SIZE), 1)
            pygame.draw.line(self.tiles[i][j], c.BLACK, (0, 0), (c.SQUARE_SIZE, 0), 1)

    def draw_images(self, channel: list[Tuple[int, int]], image):
        """
        Draw images on the self.screen

        Args:
            self.screen: pygame self.screen object
            channel: list of tuples with indices of the channel
            image: pygame image object
        """
        for i, j in channel:
            self.tiles[i][j].blit(image, (3, 2))
