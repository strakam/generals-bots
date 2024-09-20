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
        self.grid_offset = c.UI_ROW_HEIGHT * (len(self.agents) + 1) # area for GUI
        self.window_width = max(c.MINIMUM_WINDOW_SIZE, c.SQUARE_SIZE * self.grid_size)
        self.window_height = max(
            c.MINIMUM_WINDOW_SIZE, c.SQUARE_SIZE * self.grid_size + self.grid_offset
        )

        # Surfaces
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        # Scoreboard
        self.scoreboard = pygame.Surface(
            (self.window_width, c.UI_ROW_HEIGHT * (len(self.agents) + 1))
        )
        # Game area
        self.game_area = pygame.Surface(
            (self.window_width, self.window_height - c.UI_ROW_HEIGHT * (len(self.agents) + 1))
        )

        self.agent_fov = {name: True for name in self.agents}
        self.game_speed = 1
        self.paused = True
        self.changed = True
        self.clock = pygame.time.Clock()
        self.last_render_time = time.time()

        self._mountain_img = pygame.image.load(str(c.MOUNTAIN_PATH), "png").convert_alpha()
        self._general_img = pygame.image.load(str(c.GENERAL_PATH), "png").convert_alpha()
        self._city_img = pygame.image.load(str(c.CITY_PATH), "png").convert_alpha()

        self._font = pygame.font.Font(c.FONT_PATH, c.FONT_SIZE)


    def handle_events(self):
        """
        Handle pygame GUI events
        """
        agents = self.game.agents
        control_events = {
            'time_change': 0,
        }
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                self.changed = True
                control_events['changed'] = True
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()

                # Speed up game right arrow is pressed
                if event.key == pygame.K_RIGHT and not self.paused:
                    self.game_speed = max(1/16, self.game_speed / 2)
                # Slow down game left arrow is pressed
                if event.key == pygame.K_LEFT and not self.paused:
                    self.game_speed = min(32, self.game_speed * 2)
                # Pause game
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                # Control replay frames
                if event.key == pygame.K_h:
                    control_events['time_change'] = -1
                    self.paused = True
                if event.key == pygame.K_l:
                    control_events['time_change'] = 1
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
        self.game_area.fill(c.WHITE)
        self.screen.blit(self.scoreboard, (0, 0))
        self.screen.blit(self.game_area, (0, self.grid_offset))
        pygame.display.flip()
        return control_events

    def render_gui(self, from_replay=False):
        """
        Draw player stats on the scoreboard surface
        """
        names = self.game.agents
        ids = [self.game.agent_id[name] for name in names]
        player_stats = self.game.get_infos()
        army_counts = ["Army"] + [
            str(player_stats[name]["army"]) for name in names
        ]
        land_counts = ["Land"] + [
            str(player_stats[name]["land"]) for name in names
        ]
        fovs = ["FOV"] + ["  X" if self.agent_fov[name] else " " for name in names]
        names = ["Turn"] + [name for name in names]

        # White background for GUI
        self.scoreboard.fill(c.WHITE)

        # Draw rows with player colors
        for i, agent_id in enumerate(ids):
            pygame.draw.rect(
                self.scoreboard,
                c.PLAYER_COLORS[agent_id],
                (
                    0,
                    (i + 1) * c.UI_ROW_HEIGHT,
                    self.scoreboard.get_width() - 3 * c.GUI_CELL_WIDTH,
                    c.UI_ROW_HEIGHT,
                ),
            )

        # Draw lines between rows
        for i in range(1, 3):
            pygame.draw.line(
                self.scoreboard,
                c.BLACK,
                (0, i * c.UI_ROW_HEIGHT),
                (self.scoreboard.get_width(), i * c.UI_ROW_HEIGHT),
                c.LINE_WIDTH,
            )

        # Draw vertical lines cell_width from the end and 2*cell_width from the end
        for i in range(1, 4):
            pygame.draw.line(
                self.screen,
                c.BLACK,
                (self.window_width - i * c.GUI_CELL_WIDTH, 0),
                (self.window_width - i * c.GUI_CELL_WIDTH, self.grid_offset),
                c.LINE_WIDTH,
            )

        if from_replay:
            speed = (
                "Paused" if self.paused else str(1 /  self.game_speed) + "x"
            )
            text = self._font.render(f"Game speed: {speed}", True, c.BLACK)
            self.scoreboard.blit(text, (150, 15))

        # For each player print his name, army, land and FOV (Field of View)
        for i in range(len(ids) + 1):
            if i == 0:
                turn = str(self.game.time // 2) + ("." if self.game.time % 2 == 1 else "")
                text = self._font.render(f"{names[i]}: {turn}", True, c.BLACK)
            else:
                text = self._font.render(f"{names[i]}", True, c.BLACK)
            y_offset = i * c.UI_ROW_HEIGHT + 15
            x_offset = 27
            self.scoreboard.blit(text, (10, y_offset))
            text = self._font.render(army_counts[i], True, c.BLACK)
            self.scoreboard.blit(
                text, (self.window_width - 2 * c.GUI_CELL_WIDTH + x_offset, y_offset)
            )
            text = self._font.render(land_counts[i], True, c.BLACK)
            self.scoreboard.blit(
                text, (self.window_width - c.GUI_CELL_WIDTH + x_offset, y_offset)
            )
            text = self._font.render(fovs[i], True, c.BLACK)
            self.scoreboard.blit(
                text, (self.window_width - 3 * c.GUI_CELL_WIDTH + x_offset, y_offset)
            )
        self.changed = False

    def render_grid(self):
        """
        Render grid part of the game.

        Args:
            game: Game object
            agent_ids: list of agent ids from which perspective the game is rendered
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
            self.draw_channel(ownership_indices, c.PLAYER_COLORS[self.game.agent_id[agent]])

        # draw lines
        for i in range(1, self.grid_size):
            pygame.draw.line(
                self.screen,
                c.BLACK,
                (0, i * c.SQUARE_SIZE + self.grid_offset),
                (self.window_width, i * c.SQUARE_SIZE + self.grid_offset),
                c.LINE_WIDTH,
            )
            pygame.draw.line(
                self.screen,
                c.BLACK,
                (i * c.SQUARE_SIZE, self.grid_offset),
                (i * c.SQUARE_SIZE, self.window_height),
                c.LINE_WIDTH,
            )

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
        visible_cities_neutral_indices = self.game.channel_to_indices(visible_cities_neutral)
        self.draw_channel(visible_cities_neutral_indices, c.NEUTRAL_CASTLE)

        # draw visible city images
        visible_cities_indices = self.game.channel_to_indices(visible_cities)
        self.draw_images(visible_cities_indices, self._city_img)

        # draw army counts on visibility mask
        army = self.game.channels["army"] * visible_map
        visible_army_indices = self.game.channel_to_indices(army)
        y_offset = 15
        for i, j in visible_army_indices:
            text = self._font.render(str(int(army[i, j])), True, c.WHITE)
            x_offset = c.FONT_OFFSETS[
                min(len(c.FONT_OFFSETS) - 1, len(str(int(army[i, j]))) - 1)
            ]
            self.screen.blit(
                text,
                (
                    j * c.SQUARE_SIZE + x_offset,
                    i * c.SQUARE_SIZE + y_offset + self.grid_offset,
                ),
            )

    def draw_channel(self, channel: list[Tuple[int, int]], color: Tuple[int, int, int]):
        """
        Draw channel squares on the self.screen

        Args:
            channel: list of tuples with indices of the channel
            color: color of the squares
        """
        size, offset = c.SQUARE_SIZE, self.grid_offset
        w = c.LINE_WIDTH
        for i, j in channel:
            pygame.draw.rect(
                self.screen,
                color,
                (j * size + w, i * size + w + offset, size - w, size - w),
            )

    def draw_images(self, channel: list[Tuple[int, int]], image):
        """
        Draw images on the self.screen

        Args:
            self.screen: pygame self.screen object
            channel: list of tuples with indices of the channel
            image: pygame image object
        """
        size, offset = c.SQUARE_SIZE, self.grid_offset
        for i, j in channel:
            self.screen.blit(image, (j * size + 3, i * size + 3 + offset))
