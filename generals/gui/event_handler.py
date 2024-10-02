import pygame
from pygame.event import Event

from .properties import Properties
from generals.core import config as c

######################
# Replay keybindings #
######################
RIGHT = pygame.K_RIGHT
LEFT = pygame.K_LEFT
SPACE = pygame.K_SPACE
Q = pygame.K_q
R = pygame.K_r
H = pygame.K_h
L = pygame.K_l


class Command:
    def __init__(self):
        self.quit: bool = False


class ReplayCommand(Command):
    def __init__(self):
        super().__init__()
        self.frame_change: int = 0
        self.speed_change: float = 1.0
        self.restart: bool = False
        self.pause: bool = False


class GameCommand(Command):
    def __init__(self):
        super().__init__()
        raise NotImplementedError


class TrainCommand(Command):
    def __init__(self):
        super().__init__()


class EventHandler:
    def __init__(self, properties: Properties):
        """
        Initialize the event handler.

        Args:
            properties: the Properties object
        """
        self.properties = properties
        self.mode = properties.mode
        self.handler_fn = self.initialize_handler_fn()
        self.command = self.initialize_command()

    def initialize_handler_fn(self):
        """
        Initialize the handler function based on the mode.
        """
        if self.mode == "replay":
            return self.__handle_replay_key_controls
        elif self.mode == "game":
            return self.__handle_game_key_controls
        elif self.mode == "train":
            return self.__handle_train_key_controls
        raise ValueError("Invalid mode")

    def initialize_command(self):
        """
        Initialize the command type based on the mode.
        """
        if self.mode == "replay":
            return ReplayCommand
        elif self.mode == "game":
            return GameCommand
        elif self.mode == "train":
            return TrainCommand
        raise ValueError("Invalid mode")

    def handle_events(self) -> Command:
        """
        Handle pygame GUI events
        """
        command = self.command()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                command.quit = True
            if event.type == pygame.KEYDOWN:
                command = self.handler_fn(event, command)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.__handle_mouse_click()
        return command

    def __handle_replay_key_controls(self, event: Event, command: Command) -> Command:
        """
        Handle key controls for replay mode.
        Control game speed, pause, and replay frames.
        """
        if event.key == Q:
            command.quit = True
        elif event.key == RIGHT:
            command.speed_change = 2.0
        elif event.key == LEFT:
            command.speed_change = 0.5
        elif event.key == SPACE:
            command.pause = True
        elif event.key == R:
            command.restart = True
            command.pause = True
        elif event.key == H:
            command.frame_change = -1
            command.pause = True
        elif event.key == L:
            command.frame_change = 1
            self.properties.paused = True
        return command

    def __handle_game_key_controls(
        self, event: Event, command: Command
    ) -> dict[str, any]:
        raise NotImplementedError

    def __handle_train_key_controls(
        self, event: Event, command: Command
    ) -> dict[str, any]:
        if event.key == Q:
            command.quit = True
        return command

    def __handle_mouse_click(self):
        """
        Handle mouse click event.
        """
        if self.properties.mode == "replay":
            self.__handle_replay_clicks()
        elif self.properties.mode == "game":
            self.__handle_game_clicks()
        elif self.properties.mode == "train":
            self.__handle_train_clicks()

    def __handle_game_clicks(self):
        """
        Handle mouse clicks in game mode.
        """
        pass

    def __handle_train_clicks(self):
        """
        Handle mouse clicks in training mode.
        """
        pass

    def __handle_replay_clicks(self):
        """
        Handle mouse clicks in replay mode.
        """
        agents = self.properties.game.agents
        agent_fov = self.properties.agent_fov

        x, y = pygame.mouse.get_pos()
        for i, agent in enumerate(agents):
            if self.is_click_on_agents_row(x, y, i):
                agent_fov[agent] = not agent_fov[agent]
                break

    def is_click_on_agents_row(self, x: int, y: int, i: int) -> bool:
        """
        Check if the click is on an agent's row.

        Args:
            x: int, x-coordinate of the click
            y: int, y-coordinate of the click
            i: int, index of the row
        """
        return (
            x >= self.properties.display_grid_width
            and (i + 1) * c.GUI_ROW_HEIGHT <= y < (i + 2) * c.GUI_ROW_HEIGHT
        )
