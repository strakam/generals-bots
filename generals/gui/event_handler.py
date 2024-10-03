import pygame
from pygame.event import Event
from abc import abstractmethod

from .properties import Properties
from generals.core import config as c

# keybindings #
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
        self.pause_toggle: bool = False


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

    def handle_events(self) -> Command:
        """
        Handle pygame GUI events
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.command.quit = True
            if event.type == pygame.KEYDOWN:
                self.handle_key_event(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_event()
        return self.command

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

    def toggle_player_fov(self):
        agents = self.properties.game.agents
        agent_fov = self.properties.agent_fov

        x, y = pygame.mouse.get_pos()
        for i, agent in enumerate(agents):
            if self.is_click_on_agents_row(x, y, i):
                agent_fov[agent] = not agent_fov[agent]
                break

    @abstractmethod
    def handle_key_event(self, event: Event) -> Command:
        raise NotImplementedError

    @abstractmethod
    def handle_mouse_event(self):
        raise NotImplementedError


class ReplayEventHandler(EventHandler):
    def __init__(self, properties: Properties):
        super().__init__(properties)
        self.command = ReplayCommand()

    def handle_key_event(self, event: Event) -> ReplayCommand:
        if event.key == Q:
            self.command.quit = True
        elif event.key == RIGHT:
            self.command.speed_change = 2.0
        elif event.key == LEFT:
            self.command.speed_change = 0.5
        elif event.key == SPACE:
            self.command.pause_toggle = True
        elif event.key == R:
            self.command.restart = True
        elif event.key == H:
            self.command.frame_change = -1
        elif event.key == L:
            self.command.frame_change = 1
        return self.command

    def handle_mouse_event(self) -> None:
        """
        Handle mouse clicks in replay mode.
        """
        self.toggle_player_fov()


class GameEventHandler(EventHandler):
    def __init__(self, properties: Properties):
        super().__init__(properties)
        self.command = GameCommand()

    def handle_key_event(self, event: Event) -> GameCommand:
        raise NotImplementedError

    def handle_mouse_event(self) -> None:
        self.toggle_player_fov()


class TrainEventHandler(EventHandler):
    def __init__(self, properties: Properties):
        super().__init__(properties)
        self.command = TrainCommand()

    def handle_key_event(self, event: Event) -> TrainCommand:
        if event.key == Q:
            self.command.quit = True
        return self.command

    def handle_mouse_event(self) -> None:
        self.toggle_player_fov()
