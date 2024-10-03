import pygame
from typing import Any, Literal

from generals.core.game import Game
from .properties import Properties
from .event_handler import (
    TrainEventHandler,
    GameEventHandler,
    ReplayEventHandler,
    ReplayCommand,
    Command,
)
from .rendering import Renderer


class GUI:
    def __init__(
        self,
        game: Game,
        agent_data: dict[str, dict[str, Any]],
        mode: Literal["train", "game", "replay"] = "train",
    ):
        pygame.init()
        pygame.display.set_caption("Generals")

        # Handle key repeats
        pygame.key.set_repeat(500, 64)

        self.properties = Properties(game, agent_data, mode)
        self.__renderer = Renderer(self.properties)
        self.__event_handler = self.__initialize_event_handler()

    def __initialize_event_handler(self):
        if self.properties.mode == "train":
            return TrainEventHandler(self.properties)
        elif self.properties.mode == "game":
            return GameEventHandler(self.properties)
        elif self.properties.mode == "replay":
            return ReplayEventHandler(self.properties)

    def tick(self, fps=None) -> Command:
        command = self.__event_handler.handle_events()
        if command.quit:
            quit()
        if isinstance(command, ReplayCommand):
            self.properties.update_speed(command.speed_change)
            if command.frame_change != 0 or command.restart:
                self.properties.paused = True
            if command.pause_toggle:
                self.properties.paused = not self.properties.paused
        self.__renderer.render(fps)
        return command

    def close(self):
        pygame.quit()
