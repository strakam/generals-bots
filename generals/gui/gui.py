from typing import Any

import pygame

from generals.core.game import Game

from .event_handler import (
    Command,
    EventHandler,
    ReplayCommand,
)
from .properties import GuiMode, Properties
from .rendering import Renderer


class GUI:
    def __init__(
        self,
        game: Game,
        agent_data: dict[str, dict[str, Any]],
        mode: GuiMode = GuiMode.TRAIN,
    ):
        pygame.init()
        pygame.display.set_caption("Generals")

        # Handle key repeats
        pygame.key.set_repeat(500, 64)

        self.properties = Properties(game, agent_data, mode)
        self.__renderer = Renderer(self.properties)
        self.__event_handler = EventHandler.from_mode(self.properties.mode, self.properties)

    def tick(self, fps: int | None = None) -> Command:
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
