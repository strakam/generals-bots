from typing import Any

import pygame

from generals.core.channels import Channels

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
        channels: Channels,
        agent_ids: list[str],
        grid_dims: tuple[int, int],
        gui_mode: GuiMode = GuiMode.TRAIN,
        speed_multiplier: float = 1.0,
    ):
        pygame.init()
        pygame.display.set_caption("Generals")
        # Handle key repeats
        pygame.key.set_repeat(500, 64)

        self.properties = Properties(channels, agent_ids, grid_dims, gui_mode)
        self.renderer = Renderer(self.properties)
        self.event_handler = EventHandler.from_mode(self.properties.gui_mode, self.properties)

    def tick(self, agent_id_to_infos: dict[str, Any], current_time: int, fps: int | None = None) -> Command:
        command = self.event_handler.handle_events()

        if command.quit:
            quit()

        if isinstance(command, ReplayCommand):
            self.properties.update_speed(command.speed_change)
            if command.frame_change != 0 or command.restart:
                self.properties.is_paused = True
            if command.pause_toggle:
                self.properties.is_paused = not self.properties.is_paused

        self.renderer.render(agent_id_to_infos, current_time, fps)

        return command

    def close(self):
        pygame.quit()
