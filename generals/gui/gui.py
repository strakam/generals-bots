from typing import Any, Literal

from generals.core.game import Game
from .properties import Properties
from .event_handler import EventHandler
from .rendering import Renderer


class GUI:
    def __init__(
        self,
        game: Game,
        agent_data: dict[str, dict[str, Any]],
        mode: Literal["train", "game", "replay"] = "train",
    ):
        self.properties = Properties(game, agent_data, mode)
        self.__renderer = Renderer(self.properties)
        self.__event_handler = EventHandler(self.properties)

    def tick(self, fps=None):
        command = self.__event_handler.handle_events()
        if self.properties.mode == "replay":
            command = self.__event_handler.handle_replay_command(command)
            self.properties.update_speed(command.speed_change)
            self.properties.paused = command.pause
        self.__renderer.render(fps)
        return command
