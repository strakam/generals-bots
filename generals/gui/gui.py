from typing import Any

from generals.core.game import Game
from .properties import Properties
from .event_handler import EventHandler
from .rendering import Renderer


class GUI:
    def __init__(
            self, game: Game, agent_data: dict[str, dict[str, Any]], from_replay=False
    ):
        self.properties = Properties(game, agent_data)
        self.__renderer = Renderer(self.properties)
        self.__event_handler = EventHandler(self.properties, from_replay)

    def tick(self, fps=None):
        control_events = self.__event_handler.handle_events()
        self.__renderer.render(fps)
        return control_events
